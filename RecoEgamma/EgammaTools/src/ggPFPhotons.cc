#include "RecoEgamma/EgammaTools/interface/ggPFPhotons.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
/*
Class by Rishi Patel rpatel@cern.ch
*/
ggPFPhotons::ggPFPhotons(
			 reco::Photon phot,
			 edm::Handle<reco::PhotonCollection>& pfPhotons,
			 edm::Handle<reco::GsfElectronCollection>& pfElectrons,
			 edm::Handle<PFCandidateCollection>& pfCandidates,
			 edm::Handle<EcalRecHitCollection>& EBReducedRecHits,
			 edm::Handle<EcalRecHitCollection>& EEReducedRecHits,
			 edm::Handle<EcalRecHitCollection>& ESRecHits,
			 const CaloSubdetectorGeometry* geomBar,
			 const CaloSubdetectorGeometry* geomEnd,
			 edm::Handle<BeamSpot>& beamSpotHandle
			 ):
  matchedPhot_(phot),
  pfPhotons_(pfPhotons),
  EBReducedRecHits_(EBReducedRecHits),
  EEReducedRecHits_(EEReducedRecHits),
  ESRecHits_(ESRecHits),
  geomBar_(geomBar),
  geomEnd_(geomEnd),
  beamSpotHandle_(beamSpotHandle),
  matchPFReco_(false),
  isPFEle_(false),
  EleVeto_(false),
  isConv_(false),
  hasSLConv_(false)
{
  //do Matching:
  //std::cout<<"HERE IN NEW CLASS "<<std::endl;
  
  reco::PhotonCollection::const_iterator pfphot=pfPhotons_->begin();
  for(;pfphot!=pfPhotons_->end();++pfphot){
    if(pfphot->superCluster()==matchedPhot_.superCluster()){
      PFPhoton_= *(pfphot);
      matchPFReco_=true;
      break;
    }
  }
  reco::GsfElectronCollection::const_iterator pfele=pfElectrons->begin();
  for(;pfele!=pfElectrons->end();++pfele){
    if(pfele->superCluster().isNull())continue;
    
    if(pfele->superCluster()==matchedPhot_.superCluster()){
      if(pfele->pflowSuperCluster().isNull())continue;
      
      PFElectron_= *(pfele);
      
      matchPFReco_=true;
      isPFEle_=true;
      break;
    }
    for(PFCandidateCollection::const_iterator pfParticle =pfCandidates->begin(); pfParticle!=pfCandidates->end(); pfParticle++){
      if(abs(pfParticle->pdgId())!=11)continue;
      if(pfParticle->superClusterRef().isNull())continue;
      double dR=deltaR(pfParticle->superClusterRef()->eta(), 
		       pfParticle->superClusterRef()->phi(),
		       matchedPhot_.superCluster()->eta(), 
		       matchedPhot_.superCluster()->phi());
      if(pfele->superCluster()==pfParticle->superClusterRef() &&dR<0.1){
	PFElectron_= *(pfele);
	matchPFReco_=true;
	isPFEle_=true;
	break;
      }	
    }  
  } 

}
ggPFPhotons::~ggPFPhotons(){;}
//Prompt Electron Veto:
bool ggPFPhotons::PFElectronVeto(edm::Handle<reco::ConversionCollection>& convH, edm::Handle<reco::GsfElectronCollection>& gsfElectronsHandle){
  bool isprompt=false;
  if(!isPFEle_)return isprompt;
  isprompt= ConversionTools::hasMatchedPromptElectron(PFElectron_.superCluster(), gsfElectronsHandle, convH, beamSpotHandle_->position());
  return isprompt;
}
//get Vtx Z along beam line from Single Leg Pointing if Track exists
//else it returns Vtx Z from Conversion Pair or Beamspot Z
std::pair<float, float> ggPFPhotons::SLPoint(){
  ggPFTracks pfTks(beamSpotHandle_);
  std::pair<float, float> SLPoint(0,0);
  TVector3 bs(beamSpotHandle_->position().x(),beamSpotHandle_->position().y(),
	      beamSpotHandle_->position().z());
  if(isPFEle_){
    isConv_=true;
    SLPoint=pfTks.gsfElectronProj(PFElectron_);
    return SLPoint;
  }
  SLPoint=pfTks.SLCombZVtx(PFPhoton_, hasSLConv_);
  isConv_=pfTks.isConv();//bool to flag if there are conv tracks
  return SLPoint; 
}

void ggPFPhotons::fillPFClusters(){
  //PFClusterCollection object with appropriate variables:
  
  ggPFClusters PFClusterCollection(EBReducedRecHits_, EEReducedRecHits_, geomBar_,   geomEnd_);
  
  //fill PFClusters
  if(isPFEle_)PFClusters_=PFClusterCollection.getPFClusters(*(PFElectron_.pflowSuperCluster()));
  else PFClusters_=PFClusterCollection.getPFClusters(*(PFPhoton_.pfSuperCluster()));
  //fill PFClusters with Cluster energy from Rechits inside SC
  
  PFSCFootprintClusters_.clear();
  for(unsigned int i=0; i<PFClusters_.size(); ++i){
    float SCFootPrintE=PFClusterCollection.getPFSuperclusterOverlap(PFClusters_[i],matchedPhot_);
    CaloCluster calo(SCFootPrintE, PFClusters_[i].position());
    PFSCFootprintClusters_.push_back(calo);
  }
  //Mustache Variables:
  
  
  Mustache Must;
  std::vector<unsigned int> insideMust;
  std::vector<unsigned int> outsideMust;
  Must.MustacheClust(PFClusters_, insideMust, outsideMust);
 
  //Must.MustacheID(PFClusters_, insideMust, outsideMust);
  // cout<<"Inside "<<insideMust.size()<<", Total"<<PFClusters_.size()<<endl;
  //sum MustacheEnergy and order clusters by Energy:
  
  std::multimap<float, unsigned int>OrderedClust;
  float MustacheE=0;
  for(unsigned int i=0; i<insideMust.size(); ++i){
    unsigned int index=insideMust[i];
    OrderedClust.insert(make_pair(PFClusters_[index].energy(), index));
    MustacheE=MustacheE+PFSCFootprintClusters_[index].energy();
  }
  
  float MustacheEOut=0;
  float MustacheEtOut=0;
  for(unsigned int i=0; i<outsideMust.size(); ++i){
    unsigned int index=outsideMust[i];
    MustacheEOut=MustacheEOut+PFClusters_[index].energy();
    MustacheEtOut=MustacheEtOut+PFClusters_[index].energy()*sin(PFClusters_[index].position().theta());
  }
  MustacheEOut_=MustacheEOut;
  MustacheEtOut_=MustacheEtOut;
  EinMustache_=MustacheE;
  
  //find lowest energy Cluster
  std::multimap<float, unsigned int>::iterator it;
  it=OrderedClust.begin();
  unsigned int lowEindex=(*it).second;   
  PFLowClusE_=PFSCFootprintClusters_[lowEindex].energy();
  
  //dEta and dPhi to this Cluster:
  dEtaLowestC_=PFSCFootprintClusters_[lowEindex].eta()-PFPhoton_.eta();
  dPhiLowestC_=deltaPhi(PFSCFootprintClusters_[lowEindex].phi(),PFPhoton_.phi());
  //RMS Of All PFClusters inside SuperCluster:
  
  std::pair<double, double> RMS=CalcRMS(PFSCFootprintClusters_, PFPhoton_);
  PFClPhiRMS_=RMS.second;
  std::vector<CaloCluster>MustacheNLClust;
  for(it=OrderedClust.begin(); it!=OrderedClust.end(); ++it){
    unsigned int index=(*it).second;
    if(index==lowEindex)continue; //skip lowest cluster which could be from PU
    MustacheNLClust.push_back(PFSCFootprintClusters_[index]);
  }
  if(insideMust.size()>3){
  std::pair<double, double> RMSMust=CalcRMS(MustacheNLClust, PFPhoton_);
  PFClPhiRMSMust_=RMSMust.second;
  }
  if(insideMust.size()==2){
    MustacheNLClust.push_back(PFSCFootprintClusters_[lowEindex]);
    std::pair<double, double> RMSMust=CalcRMS(MustacheNLClust, PFPhoton_);
    PFClPhiRMSMust_=RMSMust.second;
  }
  if(insideMust.size()==1){
    PFClPhiRMSMust_=matchedPhot_.superCluster()->phiWidth();
    PFClPhiRMS_=PFClPhiRMSMust_;
  }
  
  //fill ES Clusters
  
  ggPFESClusters PFPSClusterCollection(ESRecHits_, geomEnd_);
  vector<reco::PreshowerCluster>PFPS;
  if(isPFEle_)PFPS=PFPSClusterCollection.getPFESClusters(*((PFElectron_.pflowSuperCluster())));
  else PFPS=PFPSClusterCollection.getPFESClusters(*(PFPhoton_.pfSuperCluster()));
  float PFPS1=0;
  float PFPS2=0;
  for(unsigned int i=0; i<PFPS.size(); ++i){
    if(PFPS[i].plane()==1)PFPS1=PFPS1+PFPS[i].energy();
    if(PFPS[i].plane()==2)PFPS2=PFPS2+PFPS[i].energy();
  }
  PFPreShower1_=PFPS1;
  PFPreShower2_=PFPS2;
  
}

std::pair<double, double> ggPFPhotons::CalcRMS(vector<reco::CaloCluster> PFClust, reco::Photon PFPhoton){
  double delPhi2=0;
  double delPhiSum=0;
  double delEta2=0;
  double delEtaSum=0;
  double ClusSum=0;
  float PFPhoPhi=PFPhoton.phi();
  float PFPhoEta=PFPhoton.eta();

  float PFClusPhiRMS=0;
  float PFClusEtaRMS=0;
  std::pair<double, double> RMS;  
  for(unsigned int c=0; c<PFClust.size(); ++c){
    delPhi2=(acos(cos(PFPhoPhi-PFClust[c].phi()))* acos(cos(PFPhoPhi-PFClust[c].phi())) )+delPhi2;
    delPhiSum=delPhiSum+ acos(cos(PFPhoPhi-PFClust[c].phi()))*PFClust[c].energy();
    delEta2=(PFPhoEta-PFClust[c].eta())*(PFPhoEta-PFClust[c].eta()) *PFClust[c].energy() + delEta2;
    delEtaSum=delEtaSum+((PFPhoEta-PFClust[c].eta())*PFClust[c].energy());
    ClusSum=ClusSum+PFClust[c].energy();
  }
  double meandPhi=delPhiSum/ClusSum;
  PFClusPhiRMS=sqrt(fabs(delPhi2/ClusSum - (meandPhi*meandPhi)));
  double meandEta=delEtaSum/ClusSum;
  PFClusEtaRMS=sqrt(fabs(delEta2/ClusSum - (meandEta*meandEta)));
  RMS.first=PFClusEtaRMS;
  RMS.second=PFClusPhiRMS;
  return RMS;
}

double ggPFPhotons::getPFPhoECorr( std::vector<reco::CaloCluster>PFClusters, const GBRForest *ReaderLCEB, const GBRForest *ReaderLCEE){
  TVector3 bs(beamSpotHandle_->position().x(),beamSpotHandle_->position().y(),
	      beamSpotHandle_->position().z());
  float beamspotZ=bs.Z();
  //PFClusterCollection object with appropriate variables:
  ggPFClusters PFClusterCollection(EBReducedRecHits_, EEReducedRecHits_, geomBar_,   geomEnd_);
  //fill PFClusters
  PFClusters_.clear();
  if(isPFEle_)PFClusters_=PFClusterCollection.getPFClusters(*(PFElectron_.pflowSuperCluster()));
  else PFClusters_=PFClusterCollection.getPFClusters(*(PFPhoton_.pfSuperCluster()));
  Mustache Must;
  std::vector<unsigned int> insideMust;
  std::vector<unsigned int> outsideMust;
  Must.MustacheClust(PFClusters_, insideMust, outsideMust);
  float ECorr=0;
  for(unsigned int i=0; i<insideMust.size();++i){
    unsigned int index=insideMust[i];
    ECorr=ECorr+PFClusterCollection.LocalEnergyCorrection(ReaderLCEB, ReaderLCEE, PFClusters_[index], beamspotZ);
  }
  PFPhoLocallyCorrE_=ECorr;
  
  return PFPhoLocallyCorrE_;
}

std::vector<reco::CaloCluster> ggPFPhotons::recoPhotonClusterLink(reco::Photon phot, 
								 edm::Handle<PFCandidateCollection>& pfCandidates
								 ){
  PFClusters_.clear();
  //initialize variables:
  EinMustache_=0;
  MustacheEOut_=0;
  MustacheEtOut_=0;
  PFPreShower1_=0;
  PFPreShower2_=0;
  PFLowClusE_=0;
  dEtaLowestC_=0;
  dPhiLowestC_=0;
  PFClPhiRMS_=0;
  PFClPhiRMSMust_=0;
  //find SC boundaries from RecHits:
  std::pair<double,double>Boundary=SuperClusterSize(phot);		
  double etabound=Boundary.first;
  double phibound=Boundary.second;
  double seedEta=phot.superCluster()->seed()->eta();
  double seedPhi=phot.superCluster()->seed()->phi();
  for (PFCandidateCollection::const_iterator pfParticle =pfCandidates->begin(); pfParticle!=pfCandidates->end(); pfParticle++){
    if(pfParticle->pdgId()!=22) continue; //if photon
    //use a box about superCluster Width:
    float dphi=acos(cos(seedPhi-pfParticle->phi()));
    float deta=fabs(seedEta-pfParticle->eta());
    PFPreShower1_=0;
    PFPreShower2_=0;
    if(deta<etabound && dphi<phibound){
      //hard coded size for now but will make it more dynamic
      //based on SuperCluster Shape
      math::XYZPoint position(pfParticle->positionAtECALEntrance().X(), pfParticle->positionAtECALEntrance().Y(), pfParticle->positionAtECALEntrance().Z()) ;
      CaloCluster calo(pfParticle->rawEcalEnergy() ,position );
      PFClusters_.push_back(calo);
      PFPreShower1_=PFPreShower1_+pfParticle->pS1Energy();
      PFPreShower2_=PFPreShower2_+pfParticle->pS2Energy();
    }
  }
 
    //Mustache Variables:
  
  if(PFClusters_.size()>0){
  Mustache Must;
  std::vector<unsigned int> insideMust;
  std::vector<unsigned int> outsideMust;
  Must.MustacheClust(PFClusters_, insideMust, outsideMust);
  //sum MustacheEnergy and order clusters by Energy:
  
  std::multimap<float, unsigned int>OrderedClust;
  float MustacheE=0;
  for(unsigned int i=0; i<insideMust.size(); ++i){
    unsigned int index=insideMust[i];
    OrderedClust.insert(make_pair(PFClusters_[index].energy(), index));
    MustacheE=MustacheE+PFClusters_[index].energy();
  }
  
  float MustacheEOut=0;
  float MustacheEtOut=0;
  for(unsigned int i=0; i<outsideMust.size(); ++i){
    unsigned int index=outsideMust[i];
    MustacheEOut=MustacheEOut+PFClusters_[index].energy();
    MustacheEtOut=MustacheEtOut+PFClusters_[index].energy()*sin(PFClusters_[index].position().theta());
  }
  MustacheEOut_=MustacheEOut;
  MustacheEtOut_=MustacheEtOut;
  EinMustache_=MustacheE;
  
  //find lowest energy Cluster
  std::multimap<float, unsigned int>::iterator it;
  it=OrderedClust.begin();
  unsigned int lowEindex=(*it).second;   
  PFLowClusE_=PFClusters_[lowEindex].energy();
  //dEta and dPhi to this Cluster:
  dEtaLowestC_=PFClusters_[lowEindex].eta()-PFPhoton_.eta();
  dPhiLowestC_=deltaPhi(PFClusters_[lowEindex].phi(),PFPhoton_.phi());
  //RMS Of All PFClusters inside SuperCluster:
  if(insideMust.size()==1){
    PFLowClusE_=0;
    dEtaLowestC_=0;
    dPhiLowestC_=0;
  }
  std::pair<double, double> RMS=CalcRMS(PFClusters_, PFPhoton_);
  PFClPhiRMS_=RMS.second;
  std::vector<CaloCluster>MustacheNLClust;
  for(it=OrderedClust.begin(); it!=OrderedClust.end(); ++it){
    unsigned int index=(*it).second;
    if(index==lowEindex)continue; //skip lowest cluster which could be from PU
    MustacheNLClust.push_back(PFClusters_[index]);
  }
  if(insideMust.size()>3){
  std::pair<double, double> RMSMust=CalcRMS(MustacheNLClust, PFPhoton_);
  PFClPhiRMSMust_=RMSMust.second;
  }
  if(insideMust.size()==2){
    MustacheNLClust.push_back(PFClusters_[lowEindex]);
    std::pair<double, double> RMSMust=CalcRMS(MustacheNLClust, PFPhoton_);
    PFClPhiRMSMust_=RMSMust.second;
  }
  if(insideMust.size()==1){
    PFClPhiRMSMust_=matchedPhot_.superCluster()->phiWidth();
    PFClPhiRMS_=PFClPhiRMSMust_;
  }
  
  }
  return PFClusters_;
  
}

std::pair<double, double>ggPFPhotons::SuperClusterSize(
						       reco::Photon phot
						       ){
  std::pair<double, double>SCsize(0.1,0.4);//Eta, Phi
  //find maximum distance between SuperCluster Seed and Rec Hits
  SuperClusterRef recoSC=phot.superCluster();
  reco::CaloCluster_iterator cit=recoSC->clustersBegin();
  double seedeta=recoSC->seed()->eta();
  double seedphi=recoSC->seed()->phi();
  EcalRecHitCollection::const_iterator eb;
  EcalRecHitCollection::const_iterator ee;
  double MaxEta=-99;
  double MaxPhi=-99;
  for(;cit!=recoSC->clustersEnd();++cit){
    std::vector< std::pair<DetId, float> >bcCells=(*cit)->hitsAndFractions();
    if(phot.isEB()){
      for(unsigned int i=0; i<bcCells.size(); ++i){
	for(eb=EBReducedRecHits_->begin();eb!= EBReducedRecHits_->end();++eb){
	  if(eb->id().rawId()==bcCells[i].first.rawId()){
	    DetId id=bcCells[i].first;
	    float eta=geomBar_->getGeometry(id)->getPosition().eta();
	    float dEta = fabs(seedeta-eta);
	    if(dEta>MaxEta){
	      MaxEta=dEta;
	      float phi=geomBar_->getGeometry(id)->getPosition().phi();
	      float dPhi = acos(cos( seedphi-phi));
	      if(dPhi>MaxPhi){
		MaxPhi=dPhi;
	      }
	    }
	  }
	}
	
      }
    }
    else{
      for(unsigned int i=0; i<bcCells.size(); ++i){
	for(ee=EEReducedRecHits_->begin();ee!= EEReducedRecHits_->end();++ee){
	  if(ee->id().rawId()==bcCells[i].first.rawId()){
	    DetId id=bcCells[i].first;
	    float eta=geomEnd_->getGeometry(id)->getPosition().eta();
	    float dEta = fabs(seedeta-eta);
	    if(dEta>MaxEta){
	      MaxEta=dEta;
	    }
	    float phi=geomEnd_->getGeometry(id)->getPosition().phi();
	    float dPhi = acos(cos(seedphi-phi));
	    if(dPhi>MaxPhi){
	      MaxPhi=dPhi;
	    }
	  }
	}
	
      }
    }
  }
  SCsize.first=MaxEta; SCsize.second=MaxPhi;
  return SCsize;
}

void ggPFPhotons::recoPhotonClusterLink(
					reco::SuperCluster sc, 
					std::vector<reco::PFCandidatePtr>&insideMust, 
					std::vector<reco::PFCandidatePtr>&outsideMust,
					edm::Handle<PFCandidateCollection>& pfCandidates,
					double etabound,
					double phibound
					
					
					){
  std::vector<reco::CaloCluster>PFClusters;
  std::vector<reco::PFCandidatePtr>PFCand;
  double seedEta=sc.seed()->eta();
  double seedPhi=sc.seed()->phi();
  for(PFCandidateCollection::const_iterator pfParticle =pfCandidates->begin(); pfParticle!=pfCandidates->end(); pfParticle++){
    unsigned int index=pfParticle - pfCandidates->begin();
    if(pfParticle->pdgId()!=22) continue; //if photon
    //use a box about superCluster Width:
    float dphi=acos(cos(seedPhi-pfParticle->phi()));
    float deta=fabs(seedEta-pfParticle->eta());
    if(deta<etabound && dphi<phibound){
      //hard coded size for now but will make it more dynamic
      //based on SuperCluster Shape
      math::XYZPoint position(pfParticle->positionAtECALEntrance().X(), pfParticle->positionAtECALEntrance().Y(), pfParticle->positionAtECALEntrance().Z()) ;
      CaloCluster calo(pfParticle->rawEcalEnergy() ,position );
      PFClusters.push_back(calo);
      reco::PFCandidatePtr pfRef(pfCandidates,index);
      PFCand.push_back(pfRef);
    }
    
  }
  Mustache Must;
  std::vector<unsigned int> insideMustindex;
  std::vector<unsigned int> outsideMustindex;
  Must.MustacheClust(PFClusters, insideMustindex, outsideMustindex);
  for(unsigned int i=0; i<insideMustindex.size(); ++i){
    unsigned int index=insideMustindex[i];
    insideMust.push_back(PFCand[index]);
  }
  for(unsigned int i=0; i<outsideMustindex.size(); ++i){
    unsigned int index=outsideMustindex[i];
    outsideMust.push_back(PFCand[index]);
  }
    
}
std::pair<double, double>ggPFPhotons::SuperClusterSize(reco::SuperCluster sc,
						 Handle<EcalRecHitCollection>&   EBReducedRecHits,
						 Handle<EcalRecHitCollection>&   EEReducedRecHits,
						 const CaloSubdetectorGeometry* geomBar,
						 const CaloSubdetectorGeometry* geomEnd
						 ){
  std::pair<double, double>SCsize(0.1,0.4);//Eta, Phi
  //find maximum distance between SuperCluster Seed and Rec Hits
  reco::CaloCluster_iterator cit=sc.clustersBegin();
  double seedeta=sc.seed()->eta();
  double seedphi=sc.seed()->phi();
  EcalRecHitCollection::const_iterator eb;
  EcalRecHitCollection::const_iterator ee;
  double MaxEta=-99;
  double MaxPhi=-99;
  for(;cit!=sc.clustersEnd();++cit){
    std::vector< std::pair<DetId, float> >bcCells=(*cit)->hitsAndFractions();
    DetId seedXtalId = bcCells[0].first ;
    int detector = seedXtalId.subdetId(); //use Seed to check if EB or EE
    bool isEb;
    if(detector==1)isEb=true;
    else isEb=false;
    if(isEb){
      for(unsigned int i=0; i<bcCells.size(); ++i){
	for(eb=EBReducedRecHits->begin();eb!= EBReducedRecHits->end();++eb){
	  if(eb->id().rawId()==bcCells[i].first.rawId()){
	    DetId id=bcCells[i].first;
	    float eta=geomBar->getGeometry(id)->getPosition().eta();
	    float dEta = fabs(seedeta-eta);
	    if(dEta>MaxEta){
	      MaxEta=dEta;
	      float phi=geomBar->getGeometry(id)->getPosition().phi();
	      float dPhi = acos(cos( seedphi-phi));
	      if(dPhi>MaxPhi){
		MaxPhi=dPhi;
	      }
	    }
	  }
	}
	
      }
    }
    else{
      for(unsigned int i=0; i<bcCells.size(); ++i){
	for(ee=EEReducedRecHits->begin();ee!= EEReducedRecHits->end();++ee){
	  if(ee->id().rawId()==bcCells[i].first.rawId()){
	    DetId id=bcCells[i].first;
	    float eta=geomEnd->getGeometry(id)->getPosition().eta();
	    float dEta = fabs(seedeta-eta);
	    if(dEta>MaxEta){
	      MaxEta=dEta;
	    }
	    float phi=geomEnd->getGeometry(id)->getPosition().phi();
	    float dPhi = acos(cos(seedphi-phi));
	    if(dPhi>MaxPhi){
	      MaxPhi=dPhi;
	    }
	  }
	}
	
      }
    }
  }
  SCsize.first=MaxEta; SCsize.second=MaxPhi;
  return SCsize;
  
  
}
