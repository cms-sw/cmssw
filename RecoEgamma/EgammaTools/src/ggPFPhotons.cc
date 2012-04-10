#include "RecoEgamma/EgammaTools/interface/ggPFPhotons.h"
#include "DataFormats/Math/interface/deltaPhi.h"
ggPFPhotons::ggPFPhotons(
			 reco::Photon phot,
			 edm::Handle<reco::PhotonCollection>& pfPhotons,
			 edm::Handle<reco::GsfElectronCollection>& pfElectrons,
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
      
  } 

    
}
ggPFPhotons::~ggPFPhotons(){;}

//get Vtx Z along beam line from Single Leg Pointing if Track exists
//else it returns Vtx Z from Conversion Pair or Beamspot Z
std::pair<float, float> ggPFPhotons::SLPoint(){
  ggPFTracks pfTks(beamSpotHandle_);
  std::pair<float, float> SLPoint(0,0);
  TVector3 bs(beamSpotHandle_->position().x(),beamSpotHandle_->position().y(),
	      beamSpotHandle_->position().z());
  if(isPFEle_){
    isConv_=true;
    SLPoint=pfTks.gsfTrackProj(PFElectron_.gsfTrack());
    return SLPoint;
  }
  SLPoint=pfTks.SLCombZVtx(PFPhoton_, hasSLConv_);
  isConv_=pfTks.isConv();//bool to flag if there are conv tracks
  return SLPoint; 
}

void ggPFPhotons::fillPFClusters(){
  //PFClusterCollection object with appropriate variables:
  ggPFClusters PFClusterCollection(PFPhoton_, EBReducedRecHits_, EEReducedRecHits_, geomBar_,   geomEnd_);
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
  Must.MustacheID(PFClusters_, insideMust, outsideMust);
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
  ggPFESClusters PFPSClusterCollection(ESRecHits_);
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
  //Local Coordinates 
  
  //Shower Shape
  
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


