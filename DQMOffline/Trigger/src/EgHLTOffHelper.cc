#include "DQMOffline/Trigger/interface/EgHLTOffHelper.h"


#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTTrackIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"
#include "DQMOffline/Trigger/interface/EgHLTErrCodes.h"

#include <iostream>

using namespace egHLT;



OffHelper::~OffHelper()
{
  if(hltEleTrkIsolAlgo_) delete hltEleTrkIsolAlgo_;
  if(hltPhoTrkIsolAlgo_) delete hltPhoTrkIsolAlgo_;
}

void OffHelper::setup(const edm::ParameterSet& conf)
{

  ecalRecHitsEBTag_ = conf.getParameter<edm::InputTag>("BarrelRecHitCollection");
  ecalRecHitsEETag_ = conf.getParameter<edm::InputTag>("EndcapRecHitCollection");
  caloJetsTag_ = conf.getParameter<edm::InputTag>("CaloJetCollection");
  isolTrkTag_ = conf.getParameter<edm::InputTag>("IsolTrackCollection");
  hbheHitsTag_ = conf.getParameter<edm::InputTag>("HBHERecHitCollection");
  hfHitsTag_ = conf.getParameter<edm::InputTag>("HFRecHitCollection");
  electronsTag_ = conf.getParameter<edm::InputTag>("ElectronCollection");
  photonsTag_ = conf.getParameter<edm::InputTag>("PhotonCollection");
  triggerSummaryLabel_ = conf.getParameter<edm::InputTag>("triggerSummaryLabel");
  hltTag_ = conf.getParameter<std::string>("hltTag");
  beamSpotTag_ = conf.getParameter<edm::InputTag>("BeamSpotProducer");
  caloTowersTag_ = conf.getParameter<edm::InputTag>("CaloTowers");
  trigResultsTag_ = conf.getParameter<edm::InputTag>("TrigResults");
  vertexTag_ = conf.getParameter<edm::InputTag>("VertexCollection");

  eleCuts_.setup(conf.getParameter<edm::ParameterSet>("eleCuts"));
  eleLooseCuts_.setup(conf.getParameter<edm::ParameterSet>("eleLooseCuts"));
  phoCuts_.setup(conf.getParameter<edm::ParameterSet>("phoCuts"));
  phoLooseCuts_.setup(conf.getParameter<edm::ParameterSet>("phoLooseCuts"));
 
  //now we have the isolations completely configurable via python
  hltEMIsolOuterCone_ = conf.getParameter<double>("hltEMIsolOuterCone");
  hltEMIsolInnerConeEB_ = conf.getParameter<double>("hltEMIsolInnerConeEB");
  hltEMIsolEtaSliceEB_ = conf.getParameter<double>("hltEMIsolEtaSliceEB");
  hltEMIsolEtMinEB_ = conf.getParameter<double>("hltEMIsolEtMinEB");
  hltEMIsolEMinEB_ = conf.getParameter<double>("hltEMIsolEMinEB");
  hltEMIsolInnerConeEE_ = conf.getParameter<double>("hltEMIsolInnerConeEE");
  hltEMIsolEtaSliceEE_ = conf.getParameter<double>("hltEMIsolEtaSliceEE");
  hltEMIsolEtMinEE_ = conf.getParameter<double>("hltEMIsolEtMinEE");
  hltEMIsolEMinEE_ = conf.getParameter<double>("hltEMIsolEMinEE");

  hltPhoTrkIsolPtMin_ = conf.getParameter<double>("hltPhoTrkIsolPtMin");
  hltPhoTrkIsolOuterCone_ = conf.getParameter<double>("hltPhoTrkIsolOuterCone");
  hltPhoTrkIsolInnerCone_ = conf.getParameter<double>("hltPhoTrkIsolInnerCone"); 
  hltPhoTrkIsolZSpan_ = conf.getParameter<double>("hltPhoTrkIsolZSpan");
  hltPhoTrkIsolRSpan_ = conf.getParameter<double>("hltPhoTrkIsolZSpan");
  hltPhoTrkIsolCountTrks_ = conf.getParameter<bool>("hltPhoTrkIsolCountTrks");

  hltEleTrkIsolPtMin_ = conf.getParameter<double>("hltEleTrkIsolPtMin");
  hltEleTrkIsolOuterCone_ = conf.getParameter<double>("hltEleTrkIsolOuterCone");
  hltEleTrkIsolInnerCone_ = conf.getParameter<double>("hltEleTrkIsolInnerCone"); 
  hltEleTrkIsolZSpan_ = conf.getParameter<double>("hltEleTrkIsolZSpan");
  hltEleTrkIsolRSpan_ = conf.getParameter<double>("hltEleTrkIsolZSpan");

  hltHadIsolOuterCone_ = conf.getParameter<double>("hltHadIsolOuterCone");
  hltHadIsolInnerCone_ = conf.getParameter<double>("hltHadIsolInnerCone");
  hltHadIsolEtMin_ = conf.getParameter<double>("hltHadIsolEtMin");
  hltHadIsolDepth_ = conf.getParameter<int>("hltHadIsolDepth"); 

  calHLTHcalIsol_ = conf.getParameter<bool>("calHLTHcalIsol");
  calHLTEmIsol_ = conf.getParameter<bool>("calHLTEmIsol");
  calHLTEleTrkIsol_ = conf.getParameter<bool>("calHLTEleTrkIsol");
  calHLTPhoTrkIsol_ = conf.getParameter<bool>("calHLTPhoTrkIsol");
 
  trigCutParams_ = conf.getParameter<std::vector<edm::ParameterSet> >("triggerCuts"); //setupTriggers used to be in this function but had to be moved due to HLTConfigChanges (has to be called beginRun) so we have to save this for later.

  hltEleTrkIsolAlgo_ = new EgammaHLTTrackIsolation(hltEleTrkIsolPtMin_,hltEleTrkIsolOuterCone_,hltEleTrkIsolZSpan_,hltEleTrkIsolRSpan_,hltEleTrkIsolInnerCone_);
  hltPhoTrkIsolAlgo_ = new EgammaHLTTrackIsolation(hltPhoTrkIsolPtMin_,hltPhoTrkIsolOuterCone_,hltPhoTrkIsolZSpan_,hltPhoTrkIsolRSpan_,hltPhoTrkIsolInnerCone_);
 					       

}

//this code was taken out of OffHelper::setup due to HLTConfigProvider changes
//it still assumes that this is called only once
void OffHelper::setupTriggers(const HLTConfigProvider& hltConfig,const std::vector<std::string>& hltFiltersUsed)
{
  hltFiltersUsed_ = hltFiltersUsed; //expensive but only do this once and faster ways could make things less clear
  //now work out how many objects are requires to pass filter for it to accept
  hltFiltersUsedWithNrCandsCut_.clear();
  for(size_t filterNr=0;filterNr<hltFiltersUsed_.size();filterNr++){
    hltFiltersUsedWithNrCandsCut_.push_back(std::make_pair(hltFiltersUsed_[filterNr],egHLT::trigTools::getMinNrObjsRequiredByFilter(hltFiltersUsed_[filterNr])));
  }

  //now loading the cuts for every trigger into our vector which stores them
  //only load cuts for triggers that are in hltFiltersUsed
  
  for(size_t trigNr=0;trigNr<trigCutParams_.size();trigNr++) {
    std::string trigName = trigCutParams_[trigNr].getParameter<std::string>("trigName");
    if(std::find(hltFiltersUsed_.begin(),hltFiltersUsed_.end(),trigName)!=hltFiltersUsed_.end()){ //perhaps I should sort hltFiltersUsed_....
      trigCuts_.push_back(std::make_pair(TrigCodes::getCode(trigName),OffEgSel(trigCutParams_[trigNr])));
      //   std::cout<<trigName<<std::endl<<"between"<<std::endl<<trigCutParams_[trigNr]<<std::endl<<"after"<<std::endl;
    }
  }
  trigCutParams_.clear();//dont need it any more, get rid of it

  //to make my life difficult, the scaled l1 paths are special
  //and arent stored in trigger event
  //to I have to figure out the path, see if it passes
  //and then hunt down the l1 seed filter and use that to match to the pho/ele
  //matching on l1 seed filter is not enough as that will be passed for normal 
  //electron triggers even if pre-scale hasnt fired
  l1PreScaledFilters_.clear();
  l1PreScaledPaths_.clear();
  l1PreAndSeedFilters_.clear();
  for(size_t filterNr=0;filterNr<hltFiltersUsed_.size();filterNr++){
    if(hltFiltersUsed_[filterNr].find("hltPreL1")==0){ //l1 prescaled path
      l1PreScaledFilters_.push_back(hltFiltersUsed_[filterNr]);
    }
  }

  egHLT::trigTools::translateFiltersToPathNames(hltConfig,l1PreScaledFilters_,l1PreScaledPaths_);
  if(l1PreScaledPaths_.size()==l1PreScaledFilters_.size()){
    for(size_t pathNr=0;pathNr<l1PreScaledPaths_.size();pathNr++){
     
      std::string l1SeedFilter =egHLT::trigTools::getL1SeedFilterOfPath(hltConfig,l1PreScaledPaths_[pathNr]);
      //---Morse====
      //std::cout<<l1PreScaledFilters_[pathNr]<<"  "<<l1PreScaledPaths_[pathNr]<<"  "<<l1SeedFilter<<std::endl;
      //------------
      l1PreAndSeedFilters_.push_back(std::make_pair(l1PreScaledFilters_[pathNr],l1SeedFilter));
    }
  }
}

int OffHelper::makeOffEvt(const edm::Event& edmEvent,const edm::EventSetup& setup,egHLT::OffEvt& offEvent)
{
  offEvent.clear();
  int errCode=0; //excution stops as soon as an error is flagged
  if(errCode==0) errCode = getHandles(edmEvent,setup);
  if(errCode==0) errCode = fillOffEleVec(offEvent.eles());
  if(errCode==0) errCode = fillOffPhoVec(offEvent.phos());
  if(errCode==0) errCode = setTrigInfo(edmEvent, offEvent);
  if(errCode==0) offEvent.setJets(recoJets_);
  return errCode;
}


int OffHelper::getHandles(const edm::Event& event,const edm::EventSetup& setup)
{
  try { 
    setup.get<CaloGeometryRecord>().get(caloGeom_);
    setup.get<CaloTopologyRecord>().get(caloTopology_);
    //setup.get<EcalSeverityLevelAlgoRcd>().get(ecalSeverityLevel_);
  }catch(cms::Exception& iException){
    return errCodes::Geom;
  }
  try {
    setup.get<IdealMagneticFieldRecord>().get(magField_);
  }catch(cms::Exception& iException){
    return errCodes::MagField;
  }

  //get objects
  if(!getHandle(event,triggerSummaryLabel_,trigEvt_)) return errCodes::TrigEvent; //must have this, otherwise skip event
  if(!getHandle(event,trigResultsTag_,trigResults_)) return errCodes::TrigEvent; //re using bit to minimise bug fix code changes
  if(!getHandle(event,electronsTag_,recoEles_)) return errCodes::OffEle; //need for electrons
  if(!getHandle(event,photonsTag_, recoPhos_)) return errCodes::OffPho; //need for photons
  if(!getHandle(event,caloJetsTag_,recoJets_)) return errCodes::OffJet; //need for electrons and photons
  if(!getHandle(event,vertexTag_,recoVertices_)) return errCodes::OffVertex; //need for eff vs nVertex

  //need for HLT isolations (rec hits also need for sigmaIPhiIPhi (ele/pho) and r9 pho)
  if(!getHandle(event,ecalRecHitsEBTag_,ebRecHits_)) return errCodes::EBRecHits;
  if(!getHandle(event,ecalRecHitsEETag_,eeRecHits_)) return errCodes::EERecHits;
  if(!getHandle(event,isolTrkTag_,isolTrks_)) return errCodes::IsolTrks;
  if(!getHandle(event,hbheHitsTag_, hbheHits_)) return errCodes::HBHERecHits; //I dont think we need hbhe rec-hits any more
  if(!getHandle(event,hfHitsTag_, hfHits_)) return errCodes::HFRecHits;//I dont think we need hf rec-hits any more
  if(!getHandle(event,beamSpotTag_,beamSpot_)) return errCodes::BeamSpot;
  if(!getHandle(event,caloTowersTag_,caloTowers_)) return errCodes::CaloTowers;

  
  return 0;
}

//this function coverts GsfElectrons to a format which is actually useful to me
int OffHelper::fillOffEleVec(std::vector<OffEle>& egHLTOffEles)
{
  egHLTOffEles.clear();
  egHLTOffEles.reserve(recoEles_->size());
  for(reco::GsfElectronCollection::const_iterator gsfIter=recoEles_->begin(); gsfIter!=recoEles_->end();++gsfIter){
    if(!gsfIter->ecalDrivenSeed()) continue; //avoid PF electrons (this is Eg HLT validation and HLT is ecal driven)

    int nVertex=0;
    for(reco::VertexCollection::const_iterator nVit=recoVertices_->begin(); nVit!=recoVertices_->end();++nVit){
      if( !nVit->isFake() 
	  && nVit->ndof()>4  
	  && std::fabs( nVit->z()<24.0) 
	  && sqrt(nVit->x()*nVit->x() + nVit->y()*nVit->y())<2.0){nVertex++;}
    }
    //if(nVertex>20)std::cout<<"nVertex: "<<nVertex<<std::endl;
    OffEle::EventData eventData;
    eventData.NVertex=nVertex;

    OffEle::IsolData isolData;   
    fillIsolData(*gsfIter,isolData);
    
    OffEle::ClusShapeData clusShapeData;
    fillClusShapeData(*gsfIter,clusShapeData);

    OffEle::HLTData hltData;
    fillHLTData(*gsfIter,hltData);

    egHLTOffEles.push_back(OffEle(*gsfIter,clusShapeData,isolData,hltData,eventData));
    
    //now we would like to set the cut results
    OffEle& ele =  egHLTOffEles.back();
    ele.setCutCode(eleCuts_.getCutCode(ele));
    ele.setLooseCutCode(eleLooseCuts_.getCutCode(ele));
    
    std::vector<std::pair<TrigCodes::TrigBitSet,int> >trigCutsCutCodes;
    for(size_t i=0;i<trigCuts_.size();i++) trigCutsCutCodes.push_back(std::make_pair(trigCuts_[i].first,trigCuts_[i].second.getCutCode(ele)));
    ele.setTrigCutsCutCodes(trigCutsCutCodes);
  }//end loop over gsf electron collection
  return 0;
}

void OffHelper::fillIsolData(const reco::GsfElectron& ele,OffEle::IsolData& isolData)
{
  EgammaTowerIsolation hcalIsolAlgo(hltHadIsolOuterCone_,hltHadIsolInnerCone_,hltHadIsolEtMin_,hltHadIsolDepth_,caloTowers_.product());
  EcalRecHitMetaCollection ebHits(*ebRecHits_);
  EcalRecHitMetaCollection eeHits(*eeRecHits_);
  EgammaRecHitIsolation ecalIsolAlgoEB(hltEMIsolOuterCone_,hltEMIsolInnerConeEB_,hltEMIsolEtaSliceEB_,
				       hltEMIsolEtMinEB_,hltEMIsolEMinEB_,caloGeom_,&ebHits,ecalSeverityLevel_.product(),DetId::Ecal);
  EgammaRecHitIsolation ecalIsolAlgoEE(hltEMIsolOuterCone_,hltEMIsolInnerConeEE_,hltEMIsolEtaSliceEE_,
				       hltEMIsolEtMinEE_,hltEMIsolEMinEE_,caloGeom_,&eeHits,ecalSeverityLevel_.product(),DetId::Ecal);
  
  isolData.ptTrks=ele.dr03TkSumPt();
  isolData.nrTrks=999; //no longer supported
  isolData.em= ele.dr03EcalRecHitSumEt();
  isolData.hadDepth1 =  ele.dr03HcalDepth1TowerSumEt();
  isolData.hadDepth2 =  ele.dr03HcalDepth2TowerSumEt();   

  //now time to do the HLT algos
  if(calHLTHcalIsol_) isolData.hltHad=hcalIsolAlgo.getTowerESum(&ele);
  else isolData.hltHad = 0.;
  if(calHLTEleTrkIsol_) isolData.hltTrksEle=hltEleTrkIsolAlgo_->electronPtSum(&(*(ele.gsfTrack())),isolTrks_.product());
  else isolData.hltTrksEle = 0.;
  if(calHLTPhoTrkIsol_){
    if(hltPhoTrkIsolCountTrks_) isolData.hltTrksPho=hltPhoTrkIsolAlgo_->photonTrackCount(&ele,isolTrks_.product(),false);
    else isolData.hltTrksPho=hltPhoTrkIsolAlgo_->photonPtSum(&ele,isolTrks_.product(),false);
  }
  else isolData.hltTrksPho = 0.;
  if(calHLTEmIsol_) isolData.hltEm = ecalIsolAlgoEB.getEtSum(&ele) + 
                                     ecalIsolAlgoEE.getEtSum(&ele);
  else isolData.hltEm = 0.;
  
}


void OffHelper::fillClusShapeData(const reco::GsfElectron& ele,OffEle::ClusShapeData& clusShapeData)
{
  clusShapeData.sigmaEtaEta = ele.sigmaEtaEta();
  clusShapeData.sigmaIEtaIEta = ele.sigmaIetaIeta();
  double e5x5 = ele.e5x5();
  if(e5x5!=0.){
    clusShapeData.e1x5Over5x5 = ele.e1x5()/e5x5;
    clusShapeData.e2x5MaxOver5x5 = ele.e2x5Max()/e5x5;
  }else{
    clusShapeData.e1x5Over5x5 = -1;
    clusShapeData.e2x5MaxOver5x5 = -1;
  }
  
  //want to calculate r9, sigmaPhiPhi and sigmaIPhiIPhi, have to do old fashioned way
  const reco::BasicCluster& seedClus = *(ele.superCluster()->seed());
  const DetId seedDetId = seedClus.hitsAndFractions()[0].first; //note this may not actually be the seed hit but it doesnt matter because all hits will be in the barrel OR endcap
  if(seedDetId.subdetId()==EcalBarrel){
    std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,ebRecHits_.product(),caloTopology_.product(),caloGeom_.product());
    std::vector<float> crysCov = EcalClusterTools::localCovariances(seedClus,ebRecHits_.product(),caloTopology_.product());
    clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
      clusShapeData.sigmaIPhiIPhi =  sqrt(crysCov[2]);
      if(ele.superCluster()->rawEnergy()!=0.){
	clusShapeData.r9 = EcalClusterTools::e3x3(seedClus,ebRecHits_.product(),caloTopology_.product()) / ele.superCluster()->rawEnergy();
      }else clusShapeData.r9 = -1.;
       
  }else{
    std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,eeRecHits_.product(),caloTopology_.product(),caloGeom_.product()); 
    std::vector<float> crysCov = EcalClusterTools::localCovariances(seedClus,eeRecHits_.product(),caloTopology_.product());
    clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
    clusShapeData.sigmaIPhiIPhi = sqrt(crysCov[2]);
    if(ele.superCluster()->rawEnergy()!=0.){
	clusShapeData.r9 = EcalClusterTools::e3x3(seedClus,eeRecHits_.product(),caloTopology_.product()) / ele.superCluster()->rawEnergy();
    }else clusShapeData.r9 = -1.;
  } 
}

//reco approximations of hlt quantities
void OffHelper::fillHLTData(const reco::GsfElectron& ele,OffEle::HLTData& hltData)
{
  if(ele.closestCtfTrackRef().isNonnull() && 
     ele.closestCtfTrackRef()->extra().isNonnull()){
    reco::TrackRef ctfTrack = ele.closestCtfTrackRef();
    reco::SuperClusterRef scClus = ele.superCluster();

    //dEta
    const reco::BeamSpot::Point& bsPos = beamSpot_->position();     
    math::XYZPoint scPosWRTVtx(scClus->x()-bsPos.x(), scClus->y()-bsPos.y() , scClus->z()-ctfTrack->vz());
    hltData.dEtaIn = fabs(scPosWRTVtx.eta()-ctfTrack->eta());

    //dPhi: lifted straight from hlt code
    float deltaPhi=fabs(ctfTrack->outerPosition().phi()-scClus->phi());
    if(deltaPhi>6.283185308) deltaPhi -= 6.283185308;
    if(deltaPhi>3.141592654) deltaPhi = 6.283185308-deltaPhi;
    hltData.dPhiIn = deltaPhi;
    
    //invEInvP
    if(ele.ecalEnergy()!=0 && ctfTrack->p()!=0) hltData.invEInvP= 1/ele.ecalEnergy() - 1/ctfTrack->p();
    else hltData.invEInvP = 0;
  }else{
    hltData.dEtaIn =999;
    hltData.dPhiIn =999;
    hltData.invEInvP = 999;

  }

  //Now get HLT p4 from triggerobject
  trigTools::fillHLTposition(ele,hltData,hltFiltersUsed_,trigEvt_.product(),hltTag_);
  //trigTools::fillHLTposition(phos(),hltFiltersUsed_,l1PreAndSeedFilters_,evtTrigBits,trigEvt_.product(),hltTag_); 
}


void OffHelper::fillHLTDataPho(const reco::Photon& pho,OffPho::HLTData& hltData)
{
  //Now get HLT p4 from triggerobject
  trigTools::fillHLTposition(pho,hltData, hltFiltersUsed_,trigEvt_.product(),hltTag_);
  //trigTools::fillHLTposition(phos(),hltFiltersUsed_,l1PreAndSeedFilters_,evtTrigBits,trigEvt_.product(),hltTag_); 
}




//this function coverts Photons to a format which more useful to me
int OffHelper::fillOffPhoVec(std::vector<OffPho>& egHLTOffPhos)
{
  egHLTOffPhos.clear();
  egHLTOffPhos.reserve(recoPhos_->size());
  for(reco::PhotonCollection::const_iterator phoIter=recoPhos_->begin(); phoIter!=recoPhos_->end();++phoIter){

    OffPho::IsolData isolData;  
    OffPho::ClusShapeData clusShapeData;
  
    fillIsolData(*phoIter,isolData);
    fillClusShapeData(*phoIter,clusShapeData);
   
    OffPho::HLTData hltData;
    fillHLTDataPho(*phoIter,hltData); 

    egHLTOffPhos.push_back(OffPho(*phoIter,clusShapeData,isolData,hltData));
    OffPho& pho =  egHLTOffPhos.back();
    pho.setCutCode(phoCuts_.getCutCode(pho));
    pho.setLooseCutCode(phoLooseCuts_.getCutCode(pho));

    std::vector<std::pair<TrigCodes::TrigBitSet,int> >trigCutsCutCodes;
    for(size_t i=0;i<trigCuts_.size();i++) trigCutsCutCodes.push_back(std::make_pair(trigCuts_[i].first,trigCuts_[i].second.getCutCode(pho)));
    pho.setTrigCutsCutCodes(trigCutsCutCodes); 


  }//end loop over photon collection
  return 0;
}


void OffHelper::fillIsolData(const reco::Photon& pho,OffPho::IsolData& isolData)
{
  EgammaTowerIsolation hcalIsolAlgo(hltHadIsolOuterCone_,hltHadIsolInnerCone_,hltHadIsolEtMin_,hltHadIsolDepth_,caloTowers_.product());
  EcalRecHitMetaCollection ebHits(*ebRecHits_);
  EcalRecHitMetaCollection eeHits(*ebRecHits_);
  EgammaRecHitIsolation ecalIsolAlgoEB(hltEMIsolOuterCone_,hltEMIsolInnerConeEB_,hltEMIsolEtaSliceEB_,
				       hltEMIsolEtMinEB_,hltEMIsolEMinEB_,caloGeom_,&ebHits,ecalSeverityLevel_.product(),DetId::Ecal);
  EgammaRecHitIsolation ecalIsolAlgoEE(hltEMIsolOuterCone_,hltEMIsolInnerConeEE_,hltEMIsolEtaSliceEE_,
				       hltEMIsolEtMinEE_,hltEMIsolEMinEE_,caloGeom_,&eeHits,ecalSeverityLevel_.product(),DetId::Ecal);
  
  isolData.nrTrks = pho.nTrkHollowConeDR03();
  isolData.ptTrks = pho.trkSumPtHollowConeDR03();
  isolData.em = pho.ecalRecHitSumEtConeDR03();
  isolData.had = pho.hcalTowerSumEtConeDR03();  

  //now calculate hlt algos
  if(calHLTHcalIsol_) isolData.hltHad=hcalIsolAlgo.getTowerESum(&pho);
  else isolData.hltHad = 0.;
  if(calHLTPhoTrkIsol_){
    if(hltPhoTrkIsolCountTrks_) isolData.hltTrks=hltPhoTrkIsolAlgo_->photonTrackCount(&pho,isolTrks_.product(),false);
    else isolData.hltTrks=hltPhoTrkIsolAlgo_->photonPtSum(&pho,isolTrks_.product(),false);
  }
  else isolData.hltTrks = 0.;
  if(calHLTEmIsol_) isolData.hltEm = ecalIsolAlgoEB.getEtSum(&pho) + 
                                     ecalIsolAlgoEE.getEtSum(&pho);
  else isolData.hltEm = 0.;
  
}

void OffHelper::fillClusShapeData(const reco::Photon& pho,OffPho::ClusShapeData& clusShapeData)
{
  clusShapeData.sigmaEtaEta = pho.sigmaEtaEta();
  clusShapeData.sigmaIEtaIEta =  pho.sigmaIetaIeta();
  double e5x5 =  pho.e5x5();
  if(e5x5!=0.){ //even though it is almost impossible for this to be 0., this code can never ever crash under any situation
    clusShapeData.e1x5Over5x5 =  pho.e1x5()/e5x5;
    clusShapeData.e2x5MaxOver5x5 = pho.e2x5()/e5x5;
  }else{
    clusShapeData.e1x5Over5x5 = -1;
    clusShapeData.e2x5MaxOver5x5 = -1;
  }
  clusShapeData.r9 = pho.r9();
  
  //sigmaPhiPhi and sigmaIPhiIPhi are not in object (and nor should they be) so have to get them old fashioned way
  //need to figure out if its in the barrel or endcap
  //get the first hit of the cluster and figure out if its barrel or endcap 
  const reco::BasicCluster& seedClus = *(pho.superCluster()->seed());
  const DetId seedDetId = seedClus.hitsAndFractions()[0].first; //note this may not actually be the seed hit but it doesnt matter because all hits will be in the barrel OR endcap (it is also incredably inefficient as it getHitsByDetId passes the vector by value not reference
  if(seedDetId.subdetId()==EcalBarrel){
    std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,ebRecHits_.product(),caloTopology_.product(),caloGeom_.product());
    std::vector<float> crysCov = EcalClusterTools::localCovariances(seedClus,ebRecHits_.product(),caloTopology_.product());
    clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
    clusShapeData.sigmaIPhiIPhi =  sqrt(crysCov[2]);
    }else{
    std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,eeRecHits_.product(),caloTopology_.product(),caloGeom_.product()); 
    std::vector<float> crysCov = EcalClusterTools::localCovariances(seedClus,eeRecHits_.product(),caloTopology_.product());
    
    clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
    clusShapeData.sigmaIPhiIPhi = sqrt(crysCov[2]); 
  }
}  

int OffHelper::setTrigInfo(const edm::Event & edmEvent, egHLT::OffEvt& offEvent)
{
  TrigCodes::TrigBitSet evtTrigBits = trigTools::getFiltersPassed(hltFiltersUsedWithNrCandsCut_,trigEvt_.product(),hltTag_);
  //the l1 prescale paths dont have a filter with I can figure out if it passed or failed with so have to use TriggerResults
  if(l1PreScaledPaths_.size()==l1PreScaledFilters_.size()){ //check to ensure both vectors have same number of events incase of screw ups     
    const edm::TriggerNames & triggerNames = edmEvent.triggerNames(*trigResults_);
    for(size_t pathNr=0;pathNr<l1PreScaledPaths_.size();pathNr++){ //now we have to check the prescaled l1 trigger paths
      unsigned int pathIndex = triggerNames.triggerIndex(l1PreScaledPaths_[pathNr]);
      if(pathIndex<trigResults_->size() && trigResults_->accept(pathIndex)){
	evtTrigBits |=TrigCodes::getCode(l1PreScaledFilters_[pathNr]);
      }    
    }
  }

  offEvent.setEvtTrigBits(evtTrigBits);

  trigTools::setFiltersObjPasses(offEvent.eles(),hltFiltersUsed_,l1PreAndSeedFilters_,evtTrigBits,trigEvt_.product(),hltTag_);
  trigTools::setFiltersObjPasses(offEvent.phos(),hltFiltersUsed_,l1PreAndSeedFilters_,evtTrigBits,trigEvt_.product(),hltTag_); 
  return 0;
}
