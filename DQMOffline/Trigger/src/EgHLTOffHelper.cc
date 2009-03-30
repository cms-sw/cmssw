#include "DQMOffline/Trigger/interface/EgHLTOffHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTTrackIsolation.h"
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolation.h"

#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"
#include "DQMOffline/Trigger/interface/EgHLTErrCodes.h"
using namespace egHLT;

OffHelper::~OffHelper()
{
  if(hltHcalIsolAlgo_) delete hltHcalIsolAlgo_;
  if(hltEleTrkIsolAlgo_) delete hltEleTrkIsolAlgo_;
  if(hltPhoTrkIsolAlgo_) delete hltPhoTrkIsolAlgo_;
}

void OffHelper::setup(const edm::ParameterSet& conf,const std::vector<std::string>& hltFiltersUsed)
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
  eleEcalIsolTag_ = conf.getParameter<edm::InputTag>("eleEcalIsolTag");
  eleHcalDepth1IsolTag_ = conf.getParameter<edm::InputTag>("eleHcalDepth1IsolTag");
  eleHcalDepth2IsolTag_ = conf.getParameter<edm::InputTag>("eleHcalDepth2IsolTag");
  eleTrkIsolTag_ = conf.getParameter<edm::InputTag>("eleTrkIsolTag");
  //phoIDTag_ = conf.getParameter<edm::InputTag>("phoIDTag");

  eleCuts_.setup(conf.getParameter<edm::ParameterSet>("eleCuts"));
  eleLooseCuts_.setup(conf.getParameter<edm::ParameterSet>("eleLooseCuts"));
  phoCuts_.setup(conf.getParameter<edm::ParameterSet>("phoCuts"));
  phoLooseCuts_.setup(conf.getParameter<edm::ParameterSet>("phoLooseCuts"));
 

  hltFiltersUsed_ = hltFiltersUsed; //expensive but only do this once and faster ways could make things less clear

  std::vector<edm::ParameterSet> trigCutParam(conf.getParameter<std::vector<edm::ParameterSet> >("triggerCuts"));
  for(size_t trigNr=0;trigNr<trigCutParam.size();trigNr++) {
    trigCuts_.push_back(std::make_pair(TrigCodes::getCode(trigCutParam[trigNr].getParameter<std::string>("trigName")),OffEgSel(trigCutParam[trigNr])));
  }

  //  hltHcalIsolAlgo_ = new EgammaHLTHcalIsolation(conf.getParameter<double>("hltHcalIsolMinPt"),conf.getParameter<double>("hltHcalIsolConeSize"));
  hltHcalIsolAlgo_ = new EgammaHLTHcalIsolation(0.,0.3);
  hltEleTrkIsolAlgo_ = new EgammaHLTTrackIsolation(1.5,0.2,0.1,999999.0,0.02);
  hltPhoTrkIsolAlgo_ = new EgammaHLTTrackIsolation(1.5,0.3,999999.,999999.,0.);

}

int OffHelper::makeOffEvt(const edm::Event& edmEvent,const edm::EventSetup& setup,egHLT::OffEvt& offEvent)
{
  offEvent.clear();
  int errCode=0; //excution stops as soon as an error is flagged
  if(errCode==0) errCode = getHandles(edmEvent,setup);
  if(errCode==0) errCode = fillOffEleVec(offEvent.eles());
  if(errCode==0) errCode = fillOffPhoVec(offEvent.phos());
  if(errCode==0) errCode =  setTrigInfo(offEvent);
  if(errCode==0) offEvent.setJets(recoJets_);
  return errCode;
}

int OffHelper::getHandles(const edm::Event& event,const edm::EventSetup& setup)
{
  try { 
    setup.get<CaloGeometryRecord>().get(caloGeom_);
    setup.get<CaloTopologyRecord>().get(caloTopology_);
  }catch(...){
    return errCodes::Geom;
  }
  //this will have to
  if(!getHandle(event,triggerSummaryLabel_,trigEvt_)) return errCodes::TrigEvent; //must have this, otherwise skip event
  if(!getHandle(event,electronsTag_,recoEles_)) return errCodes::OffEle; //need for electrons
  if(!getHandle(event,photonsTag_, recoPhos_)) return errCodes::OffPho; //need for photons
  if(!getHandle(event,caloJetsTag_,recoJets_)) return errCodes::OffJet; //need for electrons and photons

  //need for id
  if(!getHandle(event,ecalRecHitsEBTag_,ebRecHits_)) return errCodes::EBRecHits;
  if(!getHandle(event,ecalRecHitsEETag_,eeRecHits_)) return errCodes::EERecHits;

  //  if(!getHandle(event,phoIDTag_, photonIDMap_)) return errCodes::PhoID;

  //need for HLT isolations (and also need rechits)
  if(!getHandle(event,isolTrkTag_,isolTrks_)) return errCodes::IsolTrks;
  if(!getHandle(event,hbheHitsTag_, hbheHits_)) return errCodes::HBHERecHits;
  if(!getHandle(event,hfHitsTag_, hfHits_)) return errCodes::HFRecHits;
  
  if(!getHandle(event,eleEcalIsolTag_,eleEcalIsol_)) return errCodes::EleEcalIsol;
  if(!getHandle(event,eleHcalDepth1IsolTag_,eleHcalDepth1Isol_)) return errCodes::EleHcalD1Isol;
  if(!getHandle(event,eleHcalDepth2IsolTag_,eleHcalDepth2Isol_)) return errCodes::EleHcalD1Isol;
  if(!getHandle(event,eleTrkIsolTag_,eleTrkIsol_)) return errCodes::EleTrkIsol;
  
  return 0;
}

//this function coverts GsfElectrons to a format which is actually useful to me
int OffHelper::fillOffEleVec(std::vector<OffEle>& egHLTOffEles)
{
  egHLTOffEles.clear();
  egHLTOffEles.reserve(recoEles_->size());
  for(reco::GsfElectronCollection::const_iterator gsfIter=recoEles_->begin(); gsfIter!=recoEles_->end();++gsfIter){
    const reco::GsfElectronRef eleRef(recoEles_,gsfIter-recoEles_->begin()); //we create an electron ref so we can use the valuemap

    OffEle::IsolData isolData;
    isolData.ptTrks=(*eleTrkIsol_)[eleRef];
    isolData.nrTrks=999;
    isolData.em= (*eleEcalIsol_)[eleRef];
    isolData.hadDepth1 =  (*eleHcalDepth1Isol_)[eleRef];
    isolData.hadDepth2 =  (*eleHcalDepth2Isol_)[eleRef];   
    isolData.hltHad=hltHcalIsolAlgo_->isolPtSum(&*gsfIter,hbheHits_.product(),hfHits_.product(),caloGeom_.product());
    isolData.hltTrksEle=hltEleTrkIsolAlgo_->electronPtSum(&(*(gsfIter->gsfTrack())),isolTrks_.product());
    isolData.hltTrksPho=hltPhoTrkIsolAlgo_->photonTrackCount(&*gsfIter,isolTrks_.product(),false);


    OffEle::ClusShapeData clusShapeData;
    //classification variable is unrelyable so get the first hit of the cluster and figure out if its barrel or endcap
    const reco::BasicCluster& seedClus = *(gsfIter->superCluster()->seed());
    const DetId seedDetId = seedClus.hitsAndFractions()[0].first; //note this may not actually be the seed hit but it doesnt matter because all hits will be in the barrel OR endcap (it is also incredably inefficient as it getHitsByDetId passes the vector by value not reference
    if(seedDetId.subdetId()==EcalBarrel){
      std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,ebRecHits_.product(),caloTopology_.product(),caloGeom_.product());
      std::vector<float> crysCov = EcalClusterTools::localCovariances(seedClus,ebRecHits_.product(),caloTopology_.product());
      clusShapeData.sigmaEtaEta = sqrt(stdCov[0]);
      clusShapeData.sigmaIEtaIEta =  sqrt(crysCov[0]);
      clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
      clusShapeData.sigmaIPhiIPhi =  sqrt(crysCov[2]);
      double e5x5 =  EcalClusterTools::e5x5(seedClus,ebRecHits_.product(),caloTopology_.product());
      if(e5x5!=0.){
	clusShapeData.e1x5Over5x5 =  EcalClusterTools::e1x5(seedClus,ebRecHits_.product(),caloTopology_.product())/e5x5;
	clusShapeData.e2x5MaxOver5x5 = EcalClusterTools::e2x5Max(seedClus,ebRecHits_.product(),caloTopology_.product())/e5x5;
      }else{
	clusShapeData.e1x5Over5x5 = -1;
	clusShapeData.e2x5MaxOver5x5 = -1;
      }
      if(gsfIter->superCluster()->rawEnergy()!=0.){
	clusShapeData.r9 = EcalClusterTools::e3x3(seedClus,ebRecHits_.product(),caloTopology_.product()) / gsfIter->superCluster()->rawEnergy();
      }else clusShapeData.r9 = -1.;
       
    }else{
      std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,eeRecHits_.product(),caloTopology_.product(),caloGeom_.product()); 
      std::vector<float> crysCov = EcalClusterTools::localCovariances(seedClus,eeRecHits_.product(),caloTopology_.product());
      clusShapeData.sigmaEtaEta = sqrt(stdCov[0]);  
      clusShapeData.sigmaIEtaIEta = sqrt(crysCov[0]);
      clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
      clusShapeData.sigmaIPhiIPhi = sqrt(crysCov[2]);
      clusShapeData.e1x5Over5x5 = -1.; //note defined for endcap
      clusShapeData.e2x5MaxOver5x5 = -1.; //note defined for endcap
      if(gsfIter->superCluster()->rawEnergy()!=0.){
	clusShapeData.r9 = EcalClusterTools::e3x3(seedClus,eeRecHits_.product(),caloTopology_.product()) / gsfIter->superCluster()->rawEnergy();
      }else clusShapeData.r9 = -1.;
    }

    egHLTOffEles.push_back(OffEle(*gsfIter,clusShapeData,isolData));
    
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

//this function coverts Photons to a format which is actually useful to me
int OffHelper::fillOffPhoVec(std::vector<OffPho>& egHLTOffPhos)
{
  egHLTOffPhos.clear();
  egHLTOffPhos.reserve(recoPhos_->size());
  for(reco::PhotonCollection::const_iterator phoIter=recoPhos_->begin(); phoIter!=recoPhos_->end();++phoIter){



    OffPho::IsolData isolData;  
    OffPho::ClusShapeData clusShapeData;
  
    //need to figure out if its in the barrel or endcap
    //classification variable is unrelyable so get the first hit of the cluster and figure out if its barrel or endcap
    const reco::BasicCluster& seedClus = *(phoIter->superCluster()->seed());
    const DetId seedDetId = seedClus.hitsAndFractions()[0].first; //note this may not actually be the seed hit but it doesnt matter because all hits will be in the barrel OR endcap (it is also incredably inefficient as it getHitsByDetId passes the vector by value not reference
    if(seedDetId.subdetId()==EcalBarrel){
      std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,ebRecHits_.product(),caloTopology_.product(),caloGeom_.product());
      std::vector<float> crysCov = EcalClusterTools::localCovariances(seedClus,ebRecHits_.product(),caloTopology_.product());
      clusShapeData.sigmaEtaEta = sqrt(stdCov[0]);
      clusShapeData.sigmaIEtaIEta =  sqrt(crysCov[0]);
      clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
      clusShapeData.sigmaIPhiIPhi =  sqrt(crysCov[2]);
      double e5x5 =  EcalClusterTools::e5x5(seedClus,ebRecHits_.product(),caloTopology_.product());
      if(e5x5!=0.){
	clusShapeData.e1x5Over5x5 =  EcalClusterTools::e1x5(seedClus,ebRecHits_.product(),caloTopology_.product())/e5x5;
	clusShapeData.e2x5MaxOver5x5 = EcalClusterTools::e2x5Max(seedClus,ebRecHits_.product(),caloTopology_.product())/e5x5;
      }else{
	clusShapeData.e1x5Over5x5 = -1;
	clusShapeData.e2x5MaxOver5x5 = -1;
      }
      // clusShapeData.r9 = EcalClusterTools::e3x3(seedClus,ebRecHits_,caloTopology_) / phoIter->superCluster()->rawEnergy();
     
    }else{
      std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,eeRecHits_.product(),caloTopology_.product(),caloGeom_.product()); 
      std::vector<float> crysCov = EcalClusterTools::localCovariances(seedClus,eeRecHits_.product(),caloTopology_.product());
      clusShapeData.sigmaEtaEta = sqrt(stdCov[0]);  
      clusShapeData.sigmaIEtaIEta = sqrt(crysCov[0]);
      clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
      clusShapeData.sigmaIPhiIPhi = sqrt(crysCov[2]);
      clusShapeData.e1x5Over5x5 = -1.; //note defined for endcap
      clusShapeData.e2x5MaxOver5x5 = -1.; //note defined for endcap
      //clusShapeData.r9 = EcalClusterTools::e3x3(seedClus,eeRecHits_,caloTopology_) / phoIter->superCluster()->rawEnergy();
    
    }
 
    //time to get the photon ID. Yes this is simpliest way
    //edm::Ref<reco::PhotonCollection> phoRef(recoPhos_,phoIter-recoPhos_->begin());
    //    const reco::PhotonIDRef photonID = (*photonIDMap_)[phoRef];
    clusShapeData.r9 = phoIter->r9();
    isolData.nrTrks = phoIter->nTrkHollowConeDR03();
    isolData.ptTrks = phoIter->trkSumPtHollowConeDR03();
    isolData.em = phoIter->ecalRecHitSumEtConeDR03();
    isolData.had = phoIter->hcalTowerSumEtConeDR03(); //will move to hcalTowerSumConeDR03 when the next photon tag is in (theres a typo in the function name)
    isolData.hltHad=hltHcalIsolAlgo_->isolPtSum(&*phoIter,hbheHits_.product(),hfHits_.product(),caloGeom_.product());
    isolData.hltTrks=hltPhoTrkIsolAlgo_->photonTrackCount(&*phoIter,isolTrks_.product(),false);

    egHLTOffPhos.push_back(OffPho(*phoIter,clusShapeData,isolData));
    OffPho& pho =  egHLTOffPhos.back();
    pho.setCutCode(phoCuts_.getCutCode(pho));
    pho.setLooseCutCode(phoLooseCuts_.getCutCode(pho));

    std::vector<std::pair<TrigCodes::TrigBitSet,int> >trigCutsCutCodes;
    for(size_t i=0;i<trigCuts_.size();i++) trigCutsCutCodes.push_back(std::make_pair(trigCuts_[i].first,trigCuts_[i].second.getCutCode(pho)));
    pho.setTrigCutsCutCodes(trigCutsCutCodes); 


  }//end loop over photon collection
  return 0;
}




int OffHelper::setTrigInfo(egHLT::OffEvt& offEvent)
{
  TrigCodes::TrigBitSet evtTrigBits = trigTools::getFiltersPassed(hltFiltersUsed_,trigEvt_.product(),hltTag_);
  offEvent.setEvtTrigBits(evtTrigBits);
  trigTools::setFiltersObjPasses(offEvent.eles(),hltFiltersUsed_,trigEvt_.product(),hltTag_);
  trigTools::setFiltersObjPasses(offEvent.phos(),hltFiltersUsed_,trigEvt_.product(),hltTag_); 
  return 0;
}
