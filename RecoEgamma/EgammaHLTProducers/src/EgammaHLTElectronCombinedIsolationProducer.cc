/** \class EgammaHLTElectronCombinedIsolationProducer
 *
 *  \author Alessio Ghezzi
 * 
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronCombinedIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"


#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"


EgammaHLTElectronCombinedIsolationProducer::EgammaHLTElectronCombinedIsolationProducer(const edm::ParameterSet& config) : conf_(config)
{

  electronProducer_         = conf_.getParameter<edm::InputTag>("electronProducer");
  recoEcalCandidateProducer_ = conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer");

  CaloIsolTag_ = conf_.getParameter< std::vector<edm::InputTag> > ("CaloIsolationMapTags");
  //need to be in the order EcalIso, HcalIso, EleTrackIso
  CaloIsolWeight_ = conf_.getParameter< std::vector<double> > ("CaloIsolationWeight");

  TrackIsolTag_ = conf_.getParameter<edm::InputTag>("TrackIsolationMapTag");
  TrackIsolWeight_ = conf_.getParameter<double>("TrackIsolationWeight");

  if ( CaloIsolTag_.size() != CaloIsolWeight_.size()){
    throw cms::Exception("BadConfig") << "vectors CaloIsolationMapTags and CaloIsolationWeight need to have size 3";
  }
  
  
  //  SCProducer_               = conf_.getParameter<edm::InputTag>("electronProducer");

  //register your products
  produces < reco::ElectronIsolationMap >();

}
EgammaHLTElectronCombinedIsolationProducer::~EgammaHLTElectronCombinedIsolationProducer(){}


void EgammaHLTElectronCombinedIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  using namespace std;

  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_,electronHandle);

    edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_,recoecalcandHandle);
  
  std::vector< edm::Handle<reco::RecoEcalCandidateIsolationMap> > CaloIsoMap;
  for( unsigned int u=0; u < CaloIsolTag_.size(); u++){
    edm::Handle<reco::RecoEcalCandidateIsolationMap> depMapTemp;
    if(CaloIsolWeight_[u] != 0){ iEvent.getByLabel (CaloIsolTag_[u],depMapTemp);}
    CaloIsoMap.push_back(depMapTemp);
  }

  edm::Handle<reco::ElectronIsolationMap> TrackIsoMap;
  if(TrackIsolWeight_ != 0){ iEvent.getByLabel (TrackIsolTag_,TrackIsoMap);}
  
  reco::ElectronIsolationMap TotalIsolMap;
  double TotalIso=0;
  for(reco::ElectronCollection::const_iterator iElectron = electronHandle->begin(); iElectron != electronHandle->end(); iElectron++){
    TotalIso=0; 
    reco::ElectronRef electronref(reco::ElectronRef(electronHandle,iElectron - electronHandle->begin()));
    const reco::SuperClusterRef theEleClus = electronref->superCluster();
 
   //look for corresponding recoecal candidates to search for in the ecal and Hcal iso map
    for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoecalcandHandle->begin(); iRecoEcalCand != recoecalcandHandle->end(); iRecoEcalCand++){
      reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle,iRecoEcalCand-recoecalcandHandle->begin());
      const reco::SuperClusterRef cluster = recoecalcandref->superCluster();
      if(&(*cluster) ==  &(*theEleClus)) {//recoecalcand and electron have the same SC
	for(unsigned int u=0;  u < CaloIsolTag_.size() ;u++){
	  if(CaloIsolWeight_[u]==0){continue;}
	  reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*CaloIsoMap[u]).find( recoecalcandref );
	  TotalIso += mapi->val * CaloIsolWeight_[u];
	}
	break;
      }
    }
  
    //add the track isolation
    if(TrackIsolWeight_ != 0){
      reco::ElectronIsolationMap::const_iterator mapi = (*TrackIsoMap).find( electronref );
      TotalIso += mapi->val * TrackIsolWeight_;
    }
    TotalIsolMap.insert(electronref, TotalIso);
    
  }

  std::auto_ptr<reco::ElectronIsolationMap> isolMap(new reco::ElectronIsolationMap(TotalIsolMap));
  iEvent.put(isolMap);

}

