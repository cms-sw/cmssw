/** \class EgammaHLTElectronCombinedIsolationProducer
 *
 *  \author Alessio Ghezzi
 * 
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronCombinedIsolationProducer.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EgammaHLTElectronCombinedIsolationProducer::EgammaHLTElectronCombinedIsolationProducer(const edm::ParameterSet& config) : conf_(config) {

  electronProducer_          = consumes<reco::ElectronCollection>(conf_.getParameter<edm::InputTag>("electronProducer"));
  recoEcalCandidateProducer_ = consumes<reco::RecoEcalCandidateCollection>(conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer"));

  for (edm::InputTag const & tag : conf_.getParameter< std::vector<edm::InputTag> > ("CaloIsolationMapTags"))
    CaloIsolTag_.push_back(consumes<reco::RecoEcalCandidateIsolationMap>(tag));
  
  TrackIsolTag_              = consumes<reco::ElectronIsolationMap>(conf_.getParameter<edm::InputTag>("TrackIsolationMapTag"));

  CaloIsolWeight_ = conf_.getParameter< std::vector<double> > ("CaloIsolationWeight");
  TrackIsolWeight_ = conf_.getParameter<double>("TrackIsolationWeight");
  
  if (CaloIsolTag_.size() != CaloIsolWeight_.size()){
    throw cms::Exception("BadConfig") << "vectors CaloIsolationMapTags and CaloIsolationWeight need to have size 3";
  }
  
  //register your products
  produces < reco::ElectronIsolationMap >();
}

EgammaHLTElectronCombinedIsolationProducer::~EgammaHLTElectronCombinedIsolationProducer()
{}

void EgammaHLTElectronCombinedIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electronProducer", edm::InputTag(""));
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag(""));
  desc.add<std::vector<edm::InputTag> >("CaloIsolationMapTags", std::vector<edm::InputTag>());
  desc.add<edm::InputTag>("TrackIsolationMapTag", edm::InputTag(""));
desc.add<std::vector<double> >("CaloIsolationWeight", std::vector<double>());
  desc.add<double>("TrackIsolationWeight", 0);
  descriptions.add("hltEgammaHLTElectronCombinedIsolationProducer", desc);  
}

void EgammaHLTElectronCombinedIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByToken(electronProducer_,electronHandle);

  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_,recoecalcandHandle);
  
  std::vector< edm::Handle<reco::RecoEcalCandidateIsolationMap> > CaloIsoMap;
  for( unsigned int u=0; u < CaloIsolTag_.size(); u++){
    edm::Handle<reco::RecoEcalCandidateIsolationMap> depMapTemp;
    if(CaloIsolWeight_[u] != 0) 
      iEvent.getByToken(CaloIsolTag_[u],depMapTemp);

    CaloIsoMap.push_back(depMapTemp);
  }
  
  edm::Handle<reco::ElectronIsolationMap> TrackIsoMap;
  if(TrackIsolWeight_ != 0) 
    iEvent.getByToken(TrackIsolTag_,TrackIsoMap);
  
  reco::ElectronIsolationMap TotalIsolMap(electronHandle);
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

