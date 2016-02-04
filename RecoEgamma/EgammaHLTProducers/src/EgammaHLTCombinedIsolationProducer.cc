/** \class EgammaHLTCombinedIsolationProducer
 *
 *  \author Alessio Ghezzi
 *
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTCombinedIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

EgammaHLTCombinedIsolationProducer::EgammaHLTCombinedIsolationProducer(const edm::ParameterSet& config) : conf_(config)
{
 // use configuration file to setup input/output collection names
  recoEcalCandidateProducer_ = conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer");

  IsolTag_ = conf_.getParameter< std::vector<edm::InputTag> > ("IsolationMapTags");
  IsolWeight_ = conf_.getParameter< std::vector<double> > ("IsolationWeight");

  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();

  if ( IsolTag_.size() != IsolWeight_.size()){ 
    throw cms::Exception("BadConfig") << "vectors IsolationMapTags and IsolationWeight need to have the same size";
  }

}


EgammaHLTCombinedIsolationProducer::~EgammaHLTCombinedIsolationProducer(){}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTCombinedIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_,recoecalcandHandle);

  reco::RecoEcalCandidateIsolationMap TotalIsolMap;
  double TotalIso=0;

  std::vector< edm::Handle<reco::RecoEcalCandidateIsolationMap> > IsoMap;
  for( unsigned int u=0; u < IsolWeight_.size(); u++){
    edm::Handle<reco::RecoEcalCandidateIsolationMap> depMapTemp;
    if(IsolWeight_[u] != 0){ iEvent.getByLabel (IsolTag_[u],depMapTemp);}
    IsoMap.push_back(depMapTemp);
  }
  
  for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoecalcandHandle->begin(); iRecoEcalCand != recoecalcandHandle->end(); iRecoEcalCand++){
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle,iRecoEcalCand-recoecalcandHandle->begin());
    TotalIso = 0;  
    for( unsigned int u=0; u < IsolWeight_.size(); u++){
      if(IsolWeight_[u]==0){continue;}
      reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*IsoMap[u]).find( recoecalcandref );    
      TotalIso += mapi->val * IsolWeight_[u];
    }
    TotalIsolMap.insert(recoecalcandref, TotalIso);
    
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> Map(new reco::RecoEcalCandidateIsolationMap(TotalIsolMap));
  iEvent.put(Map);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTCombinedIsolationProducer);
