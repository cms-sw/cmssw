/** \class EgammaHLTCombinedIsolationProducer
 *
 *  \author Alessio Ghezzi
 *
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTCombinedIsolationProducer.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EgammaHLTCombinedIsolationProducer::EgammaHLTCombinedIsolationProducer(const edm::ParameterSet& config) : conf_(config)
{
 // use configuration file to setup input/output collection names
  recoEcalCandidateProducer_ = consumes<reco::RecoEcalCandidateCollection>(conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer"));
  
  std::vector<edm::InputTag> tempIsolTag = conf_.getParameter< std::vector<edm::InputTag> > ("IsolationMapTags");
  for (unsigned int i=0; i<tempIsolTag.size(); i++)
    IsolTag_.push_back(consumes<reco::RecoEcalCandidateIsolationMap>(tempIsolTag[i]));
		       
  IsolWeight_ = conf_.getParameter< std::vector<double> > ("IsolationWeight");

  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();

  if (IsolTag_.size() != IsolWeight_.size()){ 
    throw cms::Exception("BadConfig") << "vectors IsolationMapTags and IsolationWeight need to have the same size";
  }
}

EgammaHLTCombinedIsolationProducer::~EgammaHLTCombinedIsolationProducer()
{}

void EgammaHLTCombinedIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<std::vector<edm::InputTag>>("IsolationMapTags",  std::vector<edm::InputTag>());
  desc.add<std::vector<double>>("IsolationWeight", std::vector<double>());
  descriptions.add("hltEgammaHLTCombinedIsolationProducer", desc);  
}

  
// ------------ method called to produce the data  ------------
void
EgammaHLTCombinedIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_, recoecalcandHandle);

  reco::RecoEcalCandidateIsolationMap TotalIsolMap;
  double TotalIso=0;

  std::vector< edm::Handle<reco::RecoEcalCandidateIsolationMap> > IsoMap;
  for( unsigned int u=0; u < IsolWeight_.size(); u++){
    edm::Handle<reco::RecoEcalCandidateIsolationMap> depMapTemp;
    if(IsolWeight_[u] != 0)
      iEvent.getByToken(IsolTag_[u], depMapTemp);
    
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
