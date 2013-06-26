/** \class HLTMhtProducer
*
*
*  \author Gheorghe Lungu
*
*/

#include "HLTrigger/JetMET/interface/HLTMhtProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
//#include "RecoMET/METProducers/interface/METProducer.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>


//
// constructors and destructor
//
HLTMhtProducer::HLTMhtProducer(const edm::ParameterSet& iConfig)
{
  inputJetTag_ = iConfig.getParameter< edm::InputTag > ("inputJetTag");
  minPtJet_= iConfig.getParameter<double> ("minPtJet");
  etaJet_= iConfig.getParameter<double> ("etaJet");
  usePt_= iConfig.getParameter<bool>("usePt");

  //register your products
  produces<reco::METCollection>();
}

HLTMhtProducer::~HLTMhtProducer(){}

void HLTMhtProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<double>("minPtJet",20.0);
  desc.add<double>("etaJet",9999.0);
  desc.add<bool>("usePt",true);
  descriptions.add("hltMhtProducer",desc);
}

// ------------ method called to produce the data  ------------
void
  HLTMhtProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  auto_ptr<reco::METCollection> result (new reco::METCollection); 

  math::XYZPoint vtx(0,0,0);
  
  //Handle<CaloJetCollection> recocalojets;
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByLabel(inputJetTag_,jets);

  // look at all candidates,  check cuts and add to result object
  double mhtx=0., mhty=0., mht;
  double jetVar;
  
  if(jets->size() > 0){
    // events with at least one jet
    //for (CaloJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); jet++) {
    for(edm::View<reco::Jet>::const_iterator jet = jets->begin(); jet != jets->end(); jet++ ) {
      jetVar = jet->pt();
      if (!usePt_) jetVar = jet->et();

      //---get MHT
      if (jetVar > minPtJet_ && std::abs(jet->eta()) < etaJet_) {
	mhtx -= jetVar*cos(jet->phi());
	mhty -= jetVar*sin(jet->phi());
      }
    }
    mht = sqrt(mhtx*mhtx + mhty*mhty);

    math::XYZTLorentzVector mhtVec(mhtx,mhty,0,mht);
    reco::MET mhtobj(mhtVec,vtx);
    result->push_back( mhtobj );
    
  } // events with at least one jet
  
    
  // put object into the Event
  iEvent.put(result);

}
