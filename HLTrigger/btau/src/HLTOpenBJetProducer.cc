/** \class HLTOpenBJetProducer
 *
 *  $Date: 2008/04/23 11:57:53 $
 *  $Revision: 1.1 $
 *
 *  \author Andrea Bocci
 *
 */

#include <string>

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/EDProducer.h"

class edm::Event;
class edm::EventSetup;
class edm::ParameterSet;

class HLTOpenBJetProducer : public edm::EDProducer
{
public:
  explicit HLTOpenBJetProducer(const edm::ParameterSet & config);
  ~HLTOpenBJetProducer();

  virtual void produce(edm::Event & event, const edm::EventSetup & setup);

private:
  edm::InputTag m_jets;         // module label of (tagged) jets
  edm::InputTag m_jetTagsL25;   // module label of L2.5 jet tags
  edm::InputTag m_jetTagsL3;    // module label of L3 jet tags
};


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

// constructors and destructor
HLTOpenBJetProducer::HLTOpenBJetProducer(const edm::ParameterSet & config) :
  m_jets(       config.getParameter<edm::InputTag> ("jets") ),
  m_jetTagsL25( config.getParameter<edm::InputTag> ("discriminatorL25") ),
  m_jetTagsL3(  config.getParameter<edm::InputTag> ("discriminatorL3") )
{
  // register your products
  produces<trigger::HLTOpenBJetCollection>();
}

HLTOpenBJetProducer::~HLTOpenBJetProducer()
{
}

// member functions

void
HLTOpenBJetProducer::produce(edm::Event& event, const edm::EventSetup& setup)
{
  std::auto_ptr<trigger::HLTOpenBJetCollection> object( new trigger::HLTOpenBJetCollection() );

  edm::Handle<edm::View<reco::Jet> > h_jets;
  event.getByLabel(m_jets, h_jets);
  const edm::View<reco::Jet> & jets = * h_jets;

  edm::Handle<reco::JetTagCollection> h_jetTagsL25;
  event.getByLabel(m_jetTagsL25, h_jetTagsL25);
  const reco::JetTagCollection & jetTagsL25 = * h_jetTagsL25;

  edm::Handle<reco::JetTagCollection > h_jetTagsL3;
  event.getByLabel(m_jetTagsL3, h_jetTagsL3);
  const reco::JetTagCollection & jetTagsL3 = * h_jetTagsL3;

  // this loop assumes that all collections are in sync, i.e., that the i-th element of each collection referes to the same jet
  // this is true only if no filters have been applied after creating the jet collection
  for (unsigned int i = 0; i < jets.size(); ++j) {
    object->addJet(jets[i]->energy(), jetTagsL25[i].second, jetTagsL3[i].second);
  }

  event.put(object);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTOpenBJetProducer);
