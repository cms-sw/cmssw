/* Template producer to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetObjects/interface/CaloJetCollection.h"

#include "RecoJets/JetProducers/interface/ToyJetCorrector.h"

using namespace std;

namespace cms
{
  ToyJetCorrector::ToyJetCorrector(edm::ParameterSet const& conf) :
    mAlgorithm (conf.getParameter<double>  ("scale")),
    mInput (conf.getParameter<string>  ("src"))
  {
    produces<CaloJetCollection>();
  }

  void ToyJetCorrector::produce(edm::Event& e, const edm::EventSetup&)
  {
    // get input
    edm::Handle<CaloJetCollection> jets;
    e.getByLabel( mInput, jets);
    auto_ptr<CaloJetCollection> result (new CaloJetCollection);  //Corrected Jet Coll
    CaloJetCollection::const_iterator jet = jets->begin ();
    for (; jet != jets->end (); jet++) {
      result->push_back (mAlgorithm.applyCorrection (*jet));
    }
    e.put(result);  //Puts Corrected Jet Collection into event
  }
}
