/* Generic Jet Corrections producer using JetCorrector services
    F.Ratnikov (UMd)
    Dec. 28, 2006
*/

#include "JetCorrectionProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"

using namespace std;
using namespace reco;

namespace cms {
  JetCorrectionProducer::JetCorrectionProducer (const edm::ParameterSet& fConfig) 
    : mInput (fConfig.getParameter <edm::InputTag> ("src")),
      mCorrectorNames (fConfig.getParameter <std::vector <std::string> > ("correctors")),
      mCorrectors (mCorrectorNames.size(), 0),
      mCacheId (0),
      mVerbose (fConfig.getUntrackedParameter <bool> ("verbose", false))
  {
    
    std::string alias = fConfig.getUntrackedParameter <std::string> ("alias", "");
    if (alias.empty ()) {
      produces <CaloJetCollection>();
    }
    else {
      produces <CaloJetCollection>().setBranchAlias (alias);
    }
  }

  void JetCorrectionProducer::produce(edm::Event& fEvent, const edm::EventSetup& fSetup) {
    // look for correctors
    const JetCorrectionsRecord& record = fSetup.get <JetCorrectionsRecord> ();
    if (record.cacheIdentifier() != mCacheId) { // need to renew cache
      for (unsigned i = 0; i < mCorrectorNames.size(); i++) {
	edm::ESHandle <JetCorrector> handle;
	record.get (mCorrectorNames [i], handle);
	mCorrectors [i] = &*handle;
      }
      mCacheId = record.cacheIdentifier();
    }
    
    edm::Handle<CaloJetCollection> jets;                           //Define Inputs
    fEvent.getByLabel (mInput, jets);                              //Get Inputs
    auto_ptr<CaloJetCollection> result (new CaloJetCollection);    //Corrected jets
    
    CaloJetCollection::const_iterator jet = jets->begin ();
    for (; jet != jets->end (); jet++) {
      const Jet* referenceJet = &*jet;
      CaloJet correctedJet = *jet; //copy original jet
      if (mVerbose) {
	std::cout << "JetCorrectionProducer::produce-> original jet: " << jet->print () << std::endl; 
      }
      for (unsigned i = 0; i < mCorrectors.size(); ++i) {
	double scale = mCorrectors[i]->correction (*referenceJet, fEvent, fSetup);
	if (mVerbose) {
	  std::cout << "JetCorrectionProducer::produce-> Corrector # " << i 
		    << ", correction factor: " << scale << std::endl;
	}
	correctedJet.scaleEnergy (scale); // apply correction
	referenceJet = &correctedJet;
      }
      if (mVerbose) {
	std::cout << "JetCorrectionProducer::produce-> corrected jet: " << correctedJet.print () << std::endl; 
      }
      result->push_back (correctedJet);
    }
    NumericSafeGreaterByPt<CaloJet> compJets;
    std::sort (result->begin (), result->end (), compJets); // reorder corrected jets
    fEvent.put(result);  //Puts Corrected Jet Collection into event
  }
}
