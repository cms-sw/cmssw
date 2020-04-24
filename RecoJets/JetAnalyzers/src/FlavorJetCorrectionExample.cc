/* Example analizer to use L5 flavor JetCorrector services
   Applies different flavor corrections randomly
    F.Ratnikov (UMd)  Nov 16, 2007
*/

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class FlavorJetCorrectionExample : public edm::EDAnalyzer {
 public:
  explicit FlavorJetCorrectionExample (const edm::ParameterSet& fParameters);
  ~FlavorJetCorrectionExample () override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
 private:
  edm::InputTag mInput;
  std::string mUDSCorrectorName;
  std::string mCCorrectorName;
  std::string mBCorrectorName;
};




#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

using namespace std;
using namespace reco;

FlavorJetCorrectionExample::FlavorJetCorrectionExample (const edm::ParameterSet& fConfig) 
  : mInput (fConfig.getParameter <edm::InputTag> ("src")),
    mUDSCorrectorName (fConfig.getParameter <std::string> ("UDSQuarksCorrector")),
    mCCorrectorName (fConfig.getParameter <std::string> ("CQuarkCorrector")),
    mBCorrectorName (fConfig.getParameter <std::string> ("BQuarkCorrector"))
{}

void FlavorJetCorrectionExample::analyze(const edm::Event& fEvent, const edm::EventSetup& fSetup) {
  // get all correctors
  const JetCorrector* udsJetCorrector = JetCorrector::getJetCorrector (mUDSCorrectorName, fSetup);
  const JetCorrector* cQuarkJetCorrector = JetCorrector::getJetCorrector (mCCorrectorName, fSetup);
  const JetCorrector* bQuarkJetCorrector = JetCorrector::getJetCorrector (mBCorrectorName, fSetup);
  const JetCorrector* corrector = nullptr;
  
  // get input jets (supposed to be MC corrected already)
  edm::Handle<CaloJetCollection> jets;                    
  fEvent.getByLabel (mInput, jets);
  // loop over jets                      
  for (unsigned ijet = 0; ijet < jets->size(); ++ijet) {
    const CaloJet& jet = (*jets)[ijet];
    std::cout << "FlavorJetCorrectionExample::analize-> jet #" << ijet;
    if (ijet%3 == 0) { // assume it is light quark
      std::cout << ": use USD quark corrections" << std::endl;
      corrector = udsJetCorrector;
    }
    else if (ijet%3 == 1) { // assume it is c quark
      std::cout << ": use c quark corrections" << std::endl;
      corrector = cQuarkJetCorrector;
    }
    else { // assume it is b quark
      std::cout << ": use b quark corrections" << std::endl;
      corrector = bQuarkJetCorrector;
    }
    // get selected correction for the jet
    double correction = corrector->correction (jet);
    // dump it
    std::cout << "  jet pt/eta/phi: " << jet.pt() << '/' <<  jet.eta() << '/' << jet.phi() 
	      << " -> correction factor: " << correction 
	      << ", corrected pt: " << jet.pt()*correction
	      << std::endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FlavorJetCorrectionExample);


