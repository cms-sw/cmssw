/* Example analizer to use L7 flavor JetCorrector services
   Applies different parton corrections randomly
   original from F.Ratnikov (UMd)  Nov 16, 2007 - Adapted deom A.Santocchia Mar 01, 2008
*/

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class PartonJetCorrectionExample : public edm::EDAnalyzer {
 public:
  explicit PartonJetCorrectionExample (const edm::ParameterSet& fParameters);
  virtual ~PartonJetCorrectionExample () {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
 private:
  edm::InputTag mInput;
  std::string m_gJ_CorrectorName;
  std::string m_qJ_CorrectorName;
  std::string m_bJ_CorrectorName;
  std::string m_bT_CorrectorName;
};




#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

using namespace std;
using namespace reco;

PartonJetCorrectionExample::PartonJetCorrectionExample (const edm::ParameterSet& fConfig) 
  : mInput (fConfig.getParameter <edm::InputTag> ("src")),
    m_gJ_CorrectorName (fConfig.getParameter <std::string> ("gJetCorrector")),
    m_qJ_CorrectorName (fConfig.getParameter <std::string> ("qJetCorrector")),
    m_bJ_CorrectorName (fConfig.getParameter <std::string> ("bJetCorrector")),
    m_bT_CorrectorName (fConfig.getParameter <std::string> ("bTopCorrector"))
{}

void PartonJetCorrectionExample::analyze(const edm::Event& fEvent, const edm::EventSetup& fSetup) {
  // get all correctors
  const JetCorrector* gJetCorrector = JetCorrector::getJetCorrector (m_gJ_CorrectorName, fSetup);
  const JetCorrector* qJetCorrector = JetCorrector::getJetCorrector (m_qJ_CorrectorName, fSetup);
  const JetCorrector* bJetCorrector = JetCorrector::getJetCorrector (m_bJ_CorrectorName, fSetup);
  const JetCorrector* bTopCorrector = JetCorrector::getJetCorrector (m_bT_CorrectorName, fSetup);
  const JetCorrector* corrector = 0;
  
  // get input jets (supposed to be MC corrected already)
  edm::Handle<CaloJetCollection> jets;                    
  fEvent.getByLabel (mInput, jets);
  // loop over jets                      
  for (unsigned ijet = 0; ijet < jets->size(); ++ijet) {
    const CaloJet& jet = (*jets)[ijet];
    std::cout << "PartonJetCorrectionExample::analize-> jet #" << ijet;
    if (ijet%4 == 0) { // assume it is gluon from diJet
      std::cout << ": use gJ corrections" << std::endl;
      corrector = gJetCorrector;
    }
    else if (ijet%4 == 1) { // assume it is light quark from diJet
      std::cout << ": use qJ corrections" << std::endl;
      corrector = qJetCorrector;
    }
    else if (ijet%4 == 2) { // assume it is b quark from diJet
      std::cout << ": use bJ corrections" << std::endl;
      corrector = bJetCorrector;
    }
    else { // assume it is b quark from ttbar
      std::cout << ": use bT corrections" << std::endl;
      corrector = bTopCorrector;
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
DEFINE_FWK_MODULE(PartonJetCorrectionExample);


