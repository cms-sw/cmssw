/*
 * =====================================================================================
 *
 *       Filename:  TauGenJetDumper.cc
 *
 *    Description:  Dump information about Generator taus
 *
 *         Author:  Evan K. Friis, UC Davis
 *
 *
 * =====================================================================================
 */
#include <sstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"

class TauGenJetDumper : public edm::EDAnalyzer {
  public:
    explicit TauGenJetDumper(const edm::ParameterSet& pset):
      genJetSrc_(pset.getParameter<edm::InputTag>("src")) {}
    ~TauGenJetDumper() override {}
    void analyze(const edm::Event& evt, const edm::EventSetup& es) override;
  private:
    edm::InputTag genJetSrc_;
};

void
TauGenJetDumper::analyze(const edm::Event& evt, const edm::EventSetup& es) {
  typedef edm::View<reco::GenJet> GenJetView;
  edm::Handle<GenJetView> jetView;
  evt.getByLabel(genJetSrc_, jetView);

  std::ostringstream output;
  output << " * * * Tau GenJet Dump " << std::endl;
  for(auto const& jet : *jetView) {
    output << "Decay mode: " << JetMCTagUtils::genTauDecayMode(jet) << "  "
      << jet.print() << std::endl;
  }
  std::cout << output.str();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TauGenJetDumper);
