/*
 * =====================================================================================
 *
 *       Filename:  RecoTauDumper.cc
 *
 *    Description:  Dump information about reco::PFTaus
 *
 *         Author:  Evan K. Friis, UC Davis
 *
 *         $Id $
 *
 * =====================================================================================
 */
#include <boost/foreach.hpp>
#include <sstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"

class RecoTauDumper : public edm::EDAnalyzer {
  public:
    explicit RecoTauDumper(const edm::ParameterSet& pset):
      tauSrc_(pset.getParameter<edm::InputTag>("src")) {}
    virtual ~RecoTauDumper() {}
    virtual void analyze(const edm::Event& evt, const edm::EventSetup& es);
  private:
    edm::InputTag tauSrc_;
};

void RecoTauDumper::analyze(const edm::Event& evt, const edm::EventSetup& es) {
  typedef edm::View<reco::PFTau> TauView;
  edm::Handle<TauView> tauView;
  evt.getByLabel(tauSrc_, tauView);

  std::ostringstream output;
  output << " * * * reco::PFTau Dump - Source: " << tauSrc_ << std::endl;
  BOOST_FOREACH(const reco::PFTau& tau, *tauView) {
    output << " ------------------------------------" << std::endl;
    output << tau << std::endl;
    tau.dump(output);
    output << std::endl;
  }
  std::cout << output.str();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauDumper);
