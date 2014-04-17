/*
 * =====================================================================================
 *
 *       Filename:  RecoTauDumper.cc
 *
 *    Description:  Dump information about reco::PFTaus
 *
 *         Author:  Evan K. Friis, UC Davis
 *
 *
 * =====================================================================================
 */
#include <boost/foreach.hpp>
#include <sstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Candidate/interface/Candidate.h"

// Methods to write the different types
namespace {

void write(std::ostringstream& output, const reco::PFTau& tau) {
  output << " ------------------------------------" << std::endl;
  output << tau << std::endl;
  tau.dump(output);
  if (tau.pfTauTagInfoRef().isNonnull()) {
    output << " TTInfoJetRefID: "
      << tau.pfTauTagInfoRef()->pfjetRef().id() << ":"
      << tau.pfTauTagInfoRef()->pfjetRef().key() << std::endl;
    output << " TTInfoJetRef: " << *(tau.pfTauTagInfoRef()->pfjetRef());
  }
  if (tau.jetRef().isNonnull()) {
    output << " JetRefID: "
      << tau.jetRef().id() << ":"
      << tau.jetRef().key() << std::endl;
    output << " JetRef: " << *(tau.jetRef());

  }
  output << std::endl;
}

void write(std::ostringstream& output, const reco::PFJet& jet) {
  output << " ------------------------------------" << std::endl;
  output << jet << std::endl;
  BOOST_FOREACH(const reco::PFCandidatePtr& cand, jet.getPFConstituents()) {
    output << " --> " << *cand << std::endl;
  }
  output << std::endl;
}

void write(std::ostringstream& output, const reco::Candidate& cand) {
  output << " ------------------------------------" << std::endl;
  output <<
    " candidate (pt/eta/phi): (" << cand.pt() << "/"
                                    << cand.eta() << "/"
                                    << cand.phi() << ")" << std::endl;
  output << std::endl;
}

}

template<typename T>
class CollectionDumper : public edm::EDAnalyzer {
  public:
    explicit CollectionDumper(const edm::ParameterSet& pset):
      src_(pset.getParameter<edm::InputTag>("src")),
      moduleName_(pset.getParameter<std::string>("@module_label")){}
    virtual ~CollectionDumper() {}
    virtual void analyze(const edm::Event& evt, const edm::EventSetup& es);
  private:
    edm::InputTag src_;
    std::string moduleName_;
};

template<typename T> void
CollectionDumper<T>::analyze(const edm::Event& evt, const edm::EventSetup& es) {
  typedef edm::View<T> TView;
  edm::Handle<TView> view;
  evt.getByLabel(src_, view);

  std::ostringstream output;
  output << " * * * <" << moduleName_
    << "> Dump - source: [" << src_ << "]" << std::endl;

  BOOST_FOREACH(const T& obj, *view) {
    write(output, obj);
  }
  std::cout << output.str();
}

typedef CollectionDumper<reco::PFTau> RecoTauDumper;
typedef CollectionDumper<reco::PFJet> PFJetDumper;
typedef CollectionDumper<reco::Candidate> CandidateDumper;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauDumper);
DEFINE_FWK_MODULE(PFJetDumper);
DEFINE_FWK_MODULE(CandidateDumper);
