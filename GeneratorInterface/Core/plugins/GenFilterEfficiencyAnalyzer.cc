// F. Cossutti
// $Revision://

// analyzer of a summary information product on filter efficiency for a user specified path
// meant for the generator filter efficiency calculation

// system include files
#include <memory>
#include <iostream>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
//
// class declaration
//

namespace gfea {
  struct Empty {};
};  // namespace gfea

class GenFilterEfficiencyAnalyzer final : public edm::global::EDAnalyzer<edm::LuminosityBlockCache<gfea::Empty>> {
public:
  explicit GenFilterEfficiencyAnalyzer(const edm::ParameterSet&);
  ~GenFilterEfficiencyAnalyzer() final;

private:
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const final;
  std::shared_ptr<gfea::Empty> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                          edm::EventSetup const&) const final;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final;
  void endJob() final;

  edm::EDGetTokenT<GenFilterInfo> genFilterInfoToken_;
  mutable std::mutex mutex_;
  CMS_THREAD_GUARD(mutex_) mutable GenFilterInfo totalGenFilterInfo_;

  // ----------member data ---------------------------
};

GenFilterEfficiencyAnalyzer::GenFilterEfficiencyAnalyzer(const edm::ParameterSet& pset)
    : genFilterInfoToken_(consumes<GenFilterInfo, edm::InLumi>(pset.getParameter<edm::InputTag>("genFilterInfoTag"))),
      totalGenFilterInfo_(0, 0, 0, 0, 0., 0., 0., 0.) {}

GenFilterEfficiencyAnalyzer::~GenFilterEfficiencyAnalyzer() {}

void GenFilterEfficiencyAnalyzer::analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const {}

std::shared_ptr<gfea::Empty> GenFilterEfficiencyAnalyzer::globalBeginLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                                                     edm::EventSetup const&) const {
  return std::shared_ptr<gfea::Empty>();
}

// ------------ method called once each job just after ending the event loop  ------------

void GenFilterEfficiencyAnalyzer::globalEndLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                           edm::EventSetup const&) const {
  edm::Handle<GenFilterInfo> genFilter;
  iLumi.getByToken(genFilterInfoToken_, genFilter);

  std::cout << "Lumi section " << iLumi.id() << std::endl;

  std::cout << "N total = " << genFilter->sumWeights() << " N passed = " << genFilter->sumPassWeights()
            << " N failed = " << genFilter->sumFailWeights() << std::endl;
  std::cout << "Generator filter efficiency = " << genFilter->filterEfficiency(-1) << " +- "
            << genFilter->filterEfficiencyError(-1) << std::endl;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    totalGenFilterInfo_.mergeProduct(*genFilter);
  }
}

void GenFilterEfficiencyAnalyzer::endJob() {
  std::cout << "Total events = " << totalGenFilterInfo_.sumWeights()
            << " Passed events = " << totalGenFilterInfo_.sumPassWeights() << std::endl;
  std::cout << "Filter efficiency = " << totalGenFilterInfo_.filterEfficiency(-1) << " +- "
            << totalGenFilterInfo_.filterEfficiencyError(-1) << std::endl;
}

DEFINE_FWK_MODULE(GenFilterEfficiencyAnalyzer);
