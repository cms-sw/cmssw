#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//
// class declaration
//

class LSNumberFilter : public edm::stream::EDFilter<> {
public:
  explicit LSNumberFilter(const edm::ParameterSet&);
  ~LSNumberFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  bool is_HLT_vetoed_;
  const unsigned int minLS_;
  const std::vector<std::string> veto_HLT_Menu_;
  HLTConfigProvider hltConfig_;
};

void LSNumberFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Filters the first minLS lumisections and reject the run according to the HLT menu name");
  desc.addUntracked<unsigned int>("minLS", 21)->setComment("first LS to accept");
  desc.addUntracked<std::vector<std::string>>("veto_HLT_Menu", {})->setComment("list of HLT menus to reject");
  descriptions.addWithDefaultLabel(desc);
}

LSNumberFilter::LSNumberFilter(const edm::ParameterSet& iConfig)
    : minLS_(iConfig.getUntrackedParameter<unsigned>("minLS", 21)),
      veto_HLT_Menu_(iConfig.getUntrackedParameter<std::vector<std::string>>("veto_HLT_Menu")) {}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool LSNumberFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if ((iEvent.luminosityBlock() < minLS_) || is_HLT_vetoed_) {
    return false;
  }

  return true;
}

void LSNumberFilter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed{false};
  hltConfig_.init(iRun, iSetup, "HLT", changed);
  is_HLT_vetoed_ = false;
  for (const auto& veto : veto_HLT_Menu_) {
    std::size_t found = hltConfig_.tableName().find(veto);
    if (found != std::string::npos) {
      is_HLT_vetoed_ = true;
      edm::LogWarning("LSNumberFilter") << "Detected " << veto
                                        << " in HLT Config tableName(): " << hltConfig_.tableName()
                                        << "; Events of this run will be ignored" << std::endl;
      break;
    }
  }
}
//define this as a plug-in
DEFINE_FWK_MODULE(LSNumberFilter);
-- dummy change --
