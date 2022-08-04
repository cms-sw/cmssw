// system include files
#include <atomic>
#include <memory>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HcalCalibObjects/interface/HcalHBHEMuonVariables.h"

//#define EDM_ML_DEBUG
//
// class declaration
//

namespace alcaHcalHBHEMuonFilter {
  struct Counters {
    Counters() : nAll_(0), nGood_(0), nFinal_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_, nFinal_;
  };
}  // namespace alcaHcalHBHEMuonFilter

class AlCaHcalHBHEMuonFilter : public edm::global::EDFilter<edm::RunCache<alcaHcalHBHEMuonFilter::Counters> > {
public:
  AlCaHcalHBHEMuonFilter(edm::ParameterSet const&);
  ~AlCaHcalHBHEMuonFilter() override = default;

  std::shared_ptr<alcaHcalHBHEMuonFilter::Counters> globalBeginRun(edm::Run const&,
                                                                   edm::EventSetup const&) const override;

  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------member data ---------------------------
  const int prescale_;
  const double minimumMuonP_;
  const edm::InputTag labelHBHEMuonVar_;
  const edm::EDGetTokenT<HcalHBHEMuonVariablesCollection> tokHBHEMuonVar_;
};

//
// constructors and destructor
//
AlCaHcalHBHEMuonFilter::AlCaHcalHBHEMuonFilter(edm::ParameterSet const& iConfig)
    : prescale_(iConfig.getParameter<int>("prescale")),
      minimumMuonP_(iConfig.getParameter<double>("minimumMuonP")),
      labelHBHEMuonVar_(iConfig.getParameter<edm::InputTag>("hbheMuonLabel")),
      tokHBHEMuonVar_(consumes<HcalHBHEMuonVariablesCollection>(labelHBHEMuonVar_)) {
  edm::LogVerbatim("HBHEMuon") << "Parameters read from config file \n\t prescale_ " << prescale_ << "\t minimumMuonP_ "
                               << minimumMuonP_ << "\n\t Labels " << labelHBHEMuonVar_;
}  // AlCaHcalHBHEMuonFilter::AlCaHcalHBHEMuonFilter  constructor

//
// member functions
//

// ------------ method called on each new Event  ------------
bool AlCaHcalHBHEMuonFilter::filter(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  bool accept(false);
  ++(runCache(iEvent.getRun().index())->nAll_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HBHEMuon") << "AlCaHcalHBHEMuonFilter::Run " << iEvent.id().run() << " Event "
                               << iEvent.id().event() << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
                               << iEvent.bunchCrossing();
#endif

  auto const& hbheMuonColl = iEvent.getHandle(tokHBHEMuonVar_);
  if (hbheMuonColl.isValid()) {
    auto hbheMuon = hbheMuonColl.product();
    if (!hbheMuon->empty()) {
      bool ok(false);
      for (auto const& muon : *hbheMuon)
        if (muon.pMuon_ >= minimumMuonP_)
          ok = true;
      if (ok) {
        ++(runCache(iEvent.getRun().index())->nGood_);
        if (prescale_ <= 1)
          accept = true;
        else if (runCache(iEvent.getRun().index())->nGood_ % prescale_ == 1)
          accept = true;
      }
    }
  } else {
    edm::LogVerbatim("HBHEMuon") << "AlCaHcalHBHEMuonFilter::Cannot find the collection for HcalHBHEMuonVariables";
  }

  // Return the acceptance flag
  if (accept)
    ++(runCache(iEvent.getRun().index())->nFinal_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HBHEMuon") << "AlCaHcalHBHEMuonFilter::Accept flag " << accept << " All "
                               << runCache(iEvent.getRun().index())->nAll_ << " Good "
                               << runCache(iEvent.getRun().index())->nGood_ << " Final "
                               << runCache(iEvent.getRun().index())->nFinal_;
#endif
  return accept;

}  // AlCaHcalHBHEMuonFilter::filter

// ------------ method called when starting to processes a run  ------------
std::shared_ptr<alcaHcalHBHEMuonFilter::Counters> AlCaHcalHBHEMuonFilter::globalBeginRun(edm::Run const& iRun,
                                                                                         edm::EventSetup const&) const {
  edm::LogVerbatim("HBHEMuon") << "Start the Run " << iRun.run();
  return std::make_shared<alcaHcalHBHEMuonFilter::Counters>();
}

// ------------ method called when ending the processing of a run  ------------
void AlCaHcalHBHEMuonFilter::globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const {
  edm::LogVerbatim("HBHEMuon") << "Select " << runCache(iRun.index())->nFinal_ << " out of "
                               << runCache(iRun.index())->nGood_ << " good and " << runCache(iRun.index())->nAll_
                               << " total events";
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void AlCaHcalHBHEMuonFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("prescale", 1);
  desc.add<double>("minimumMuonP", 10.0);
  desc.add<edm::InputTag>("hbheMuonLabel", edm::InputTag("alcaHcalHBHEMuonProducer", "hbheMuon"));
  descriptions.add("alcaHcalHBHEMuonFilter", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaHcalHBHEMuonFilter);
