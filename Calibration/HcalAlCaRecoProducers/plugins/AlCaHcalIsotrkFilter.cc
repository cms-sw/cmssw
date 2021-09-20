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
#include "DataFormats/HcalCalibObjects/interface/HcalIsoTrkCalibVariables.h"

//#define EDM_ML_DEBUG
//
// class declaration
//

namespace alcaHcalIsoTrk {
  struct Counters {
    Counters() : nAll_(0), nGood_(0), nLow_(0), nHigh_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_, nLow_, nHigh_;
  };
}  // namespace alcaHcalIsoTrk

class AlCaHcalIsotrkFilter : public edm::global::EDFilter<edm::RunCache<alcaHcalIsoTrk::Counters> > {
public:
  AlCaHcalIsotrkFilter(edm::ParameterSet const&);

  std::shared_ptr<alcaHcalIsoTrk::Counters> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;

  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------member data ---------------------------
  double pTrackLow_, pTrackHigh_;
  int prescaleLow_, prescaleHigh_;
  edm::InputTag labelIsoTkVar_;
  edm::EDGetTokenT<HcalIsoTrkCalibVariablesCollection> tokIsoTrkVar_;
};

//
// constructors and destructor
//
AlCaHcalIsotrkFilter::AlCaHcalIsotrkFilter(edm::ParameterSet const& iConfig)
    : pTrackLow_(iConfig.getParameter<double>("momentumLow")),
      pTrackHigh_(iConfig.getParameter<double>("momentumHigh")),
      prescaleLow_(iConfig.getParameter<int>("prescaleLow")),
      prescaleHigh_(iConfig.getParameter<int>("prescaleHigh")),
      labelIsoTkVar_(iConfig.getParameter<edm::InputTag>("isoTrackLabel")),
      tokIsoTrkVar_(consumes<HcalIsoTrkCalibVariablesCollection>(labelIsoTkVar_)) {
  edm::LogVerbatim("HcalIsoTrack") << "Parameters read from config file \n\t momentumLow_ " << pTrackLow_
                                   << "\t prescaleLow_ " << prescaleLow_ << "\t momentumHigh_ " << pTrackHigh_
                                   << "\t prescaleHigh_ " << prescaleHigh_ << "\n\t Labels " << labelIsoTkVar_;
}  // AlCaHcalIsotrkFilter::AlCaHcalIsotrkFilter  constructor

//
// member functions
//

// ------------ method called on each new Event  ------------
bool AlCaHcalIsotrkFilter::filter(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  bool accept(false);
  ++(runCache(iEvent.getRun().index())->nAll_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "AlCaHcalIsotrkFilter::Run " << iEvent.id().run() << " Event "
                                   << iEvent.id().event() << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
                                   << iEvent.bunchCrossing();
#endif

  auto const& isotrkCalibColl = iEvent.getHandle(tokIsoTrkVar_);
  if (isotrkCalibColl.isValid()) {
    auto isotrkCalib = isotrkCalibColl.product();
    bool low(false), high(false), inRange(false);
    for (auto itr = isotrkCalib->begin(); itr != isotrkCalib->end(); ++itr) {
      if (itr->p_ < pTrackLow_) {
        low = true;
      } else if (itr->p_ > pTrackHigh_) {
        high = true;
      } else {
        inRange = true;
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalIsoTrack") << "AlCaHcalIsotrkFilter::Finds " << isotrkCalib->size()
                                     << " entries in HcalIsoTrkCalibVariables collection with inRange " << inRange
                                     << " low " << low << " high " << high;
#endif
    if (low)
      ++(runCache(iEvent.getRun().index())->nLow_);
    if (high)
      ++(runCache(iEvent.getRun().index())->nHigh_);
    if (inRange) {
      accept = true;
    } else {
      if (low) {
        if (prescaleLow_ <= 1)
          accept = true;
        else if (runCache(iEvent.getRun().index())->nLow_ % prescaleLow_ == 1)
          accept = true;
      }
      if (high) {
        if (prescaleHigh_ <= 1)
          accept = true;
        else if (runCache(iEvent.getRun().index())->nHigh_ % prescaleHigh_ == 1)
          accept = true;
      }
    }
  } else {
    edm::LogVerbatim("HcalIsoTrack") << "AlCaHcalIsotrkFilter::Cannot find the collection for HcalIsoTrkCalibVariables";
  }

  // Return the acceptance flag
  if (accept)
    ++(runCache(iEvent.getRun().index())->nGood_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "AlCaHcalIsotrkFilter::Accept flag " << accept << " All "
                                   << runCache(iEvent.getRun().index())->nAll_ << " Good "
                                   << runCache(iEvent.getRun().index())->nGood_ << " Low "
                                   << runCache(iEvent.getRun().index())->nLow_ << " High "
                                   << runCache(iEvent.getRun().index())->nHigh_;
#endif
  return accept;

}  // AlCaHcalIsotrkFilter::filter

// ------------ method called when starting to processes a run  ------------
std::shared_ptr<alcaHcalIsoTrk::Counters> AlCaHcalIsotrkFilter::globalBeginRun(edm::Run const& iRun,
                                                                               edm::EventSetup const&) const {
  edm::LogVerbatim("HcalIsoTrack") << "Start the Run " << iRun.run();
  return std::make_shared<alcaHcalIsoTrk::Counters>();
}

// ------------ method called when ending the processing of a run  ------------
void AlCaHcalIsotrkFilter::globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const {
  edm::LogVerbatim("HcalIsoTrack") << "Select " << runCache(iRun.index())->nGood_ << " good events out of "
                                   << runCache(iRun.index())->nAll_ << " total # of events with "
                                   << runCache(iRun.index())->nLow_ << ":" << runCache(iRun.index())->nHigh_
                                   << " events below and above the required range";
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void AlCaHcalIsotrkFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("momentumLow", 40.0);
  desc.add<double>("momentumHigh", 60.0);
  desc.add<int>("prescaleLow", 1);
  desc.add<int>("prescaleHigh", 1);
  desc.add<edm::InputTag>("isoTrackLabel", edm::InputTag("alcaHcalIsotrkProducer", "HcalIsoTrack"));
  descriptions.add("alcaHcalIsotrkFilter", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaHcalIsotrkFilter);
