/** \class DisplacedMuonFilterProducer
 *
 * The filter takes a reco::Muon collection as an input and preselects
 * the muons that will be processed by the next sequences.
 *
 * 1) StandAlone muons matched to an inner track (either as Tracker or Global muons)
 *    are preselected if the StandAlone track has pt > minPtSTA_ or the tracker track 
 *    has pt > minPtTK_
 *
 *    (Global muon are contained in this subset)
 *
 * 2) StandAlone-only muons are preselected if they have number of segments > minMatches_
 *    and they have pt > minPtSTA_
 *
 * 3) Tracker muons without an StandAlone track are preselected if they have pt > minPtTK_
 *    and they are labelled as isTrackerMuon() i.e. not RPC/GEM muons.
 *
 * \author C. Fernandez Madrazo <celia.fernandez.madrazo@cern.ch>
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

#include "DataFormats/Common/interface/Handle.h"

namespace pat {
  class DisplacedMuonFilterProducer : public edm::stream::EDProducer<> {
  public:
    explicit DisplacedMuonFilterProducer(const edm::ParameterSet&);

    ~DisplacedMuonFilterProducer() override;

    void produce(edm::Event&, const edm::EventSetup&) override;
    /// description of config file parameters
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    template <typename TYPE>
    void fillMuonMap(edm::Event& event,
                     const edm::OrphanHandle<reco::MuonCollection>& muonHandle,
                     const std::vector<TYPE>& muonExtra,
                     const std::string& label);

    // muon collections
    const edm::InputTag srcMuons_;
    const edm::EDGetTokenT<reco::MuonCollection> srcMuonToken_;

    // filter criteria and selection
    const double minPtTK_;     // Minimum pt of inner tracks
    const double minPtSTA_;    // Minimum pt of standalone tracks
    const double minMatches_;  // Minimum number of matches of standalone-only muons

    // what information to fill
    bool fillDetectorBasedIsolation_;
    bool fillTimingInfo_;

    // timing info
    edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapCmbToken_;
    edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapDTToken_;
    edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapCSCToken_;

    // detector based isolation
    edm::EDGetTokenT<reco::IsoDepositMap> theTrackDepositToken_;
    edm::EDGetTokenT<reco::IsoDepositMap> theEcalDepositToken_;
    edm::EDGetTokenT<reco::IsoDepositMap> theHcalDepositToken_;
    edm::EDGetTokenT<reco::IsoDepositMap> theHoDepositToken_;
    edm::EDGetTokenT<reco::IsoDepositMap> theJetDepositToken_;
  };
}  // namespace pat

pat::DisplacedMuonFilterProducer::DisplacedMuonFilterProducer(const edm::ParameterSet& iConfig)
    : srcMuons_(iConfig.getParameter<edm::InputTag>("srcMuons")),
      srcMuonToken_(consumes<reco::MuonCollection>(srcMuons_)),
      minPtTK_(iConfig.getParameter<double>("minPtTK")),
      minPtSTA_(iConfig.getParameter<double>("minPtSTA")),
      minMatches_(iConfig.getParameter<double>("minMatches")),
      fillDetectorBasedIsolation_(iConfig.getParameter<bool>("FillDetectorBasedIsolation")),
      fillTimingInfo_(iConfig.getParameter<bool>("FillTimingInfo")) {
  produces<reco::MuonCollection>();

  if (fillTimingInfo_) {
    timeMapCmbToken_ = consumes<reco::MuonTimeExtraMap>(edm::InputTag(srcMuons_.label(), "combined"));
    timeMapDTToken_ = consumes<reco::MuonTimeExtraMap>(edm::InputTag(srcMuons_.label(), "dt"));
    timeMapCSCToken_ = consumes<reco::MuonTimeExtraMap>(edm::InputTag(srcMuons_.label(), "csc"));

    produces<reco::MuonTimeExtraMap>("combined");
    produces<reco::MuonTimeExtraMap>("dt");
    produces<reco::MuonTimeExtraMap>("csc");
  }

  if (fillDetectorBasedIsolation_) {
    theTrackDepositToken_ = consumes<reco::IsoDepositMap>(iConfig.getParameter<edm::InputTag>("TrackIsoDeposits"));
    theJetDepositToken_ = consumes<reco::IsoDepositMap>(iConfig.getParameter<edm::InputTag>("JetIsoDeposits"));
    theEcalDepositToken_ = consumes<reco::IsoDepositMap>(iConfig.getParameter<edm::InputTag>("EcalIsoDeposits"));
    theHcalDepositToken_ = consumes<reco::IsoDepositMap>(iConfig.getParameter<edm::InputTag>("HcalIsoDeposits"));
    theHoDepositToken_ = consumes<reco::IsoDepositMap>(iConfig.getParameter<edm::InputTag>("HoIsoDeposits"));
    produces<reco::IsoDepositMap>("tracker");
    produces<reco::IsoDepositMap>("ecal");
    produces<reco::IsoDepositMap>("hcal");
    produces<reco::IsoDepositMap>("ho");
    produces<reco::IsoDepositMap>("jets");
  }
}

pat::DisplacedMuonFilterProducer::~DisplacedMuonFilterProducer() {}

void pat::DisplacedMuonFilterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // filteredDisplacedMuons
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcMuons", edm::InputTag("displacedMuons"));
  desc.add<bool>("FillTimingInfo", true);
  desc.add<bool>("FillDetectorBasedIsolation", false);
  desc.add<edm::InputTag>("TrackIsoDeposits", edm::InputTag("displacedMuons", "tracker"));
  desc.add<edm::InputTag>("JetIsoDeposits", edm::InputTag("displacedMuons", "jets"));
  desc.add<edm::InputTag>("EcalIsoDeposits", edm::InputTag("displacedMuons", "ecal"));
  desc.add<edm::InputTag>("HcalIsoDeposits", edm::InputTag("displacedMuons", "hcal"));
  desc.add<edm::InputTag>("HoIsoDeposits", edm::InputTag("displacedMuons", "ho"));
  desc.add<double>("minPtSTA", 3.5);
  desc.add<double>("minPtTK", 3.5);
  desc.add<double>("minMatches", 2);
  descriptions.add("filteredDisplacedMuons", desc);
}

// Filter muons

void pat::DisplacedMuonFilterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto output = std::make_unique<reco::MuonCollection>();

  // muon collections
  edm::Handle<reco::MuonCollection> srcMuons;
  iEvent.getByToken(srcMuonToken_, srcMuons);

  int nMuons = srcMuons->size();

  // filter the muons
  std::vector<bool> filteredmuons(nMuons, true);
  int oMuons = nMuons;

  unsigned int nsegments = 0;
  for (unsigned int i = 0; i < srcMuons->size(); i++) {
    const reco::Muon& muon(srcMuons->at(i));

    if (muon.isStandAloneMuon()) {
      if (muon.innerTrack().isNonnull()) {
        // Discard STA + GL/TR muons that are below pt threshold
        if (muon.innerTrack()->pt() < minPtTK_ && muon.standAloneMuon()->pt() < minPtSTA_) {
          filteredmuons[i] = false;
          oMuons = oMuons - 1;
          continue;
        }
        // Discard STA-only muons below pt threshold
      } else if (muon.standAloneMuon()->pt() < minPtSTA_) {
        filteredmuons[i] = false;
        oMuons = oMuons - 1;
        continue;
      } else {
        // Compute number of DT+CSC segments and discard those that have less than the minimum required
        nsegments = 0;
        for (trackingRecHit_iterator hit = muon.standAloneMuon()->recHitsBegin();
             hit != muon.standAloneMuon()->recHitsEnd();
             ++hit) {
          if (!(*hit)->isValid())
            continue;
          DetId id = (*hit)->geographicalId();
          if (id.det() != DetId::Muon)
            continue;
          if (id.subdetId() == MuonSubdetId::DT || id.subdetId() == MuonSubdetId::CSC) {
            nsegments++;
          }
        }
        // Discard STA-only muons with less than minMatches_ segments
        if (nsegments < minMatches_) {
          filteredmuons[i] = false;
          oMuons = oMuons - 1;
          continue;
        }
      }
    } else {
      if (muon.innerTrack().isNonnull()) {
        if (muon.innerTrack()->pt() < minPtTK_ || !muon.isTrackerMuon()) {
          filteredmuons[i] = false;
          oMuons = oMuons - 1;
          continue;
        }
      } else {  // Should never happen
        edm::LogWarning("muonBadTracks") << "Muon that has not standalone nor tracker track."
                                         << "There should be no such object. Muon is skipped.";
        filteredmuons[i] = false;
        oMuons = oMuons - 1;
        continue;
      }
    }
  }

  // timing information
  edm::Handle<reco::MuonTimeExtraMap> timeMapCmb;
  edm::Handle<reco::MuonTimeExtraMap> timeMapDT;
  edm::Handle<reco::MuonTimeExtraMap> timeMapCSC;

  std::vector<reco::MuonTimeExtra> dtTimeColl(oMuons);
  std::vector<reco::MuonTimeExtra> cscTimeColl(oMuons);
  std::vector<reco::MuonTimeExtra> combinedTimeColl(oMuons);

  if (fillTimingInfo_) {
    iEvent.getByToken(timeMapCmbToken_, timeMapCmb);
    iEvent.getByToken(timeMapDTToken_, timeMapDT);
    iEvent.getByToken(timeMapCSCToken_, timeMapCSC);
  }

  // detector based isolation
  std::vector<reco::IsoDeposit> trackDepColl(oMuons);
  std::vector<reco::IsoDeposit> ecalDepColl(oMuons);
  std::vector<reco::IsoDeposit> hcalDepColl(oMuons);
  std::vector<reco::IsoDeposit> hoDepColl(oMuons);
  std::vector<reco::IsoDeposit> jetDepColl(oMuons);

  edm::Handle<reco::IsoDepositMap> trackIsoDepMap;
  edm::Handle<reco::IsoDepositMap> ecalIsoDepMap;
  edm::Handle<reco::IsoDepositMap> hcalIsoDepMap;
  edm::Handle<reco::IsoDepositMap> hoIsoDepMap;
  edm::Handle<reco::IsoDepositMap> jetIsoDepMap;

  if (fillDetectorBasedIsolation_) {
    iEvent.getByToken(theTrackDepositToken_, trackIsoDepMap);
    iEvent.getByToken(theEcalDepositToken_, ecalIsoDepMap);
    iEvent.getByToken(theHcalDepositToken_, hcalIsoDepMap);
    iEvent.getByToken(theHoDepositToken_, hoIsoDepMap);
    iEvent.getByToken(theJetDepositToken_, jetIsoDepMap);
  }

  // save filtered muons
  unsigned int k = 0;
  for (unsigned int i = 0; i < srcMuons->size(); i++) {
    if (filteredmuons[i]) {
      const reco::Muon& inMuon(srcMuons->at(i));
      reco::MuonRef muRef(srcMuons, i);

      // Copy the muon
      reco::Muon outMuon = inMuon;

      // Fill timing information
      if (fillTimingInfo_) {
        combinedTimeColl[k] = (*timeMapCmb)[muRef];
        dtTimeColl[k] = (*timeMapDT)[muRef];
        cscTimeColl[k] = (*timeMapCSC)[muRef];
      }

      // Fill detector based isolation
      if (fillDetectorBasedIsolation_) {
        trackDepColl[k] = (*trackIsoDepMap)[muRef];
        ecalDepColl[k] = (*ecalIsoDepMap)[muRef];
        hcalDepColl[k] = (*hcalIsoDepMap)[muRef];
        hoDepColl[k] = (*hoIsoDepMap)[muRef];
        jetDepColl[k] = (*jetIsoDepMap)[muRef];
      }

      output->push_back(outMuon);
      k++;
    }
  }

  // fill information
  edm::OrphanHandle<reco::MuonCollection> muonHandle = iEvent.put(std::move(output));

  if (fillTimingInfo_) {
    fillMuonMap<reco::MuonTimeExtra>(iEvent, muonHandle, combinedTimeColl, "combined");
    fillMuonMap<reco::MuonTimeExtra>(iEvent, muonHandle, dtTimeColl, "dt");
    fillMuonMap<reco::MuonTimeExtra>(iEvent, muonHandle, cscTimeColl, "csc");
  }

  if (fillDetectorBasedIsolation_) {
    fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, trackDepColl, "tracker");
    fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, jetDepColl, "jets");
    fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, ecalDepColl, "ecal");
    fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, hcalDepColl, "hcal");
    fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, hoDepColl, "ho");
  }
}

template <typename TYPE>
void pat::DisplacedMuonFilterProducer::fillMuonMap(edm::Event& event,
                                                   const edm::OrphanHandle<reco::MuonCollection>& muonHandle,
                                                   const std::vector<TYPE>& muonExtra,
                                                   const std::string& label) {
  typedef typename edm::ValueMap<TYPE>::Filler FILLER;

  auto muonMap = std::make_unique<edm::ValueMap<TYPE>>();
  if (!muonExtra.empty()) {
    FILLER filler(*muonMap);
    filler.insert(muonHandle, muonExtra.begin(), muonExtra.end());
    filler.fill();
  }
  event.put(std::move(muonMap), label);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using pat::DisplacedMuonFilterProducer;
DEFINE_FWK_MODULE(DisplacedMuonFilterProducer);
