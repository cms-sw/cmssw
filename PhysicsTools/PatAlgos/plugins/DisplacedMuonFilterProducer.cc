/** \class DisplacedMuonFilterProducer
 *
 *  \author C. Fernandez Madrazo <celia.fernandez.madrazo@cern.ch>
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"

namespace pat {
  class DisplacedMuonFilterProducer : public edm::stream::EDProducer<> {
  public:
    explicit DisplacedMuonFilterProducer(const edm::ParameterSet&);

    ~DisplacedMuonFilterProducer() override;

    void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    template <typename TYPE>
    void fillMuonMap(edm::Event& event,
                     const edm::OrphanHandle<reco::MuonCollection>& muonHandle,
                     const std::vector<TYPE>& muonExtra,
                     const std::string& label);

    // muon collections
    const edm::InputTag srcMuons_;
    const edm::EDGetTokenT<reco::MuonCollection> srcMuonToken_;

    const edm::InputTag refMuons_;
    const edm::EDGetTokenT<reco::MuonCollection> refMuonToken_;

    // filter criteria and selection (displacedTracker muons)
    const double min_Dxy_;             // Minimum dxy
    const double min_Dz_;              // Minimum dz
    const double min_DeltaR_;          // dR difference between displaced-prompt muon
    const double min_RelDeltaPt_;      // Rel. pT difference between displaced-prompt

    // what information to fill
    bool fillDetectorBasedIsolation_;
    bool fillTimingInfo_;

    // timing info
    edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapCmbToken_;
    edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapDTToken_;
    edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapCSCToken_;


    // detector based isolation
    edm::InputTag theTrackDepositName;
    edm::InputTag theEcalDepositName;
    edm::InputTag theHcalDepositName;
    edm::InputTag theHoDepositName;
    edm::InputTag theJetDepositName;

    std::string trackDepositName_;
    std::string ecalDepositName_;
    std::string hcalDepositName_;
    std::string hoDepositName_;
    std::string jetDepositName_;

    edm::EDGetTokenT<reco::IsoDepositMap> theTrackDepositToken_;
    edm::EDGetTokenT<reco::IsoDepositMap> theEcalDepositToken_;
    edm::EDGetTokenT<reco::IsoDepositMap> theHcalDepositToken_;
    edm::EDGetTokenT<reco::IsoDepositMap> theHoDepositToken_;
    edm::EDGetTokenT<reco::IsoDepositMap> theJetDepositToken_;

  };
} // namespace path

pat::DisplacedMuonFilterProducer::DisplacedMuonFilterProducer(const edm::ParameterSet& iConfig)
    : srcMuons_(iConfig.getParameter<edm::InputTag>("srcMuons")),
      srcMuonToken_(consumes<reco::MuonCollection>(srcMuons_)),
      refMuons_(iConfig.getParameter<edm::InputTag>("refMuons")),
      refMuonToken_(consumes<reco::MuonCollection>(refMuons_)),
      min_Dxy_(iConfig.getParameter<double>("minDxy")),
      min_Dz_(iConfig.getParameter<double>("minDz")),
      min_DeltaR_(iConfig.getParameter<double>("minDeltaR")),
      min_RelDeltaPt_(iConfig.getParameter<double>("minRelDeltaPt")),
      fillDetectorBasedIsolation_(iConfig.getParameter<bool>("FillDetectorBasedIsolation")),
      fillTimingInfo_(iConfig.getParameter<bool>("FillTimingInfo")) {

  produces<reco::MuonCollection>();

  if (fillTimingInfo_) {
    timeMapCmbToken_ = consumes<reco::MuonTimeExtraMap>(edm::InputTag(srcMuons_.label(), "combined"));
    timeMapDTToken_  = consumes<reco::MuonTimeExtraMap>(edm::InputTag(srcMuons_.label(), "dt"));
    timeMapCSCToken_ = consumes<reco::MuonTimeExtraMap>(edm::InputTag(srcMuons_.label(), "csc"));

    produces<reco::MuonTimeExtraMap>("combined");
    produces<reco::MuonTimeExtraMap>("dt");
    produces<reco::MuonTimeExtraMap>("csc");
  }

  
  if (fillDetectorBasedIsolation_) {

    theTrackDepositName   = iConfig.getParameter<edm::InputTag>("TrackIsoDeposits");
    theTrackDepositToken_ = consumes<reco::IsoDepositMap>(theTrackDepositName);

    theJetDepositName     = iConfig.getParameter<edm::InputTag>("JetIsoDeposits");
    theJetDepositToken_   = consumes<reco::IsoDepositMap>(theJetDepositName);

    theEcalDepositName    = iConfig.getParameter<edm::InputTag>("EcalIsoDeposits");
    theEcalDepositToken_  = consumes<reco::IsoDepositMap>(theEcalDepositName);

    theHcalDepositName    = iConfig.getParameter<edm::InputTag>("HcalIsoDeposits");
    theHcalDepositToken_  = consumes<reco::IsoDepositMap>(theHcalDepositName);

    theHoDepositName      = iConfig.getParameter<edm::InputTag>("HoIsoDeposits");
    theHoDepositToken_    = consumes<reco::IsoDepositMap>(theHoDepositName);

    produces<reco::IsoDepositMap>("tracker");

    produces<reco::IsoDepositMap>("ecal");

    produces<reco::IsoDepositMap>("hcal");

    produces<reco::IsoDepositMap>("ho");

    produces<reco::IsoDepositMap>("jets");
  }
  
}

pat::DisplacedMuonFilterProducer::~DisplacedMuonFilterProducer() {}

// Filter muons

void pat::DisplacedMuonFilterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto output = std::make_unique<reco::MuonCollection>();


  // muon collections
  edm::Handle<reco::MuonCollection> srcMuons;
  iEvent.getByToken(srcMuonToken_, srcMuons);

  edm::Handle<reco::MuonCollection> refMuons;
  iEvent.getByToken(refMuonToken_, refMuons);

  int nMuons = srcMuons->size();


  // filter the muons
  std::vector<bool> filteredmuons(nMuons, true);
  int oMuons = nMuons;
  for (unsigned int i = 0; i < srcMuons->size(); i++) {
    const reco::Muon& muon(srcMuons->at(i));

    if (muon.isStandAloneMuon()) {
      if (muon.innerTrack().isNonnull())
        continue;
      if (!muon.isMatchesValid() || muon.numberOfMatches() < 2) {
        filteredmuons[i] = false;
        oMuons = oMuons - 1;
      }
    } else {
      // save the muon if its impact parameters are above thresholds
      if ( fabs(muon.bestTrack()->dxy()) > min_Dxy_ && fabs(muon.bestTrack()->dz()) > min_Dz_)
        continue;
      // look for overlapping muons if not
      for (unsigned int j = 0; j < refMuons->size(); j++) {
        const reco::Muon& ref(refMuons->at(j));
        if (!ref.innerTrack().isNonnull())
          continue;
        double dR = deltaR(muon.eta(), muon.phi(), ref.innerTrack()->eta(), ref.innerTrack()->phi() );
        double reldPt = fabs(muon.pt() - ref.innerTrack()->pt())/muon.pt();
        if (dR < min_DeltaR_ && reldPt < min_RelDeltaPt_) {
          filteredmuons[i] = false;
          oMuons = oMuons - 1;
          break;
        }
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
        dtTimeColl[k]       = (*timeMapDT)[muRef];
        cscTimeColl[k]      = (*timeMapCSC)[muRef];
      }

      // Fill detector based isolation
      if (fillDetectorBasedIsolation_) {
        trackDepColl[k] = (*trackIsoDepMap)[muRef];
        ecalDepColl[k]  = (*ecalIsoDepMap)[muRef];
        hcalDepColl[k]  = (*hcalIsoDepMap)[muRef];
        hoDepColl[k]    = (*hoIsoDepMap)[muRef];
        jetDepColl[k]   = (*jetIsoDepMap)[muRef];
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
