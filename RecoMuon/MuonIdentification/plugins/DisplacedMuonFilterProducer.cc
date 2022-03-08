/** \class DisplacedMuonFilterProducer
 *
 *  \author C. Fernandez Madrazo <celia.fernandez.madrazo@cern.ch>
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "RecoMuon/MuonIdentification/plugins/DisplacedMuonFilterProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"


DisplacedMuonFilterProducer::DisplacedMuonFilterProducer(const edm::ParameterSet& iConfig)
    : srcMuons_(iConfig.getParameter<edm::InputTag>("srcMuons")),
      srcMuonToken_(consumes<reco::MuonCollection>(srcMuons_)),
      refMuons_(iConfig.getParameter<edm::InputTag>("refMuons")),
      refMuonToken_(consumes<reco::MuonCollection>(refMuons_)),
      min_Dxy_(iConfig.getParameter<double>("minDxy")),
      min_Dz_(iConfig.getParameter<double>("minDz")),
      min_DeltaR_(iConfig.getParameter<double>("minDeltaR")),
      min_DeltaPt_(iConfig.getParameter<double>("minDeltaPt")),
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

    theTrackDepositName = iConfig.getParameter<edm::InputTag>("TrackIsoDeposits");
    theTrackDepositToken_ = consumes<reco::IsoDepositMap>(theTrackDepositName);

    theJetDepositName = iConfig.getParameter<edm::InputTag>("JetIsoDeposits");
    theJetDepositToken_ = consumes<reco::IsoDepositMap>(theJetDepositName);

    theEcalDepositName = iConfig.getParameter<edm::InputTag>("EcalIsoDeposits");
    theEcalDepositToken_ = consumes<reco::IsoDepositMap>(theEcalDepositName);

    theHcalDepositName = iConfig.getParameter<edm::InputTag>("HcalIsoDeposits");
    theHcalDepositToken_ = consumes<reco::IsoDepositMap>(theHcalDepositName);

    theHoDepositName = iConfig.getParameter<edm::InputTag>("HoIsoDeposits");
    theHoDepositToken_ = consumes<reco::IsoDepositMap>(theHoDepositName);

    produces<reco::IsoDepositMap>("tracker");

    produces<reco::IsoDepositMap>("ecal");

    produces<reco::IsoDepositMap>("hcal");

    produces<reco::IsoDepositMap>("ho");

    produces<reco::IsoDepositMap>("jets");
  }
  

}

DisplacedMuonFilterProducer::~DisplacedMuonFilterProducer() {}
void DisplacedMuonFilterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcMuons", edm::InputTag("displacedMuons1stStep"));
  desc.add<edm::InputTag>("refMuons", edm::InputTag("muons1stStep"));
  desc.add<double>("minDxy", 0.);
  desc.add<double>("minDz", 0.);
  desc.add<double>("minDeltaR", 0.);
  desc.add<double>("minDeltaPt", 0.);
  desc.add<edm::InputTag>("TrackIsoDeposits", edm::InputTag("displacedMuons1stStep:tracker"));
  desc.add<edm::InputTag>("EcalIsoDeposits", edm::InputTag("displacedMuons1stStep:ecal"));
  desc.add<edm::InputTag>("HcalIsoDeposits", edm::InputTag("displacedMuons1stStep:hcal"));
  desc.add<edm::InputTag>("HoIsoDeposits", edm::InputTag("displacedMuons1stStep:ho"));
  desc.add<edm::InputTag>("JetIsoDeposits", edm::InputTag("displacedMuons1stStep:jets"));
  desc.add<bool>("FillTimingInfo", true);
  desc.add<bool>("FillDetectorBasedIsolation", true);
  descriptions.addWithDefaultLabel(desc);
}

// Filter muons

void DisplacedMuonFilterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
    // save muon if it is displaced enough
    if ( fabs(muon.bestTrack()->dxy()) > min_Dxy_ || fabs(muon.bestTrack()->dz()) > min_Dz_) { 
      continue;
    }
    // look for overlapping muons if not
    for (unsigned int j = 0; j < refMuons->size(); j++) {
      const reco::Muon& ref(refMuons->at(j));
      double dR = deltaR(muon.eta(), muon.phi(), ref.eta(), ref.phi() );
      double dPt = fabs(muon.pt() - ref.pt());
      if (dR < min_DeltaR_ && dPt < min_DeltaPt_) {
        filteredmuons[i] = false;
        oMuons = oMuons - 1;
        break;
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

  fillMuonMap<reco::MuonTimeExtra>(iEvent, muonHandle, combinedTimeColl, "combined");
  fillMuonMap<reco::MuonTimeExtra>(iEvent, muonHandle, dtTimeColl, "dt");
  fillMuonMap<reco::MuonTimeExtra>(iEvent, muonHandle, cscTimeColl, "csc");

  fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, trackDepColl, "tracker");
  fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, jetDepColl, "jets");
  fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, ecalDepColl, "ecal");
  fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, hcalDepColl, "hcal");
  fillMuonMap<reco::IsoDeposit>(iEvent, muonHandle, hoDepColl, "ho");

}


template <typename TYPE>
void DisplacedMuonFilterProducer::fillMuonMap(edm::Event& event,
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

