// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013
// Modified by Emily MacDonald, 30 Nov 2018
// Modified by Christopher Brown 27 March 2021

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMissFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// detector geometry
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using namespace l1t;

class L1TrackerEtMissProducer : public edm::stream::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;

  typedef Vertex L1VertexType;
  typedef VertexCollection L1VertexCollectionType;

  explicit L1TrackerEtMissProducer(const edm::ParameterSet&);
  ~L1TrackerEtMissProducer() override;

private:
  virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();

  // ----------member data ---------------------------
  const float maxZ0_;  // in cm
  float deltaZ_;       // in cm
  const float Chi2RphidofMax_;
  const float Chi2RzdofMax_;
  const float bendChi2Max_;
  const float minPt_;  // in GeV
  const int nStubsmin_;
  const int nPSStubsMin_;  // minimum number of stubs in PS modules
  const float maxPt_;
  const float maxEta_;
  const int highPtTracks_;       // saturate or truncate
  const bool displaced_;         // prompt/displaced tracks
  vector<double> z0Thresholds_;  // Threshold for track to vertex association
  vector<double> etaRegions_;    // Eta bins for choosing deltaZ threshold
  bool debug_;

  std::string L1MetCollectionName;
  std::string L1ExtendedMetCollectionName;

  const edm::EDGetTokenT<VertexCollection> pvToken_;
  const edm::EDGetTokenT<L1TTTrackCollectionType> trackToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
};

// constructor//
L1TrackerEtMissProducer::L1TrackerEtMissProducer(const edm::ParameterSet& iConfig)
    : maxZ0_((float)iConfig.getParameter<double>("maxZ0")),
      deltaZ_((float)iConfig.getParameter<double>("deltaZ")),
      Chi2RphidofMax_((float)iConfig.getParameter<double>("chi2rphidofMax")),
      Chi2RzdofMax_((float)iConfig.getParameter<double>("chi2rzdofMax")),
      bendChi2Max_((float)iConfig.getParameter<double>("bendChi2Max")),
      minPt_((float)iConfig.getParameter<double>("minPt")),
      nStubsmin_(iConfig.getParameter<int>("nStubsmin")),
      nPSStubsMin_(iConfig.getParameter<int>("nPSStubsMin")),
      maxPt_((float)iConfig.getParameter<double>("maxPt")),
      maxEta_((float)iConfig.getParameter<double>("maxEta")),
      highPtTracks_(iConfig.getParameter<int>("highPtTracks")),
      displaced_(iConfig.getParameter<bool>("displaced")),
      z0Thresholds_(iConfig.getParameter<std::vector<double>>("z0Thresholds")),
      etaRegions_(iConfig.getParameter<std::vector<double>>("etaRegions")),
      debug_(iConfig.getParameter<bool>("debug")),
      pvToken_(consumes<L1VertexCollectionType>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))),
      trackToken_(consumes<L1TTTrackCollectionType>(iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))) {
  L1MetCollectionName = (std::string)iConfig.getParameter<std::string>("L1MetCollectionName");

  if (displaced_) {
    L1ExtendedMetCollectionName = (std::string)iConfig.getParameter<std::string>("L1MetExtendedCollectionName");
    produces<TkEtMissCollection>(L1ExtendedMetCollectionName);
  } else
    produces<TkEtMissCollection>(L1MetCollectionName);
}

L1TrackerEtMissProducer::~L1TrackerEtMissProducer() {}

void L1TrackerEtMissProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<TkEtMissCollection> METCollection(new TkEtMissCollection);

  // Tracker Topology
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);

  edm::Handle<L1VertexCollectionType> L1VertexHandle;
  iEvent.getByToken(pvToken_, L1VertexHandle);

  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken_, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trackIter;

  if (!L1VertexHandle.isValid()) {
    LogError("L1TrackerEtMissProducer") << "\nWarning: VertexCollection not found in the event. Exit\n";
    return;
  }

  if (!L1TTTrackHandle.isValid()) {
    LogError("L1TrackerEtMissProducer") << "\nWarning: L1TTTrackCollection not found in the event. Exit\n";
    return;
  }

  float sumPx = 0;
  float sumPy = 0;
  float etTot = 0;
  double sumPx_PU = 0;
  double sumPy_PU = 0;
  double etTot_PU = 0;
  float zVTX = L1VertexHandle->begin()->z0();

  int numtracks = 0;
  int numqualitytracks = 0;
  int numassoctracks = 0;

  for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {
    numtracks++;
    float pt = trackIter->momentum().perp();
    float phi = trackIter->momentum().phi();
    float eta = trackIter->momentum().eta();
    float chi2rphidof = trackIter->chi2XYRed();
    float chi2rzdof = trackIter->chi2ZRed();
    float bendChi2 = trackIter->stubPtConsistency();
    float z0 = trackIter->z0();
    //unsigned int Sector = trackIter->phiSector();
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        theStubs = trackIter->getStubRefs();
    int nstubs = (int)theStubs.size();

    if (pt < minPt_)
      continue;
    if (std::abs(z0) > maxZ0_)
      continue;
    if (std::abs(eta) > maxEta_)
      continue;
    if (chi2rphidof > Chi2RphidofMax_)
      continue;
    if (chi2rzdof > Chi2RzdofMax_)
      continue;
    if (bendChi2 > bendChi2Max_)
      continue;

    if (maxPt_ > 0 && pt > maxPt_) {
      if (highPtTracks_ == 0)
        continue;  // ignore these very high PT tracks: truncate
      if (highPtTracks_ == 1)
        pt = maxPt_;  // saturate
    }

    int nPS = 0;  // number of stubs in PS modules
    // loop over the stubs
    for (unsigned int istub = 0; istub < (unsigned int)theStubs.size(); istub++) {
      DetId detId(theStubs.at(istub)->getDetId());
      if (detId.det() == DetId::Detector::Tracker) {
        if ((detId.subdetId() == StripSubdetector::TOB && tTopo.tobLayer(detId) <= 3) ||
            (detId.subdetId() == StripSubdetector::TID && tTopo.tidRing(detId) <= 9))
          nPS++;
      }
    }

    if (nstubs < nStubsmin_)
      continue;
    if (nPS < nPSStubsMin_)
      continue;

    numqualitytracks++;

    if (!displaced_) {  // if displaced, deltaZ = 3.0 cm, very loose
      // construct deltaZ cut to be based on track eta
      for (unsigned int reg = 0; reg < etaRegions_.size() - 1; reg++) {
        if (std::abs(eta) >= etaRegions_[reg] && std::abs(eta) < etaRegions_[reg + 1]) {
          deltaZ_ = z0Thresholds_[reg];
          break;
        }
      }
      if (std::abs(eta) >= etaRegions_[etaRegions_.size() - 1]) {
        deltaZ_ = z0Thresholds_[etaRegions_.size() - 1];
        break;
      }
    }

    if (std::abs(z0 - zVTX) <= deltaZ_) {
      numassoctracks++;
      sumPx += pt * cos(phi);
      sumPy += pt * sin(phi);
      etTot += pt;
    } else {  // PU sums
      sumPx_PU += pt * cos(phi);
      sumPy_PU += pt * sin(phi);
      etTot_PU += pt;
    }
  }  // end loop over tracks

  float et = sqrt(sumPx * sumPx + sumPy * sumPy);
  double etphi = atan2(sumPy, sumPx);

  math::XYZTLorentzVector missingEt(-sumPx, -sumPy, 0, et);

  if (debug_) {
    edm::LogVerbatim("L1TrackerEtMissProducer") << "====Global Pt===="
                                                << "\n"
                                                << "Px: " << sumPx << "| Py: " << sumPy << "\n"
                                                << "====MET==="
                                                << "\n"
                                                << "MET: " << et << "| Phi: " << etphi << "\n"

                                                << "# Intial Tracks: " << numtracks << "\n"
                                                << "# Tracks after Quality Cuts: " << numqualitytracks << "\n"
                                                << "# Tracks Associated to Vertex: " << numassoctracks << "\n"
                                                << "========================================================"
                                                << "\n";
  }

  int ibx = 0;
  METCollection->push_back(TkEtMiss(missingEt, TkEtMiss::kMET, etphi, numassoctracks, ibx));

  if (displaced_)
    iEvent.put(std::move(METCollection), L1ExtendedMetCollectionName);
  else
    iEvent.put(std::move(METCollection), L1MetCollectionName);
}  // end producer

void L1TrackerEtMissProducer::beginJob() {}

void L1TrackerEtMissProducer::endJob() {}

DEFINE_FWK_MODULE(L1TrackerEtMissProducer);
