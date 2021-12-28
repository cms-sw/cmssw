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
  typedef edm::Ref<L1TTTrackCollectionType> L1TTTrackRefType;
  typedef std::vector<L1TTTrackRefType> L1TTTrackRefCollectionType;

  typedef Vertex L1VertexType;
  typedef VertexCollection L1VertexCollectionType;

  explicit L1TrackerEtMissProducer(const edm::ParameterSet&);
  ~L1TrackerEtMissProducer() override;

private:
  virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();

  // ----------member data ---------------------------
  float deltaZ_;  // in cm
  float maxPt_;       // in GeV
  int highPtTracks_;  // saturate or truncate
  bool displaced_;    // prompt/displaced tracks

  vector<double> z0Thresholds_;  // Threshold for track to vertex association
  vector<double> etaRegions_;    // Eta bins for choosing deltaZ threshold
  bool debug_;

  std::string L1MetCollectionName;
  std::string L1ExtendedMetCollectionName;

  const edm::EDGetTokenT<VertexCollection> pvToken_;
  const edm::EDGetTokenT<L1TTTrackRefCollectionType> trackToken_;
};

// constructor//
L1TrackerEtMissProducer::L1TrackerEtMissProducer(const edm::ParameterSet& iConfig)
    : pvToken_(consumes<L1VertexCollectionType>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))),
      trackToken_(consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))) {
  deltaZ_ = (float)iConfig.getParameter<double>("deltaZ");
  maxPt_ = (float)iConfig.getParameter<double>("maxPt");
  highPtTracks_ = iConfig.getParameter<int>("highPtTracks");
  displaced_ = iConfig.getParameter<bool>("displaced");
  z0Thresholds_ = iConfig.getParameter<std::vector<double>>("z0Thresholds");
  etaRegions_ = iConfig.getParameter<std::vector<double>>("etaRegions");

  debug_ = iConfig.getParameter<bool>("debug");

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

  edm::Handle<L1TTTrackRefCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken_, L1TTTrackHandle);

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

  int numqualitytracks = 0;
  int numassoctracks = 0;

  for (const auto & track : *L1TTTrackHandle) {
    float pt = track->momentum().perp();
    float phi = track->momentum().phi();
    float eta = track->momentum().eta();
    float z0 = track->z0();

    if (maxPt_ > 0 && pt > maxPt_) {
      if (highPtTracks_ == 0)
        continue;  // ignore these very high PT tracks: truncate
      if (highPtTracks_ == 1)
        pt = maxPt_;  // saturate
    }

    numqualitytracks++;

    if (!displaced_) {  // if displaced, deltaZ = 3.0 cm, very loose
      // construct deltaZ cut to be based on track eta
      for (unsigned int reg = 0; reg < etaRegions_.size(); reg++) {
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

                                                << "# Tracks after Quality Cuts: " << L1TTTrackHandle->size() << "\n"
                                                << "# Tacks after additional highPt Cuts: " << numqualitytracks << "\n"
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
