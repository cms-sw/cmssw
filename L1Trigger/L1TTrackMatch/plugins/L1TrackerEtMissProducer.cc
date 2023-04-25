// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013
// Modified by Emily MacDonald, 30 Nov 2018
// Modified by Christopher Brown 27 March 2021

// system include files
#include <algorithm>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMissFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace l1t;

class L1TrackerEtMissProducer : public edm::global::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;
  typedef edm::RefVector<L1TTTrackCollectionType> L1TTTrackRefCollectionType;

  explicit L1TrackerEtMissProducer(const edm::ParameterSet&);
  ~L1TrackerEtMissProducer() override = default;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<L1TTTrackRefCollectionType> trackToken_;
  const edm::EDGetTokenT<L1TTTrackRefCollectionType> vtxAssocTrackToken_;
  const std::string L1MetCollectionName;
  const float maxPt_;       // in GeV
  const int highPtTracks_;  // saturate or truncate
  const bool debug_;
};

// constructor
L1TrackerEtMissProducer::L1TrackerEtMissProducer(const edm::ParameterSet& iConfig)
    : trackToken_(consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      vtxAssocTrackToken_(
          consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<edm::InputTag>("L1TrackAssociatedInputTag"))),
      L1MetCollectionName(iConfig.getParameter<std::string>("L1MetCollectionName")),
      maxPt_(iConfig.getParameter<double>("maxPt")),
      highPtTracks_(iConfig.getParameter<int>("highPtTracks")),
      debug_(iConfig.getParameter<bool>("debug")) {
  produces<TkEtMissCollection>(L1MetCollectionName);
}

void L1TrackerEtMissProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  std::unique_ptr<TkEtMissCollection> METCollection(new TkEtMissCollection);

  edm::Handle<L1TTTrackRefCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken_, L1TTTrackHandle);

  edm::Handle<L1TTTrackRefCollectionType> L1TTTrackAssociatedHandle;
  iEvent.getByToken(vtxAssocTrackToken_, L1TTTrackAssociatedHandle);

  if (!L1TTTrackHandle.isValid()) {
    LogError("L1TrackerEtMissProducer") << "\nWarning: L1TTTrackCollection not found in the event. Exit\n";
    return;
  }

  if (!L1TTTrackAssociatedHandle.isValid()) {
    LogError("L1TrackerEtMissProducer") << "\nWarning: L1TTTrackAssociatedCollection not found in the event. Exit\n";
    return;
  }

  float sumPx = 0;
  float sumPy = 0;

  int numqualitytracks = 0;
  int numassoctracks = 0;

  for (const auto& track : *L1TTTrackHandle) {
    float pt = track->momentum().perp();
    float phi = track->momentum().phi();

    if (maxPt_ > 0 && pt > maxPt_) {
      if (highPtTracks_ == 0)
        continue;  // ignore these very high PT tracks: truncate
      if (highPtTracks_ == 1)
        pt = maxPt_;  // saturate
    }

    numqualitytracks++;

    if (std::find(L1TTTrackAssociatedHandle->begin(), L1TTTrackAssociatedHandle->end(), track) !=
        L1TTTrackAssociatedHandle->end()) {
      numassoctracks++;
      sumPx += pt * cos(phi);
      sumPy += pt * sin(phi);
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

                                                << "# Tracks after quality cuts: " << L1TTTrackHandle->size() << "\n"
                                                << "# Tacks after additional highPt Cuts: " << numqualitytracks << "\n"
                                                << "# Tracks associated to vertex: " << numassoctracks << "\n"
                                                << "========================================================"
                                                << "\n";
  }

  int ibx = 0;
  METCollection->push_back(TkEtMiss(missingEt, TkEtMiss::kMET, etphi, numassoctracks, ibx));

  iEvent.put(std::move(METCollection), L1MetCollectionName);
}  // end producer

DEFINE_FWK_MODULE(L1TrackerEtMissProducer);
