// -*- C++ -*-
//
// Package:    L1Trigger/L1TTrackMatch
// Class:      L1TrackerEtMissEmulatorProducer
//
/**\class L1TrackerEtMissEmulatorProducer L1TrackerEtMissEmulatorProducer.cc
 L1Trigger/L1TTrackMatch/plugins/L1TrackerEtMissEmulatorProducer.cc

 Description: Takes L1TTTracks and performs a integer emulation of Track-based
 missing Et, outputting a collection of EtSum 
*/
//
// Original Author:  Christopher Brown
//         Created:  Fri, 19 Feb 2021
//         Updated:  Wed, 16 Jun 2021
//
//

// system include files
#include <memory>
#include <numeric>
#include <sstream>
// user include files
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1TTrackMatch/interface/Cordic.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuAlgo.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuTrackTransform.h"

using namespace l1t;

class L1TrackerEtMissEmulatorProducer : public edm::stream::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;
  typedef edm::RefVector<L1TTTrackCollectionType> L1TTTrackRefCollectionType;
  typedef l1t::VertexWordCollection L1VertexCollectionType;
  typedef l1t::VertexWord L1VertexType;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  explicit L1TrackerEtMissEmulatorProducer(const edm::ParameterSet&);
  ~L1TrackerEtMissEmulatorProducer() override;

private:
  virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();

  // ----------member data ---------------------------

  std::vector<l1tmetemu::global_phi_t> cosLUT_;  // Cos LUT array
  std::vector<l1tmetemu::global_phi_t> phiQuadrants_;
  std::vector<l1tmetemu::global_phi_t> phiShifts_;

  int cordicSteps_;
  int debug_;
  bool cordicDebug_ = false;

  bool GTTinput_ = false;

  L1TkEtMissEmuTrackTransform TrackTransform;

  std::string L1MetCollectionName_;

  const edm::EDGetTokenT<L1VertexCollectionType> pvToken_;
  const edm::EDGetTokenT<L1TTTrackRefCollectionType> trackToken_;
  const edm::EDGetTokenT<L1TTTrackRefCollectionType> vtxAssocTrackToken_;
};

// constructor//
L1TrackerEtMissEmulatorProducer::L1TrackerEtMissEmulatorProducer(const edm::ParameterSet& iConfig)
    : pvToken_(consumes<L1VertexCollectionType>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))),
      trackToken_(consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      vtxAssocTrackToken_(
          consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<edm::InputTag>("L1TrackAssociatedInputTag"))) {
  // Setup LUTs
  TrackTransform.generateLUTs();
  phiQuadrants_ = TrackTransform.getPhiQuad();
  phiShifts_ = TrackTransform.getPhiShift();

  // Get Emulator config parameters
  cordicSteps_ = (int)iConfig.getParameter<int>("nCordicSteps");
  debug_ = (int)iConfig.getParameter<int>("debug");

  GTTinput_ = (bool)iConfig.getParameter<bool>("useGTTinput");

  TrackTransform.setGTTinput(GTTinput_);

  // Name of output ED Product
  L1MetCollectionName_ = (std::string)iConfig.getParameter<std::string>("L1MetCollectionName");

  if (debug_ == 5) {
    cordicDebug_ = true;
  }

  // To have same bin spacing between 0 and pi/2 as between original phi
  // granularity
  int cosLUTbins = floor(l1tmetemu::kMaxCosLUTPhi / l1tmetemu::kStepPhi);

  // Compute LUTs
  cosLUT_ = l1tmetemu::generateCosLUT(cosLUTbins);

  produces<std::vector<EtSum>>(L1MetCollectionName_);
}

L1TrackerEtMissEmulatorProducer::~L1TrackerEtMissEmulatorProducer() {}

void L1TrackerEtMissEmulatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<std::vector<l1t::EtSum>> METCollection(new std::vector<l1t::EtSum>(0));

  edm::Handle<L1VertexCollectionType> L1VertexHandle;
  iEvent.getByToken(pvToken_, L1VertexHandle);

  edm::Handle<L1TTTrackRefCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken_, L1TTTrackHandle);

  edm::Handle<L1TTTrackRefCollectionType> L1TTTrackAssociatedHandle;
  iEvent.getByToken(vtxAssocTrackToken_, L1TTTrackAssociatedHandle);

  // Initialize cordic class
  Cordic cordicSqrt(l1tmetemu::kMETPhiBins, l1tmetemu::kMETSize, cordicSteps_, cordicDebug_);

  if (!L1VertexHandle.isValid()) {
    LogError("L1TrackerEtMissEmulatorProducer") << "\nWarning: VertexCollection not found in the event. Exit\n";
    return;
  }

  if (!L1TTTrackHandle.isValid()) {
    LogError("L1TrackerEtMissEmulatorProducer") << "\nWarning: L1TTTrackCollection not found in the event. Exit\n";
    return;
  }

  if (!L1TTTrackAssociatedHandle.isValid()) {
    LogError("L1TrackerEtMissEmulatorProducer")
        << "\nWarning: L1TTTrackAssociatedCollection not found in the event. Exit\n";
    return;
  }

  // Initialize sector sums, need 0 initialization in case a sector has no
  // tracks
  l1tmetemu::Et_t sumPx[l1tmetemu::kNSector * 2] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  l1tmetemu::Et_t sumPy[l1tmetemu::kNSector * 2] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int link_totals[l1tmetemu::kNSector * 2] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int sector_totals[l1tmetemu::kNSector] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Track counters
  int num_quality_tracks{0};
  int num_assoc_tracks{0};

  // Get reference to first vertex in event vertex collection
  L1VertexType& vtx = const_cast<L1VertexType&>(L1VertexHandle->at(0));

  for (const auto& track : *L1TTTrackHandle) {
    num_quality_tracks++;
    L1TTTrackType& track_ref = const_cast<L1TTTrackType&>(*track);  // Get Reference to track to pass to TrackTransform

    // Convert to internal track representation
    InternalEtWord EtTrack = TrackTransform.transformTrack<L1TTTrackType, L1VertexType>(track_ref, vtx);

    if (std::find(L1TTTrackAssociatedHandle->begin(), L1TTTrackAssociatedHandle->end(), track) !=
        L1TTTrackAssociatedHandle->end()) {
      num_assoc_tracks++;
      if (debug_ == 7) {
        edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
            << "Track to Vertex ID: " << num_quality_tracks << "\n"
            << "Phi Sector: " << EtTrack.Sector << " pT: " << EtTrack.pt << " Phi: " << EtTrack.globalPhi
            << " TanL: " << EtTrack.eta << " Z0: " << EtTrack.z0 << " Nstub: " << EtTrack.nstubs
            << " Chi2rphi: " << EtTrack.chi2rphidof << " Chi2rz: " << EtTrack.chi2rzdof
            << " bendChi2: " << EtTrack.bendChi2 << " PV: " << EtTrack.pV << "\n"
            << "--------------------------------------------------------------\n";
      }

      if (debug_ == 2) {
        edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
            << "========================Phi debug=================================\n"
            << "Int pT: " << EtTrack.pt << "\n"
            << "Int Phi: " << EtTrack.globalPhi << " Float Phi: " << EtTrack.phi
            << " Actual Float Cos(Phi): " << cos(EtTrack.phi) << " Actual Float Sin(Phi): " << sin(EtTrack.phi) << "\n";
      }
      l1tmetemu::Et_t temppx = 0;
      l1tmetemu::Et_t temppy = 0;

      // Split tracks in phi quadrants and access cosLUT_, backwards iteration
      // through cosLUT_ gives sin Sum sector Et -ve when cos or sin phi are -ve
      sector_totals[EtTrack.Sector] += 1;
      if (EtTrack.globalPhi >= phiQuadrants_[0] && EtTrack.globalPhi < phiQuadrants_[1]) {
        temppx = (EtTrack.pt * cosLUT_[EtTrack.globalPhi]);
        temppy = (EtTrack.pt * cosLUT_[phiQuadrants_[1] - 1 - EtTrack.globalPhi]);

        if (debug_ == 2) {
          edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
              << "Sector: " << EtTrack.Sector << " Quadrant: " << 1 << "\n"
              << "Int Phi: " << EtTrack.globalPhi << " Int Cos(Phi): " << cosLUT_[EtTrack.globalPhi]
              << " Int Sin(Phi): " << cosLUT_[phiQuadrants_[1] - 1 - EtTrack.globalPhi] << "\n"

              << "Float Phi: " << (float)EtTrack.globalPhi / l1tmetemu::kGlobalPhiBins
              << " Float Cos(Phi): " << (float)cosLUT_[EtTrack.globalPhi] / l1tmetemu::kGlobalPhiBins
              << " Float Sin(Phi): "
              << (float)cosLUT_[phiQuadrants_[1] - 1 - EtTrack.globalPhi] / l1tmetemu::kGlobalPhiBins << "\n";
        }
      } else if (EtTrack.globalPhi >= phiQuadrants_[1] && EtTrack.globalPhi < phiQuadrants_[2]) {
        temppx = -(EtTrack.pt * cosLUT_[phiQuadrants_[2] - 1 - EtTrack.globalPhi]);
        temppy = (EtTrack.pt * cosLUT_[EtTrack.globalPhi - phiQuadrants_[1]]);

        if (debug_ == 2) {
          edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
              << "Sector: " << EtTrack.Sector << " Quadrant: " << 2 << "\n"
              << "Int Phi: " << EtTrack.globalPhi << " Int Cos(Phi): -"
              << cosLUT_[phiQuadrants_[2] - 1 - EtTrack.globalPhi]
              << " Int Sin(Phi): " << cosLUT_[EtTrack.globalPhi - phiQuadrants_[1]] << "\n"

              << "Float Phi: " << (float)EtTrack.globalPhi / l1tmetemu::kGlobalPhiBins << " Float Cos(Phi): -"
              << (float)cosLUT_[phiQuadrants_[2] - 1 - EtTrack.globalPhi] / l1tmetemu::kGlobalPhiBins
              << " Float Sin(Phi): " << (float)cosLUT_[EtTrack.globalPhi - phiQuadrants_[1]] / l1tmetemu::kGlobalPhiBins
              << "\n";
        }
      } else if (EtTrack.globalPhi >= phiQuadrants_[2] && EtTrack.globalPhi < phiQuadrants_[3]) {
        temppx = -(EtTrack.pt * cosLUT_[EtTrack.globalPhi - phiQuadrants_[2]]);
        temppy = -(EtTrack.pt * cosLUT_[phiQuadrants_[3] - 1 - EtTrack.globalPhi]);

        if (debug_ == 2) {
          edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
              << "Sector: " << EtTrack.Sector << " Quadrant: " << 3 << "\n"
              << "Int Phi: " << EtTrack.globalPhi << " Int Cos(Phi): -" << cosLUT_[EtTrack.globalPhi - phiQuadrants_[2]]
              << " Int Sin(Phi): -" << cosLUT_[phiQuadrants_[3] - 1 - EtTrack.globalPhi] << "\n"

              << "Float Phi: " << (float)EtTrack.globalPhi / l1tmetemu::kGlobalPhiBins << " Float Cos(Phi): -"
              << (float)cosLUT_[EtTrack.globalPhi - phiQuadrants_[2]] / l1tmetemu::kGlobalPhiBins
              << " Float Sin(Phi): -"
              << (float)cosLUT_[phiQuadrants_[3] - 1 - EtTrack.globalPhi] / l1tmetemu::kGlobalPhiBins << "\n";
        }

      } else if (EtTrack.globalPhi >= phiQuadrants_[3] && EtTrack.globalPhi < phiQuadrants_[4]) {
        temppx = (EtTrack.pt * cosLUT_[phiQuadrants_[4] - 1 - EtTrack.globalPhi]);
        temppy = -(EtTrack.pt * cosLUT_[EtTrack.globalPhi - phiQuadrants_[3]]);

        if (debug_ == 2) {
          edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
              << "Sector: " << EtTrack.Sector << " Quadrant: " << 4 << "\n"
              << "Int Phi: " << EtTrack.globalPhi
              << " Int Cos(Phi): " << cosLUT_[phiQuadrants_[4] - 1 - EtTrack.globalPhi] << " Int Sin(Phi): -"
              << cosLUT_[EtTrack.globalPhi - phiQuadrants_[3]] << "\n"

              << "Float Phi: " << (float)EtTrack.globalPhi / l1tmetemu::kGlobalPhiBins << " Float Cos(Phi): "
              << (float)cosLUT_[phiQuadrants_[4] - 1 - EtTrack.globalPhi] / l1tmetemu::kGlobalPhiBins
              << " Float Sin(Phi): -"
              << (float)cosLUT_[EtTrack.globalPhi - phiQuadrants_[3]] / l1tmetemu::kGlobalPhiBins << "\n";
        }
      } else {
        temppx = 0;
        temppy = 0;
      }

      int link_number = (EtTrack.Sector * 2) + ((EtTrack.EtaSector) ? 0 : 1);
      link_totals[link_number] += 1;
      sumPx[link_number] += temppx;
      sumPy[link_number] += temppy;
      if (debug_ == 4) {
        edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
            << "Sector: " << EtTrack.Sector << " Eta: " << EtTrack.EtaSector << "\n"
            << "Int Track Px: " << temppx << " Int Track Py: " << temppy << "\n"
            << "Float Track Px: " << (float)temppx * l1tmetemu::kStepPt
            << " Float Track Py:" << (float)temppy * l1tmetemu::kStepPt << "\n"
            << "Int Sector Sum Px: " << sumPx[link_number] << " Int Sector Sum Py: " << sumPy[link_number] << "\n"
            << "Float Sector Sum Px: " << (float)sumPx[link_number] * l1tmetemu::kStepPt
            << " Float Sector Sum Py: " << (float)sumPy[link_number] * l1tmetemu::kStepPt << "\n";
      }
    }

  }  // end loop over tracks

  l1tmetemu::Et_t GlobalPx = 0;
  l1tmetemu::Et_t GlobalPy = 0;

  float tempsumPx = 0;
  float tempsumPy = 0;

  // Global Et sum as floats to emulate rounding in HW
  for (unsigned int i = 0; i < l1tmetemu::kNSector * 2; i++) {
    tempsumPx += floor((float)sumPx[i] / (float)l1tmetemu::kGlobalPhiBins);
    tempsumPy += floor((float)sumPy[i] / (float)l1tmetemu::kGlobalPhiBins);
  }

  // Recast rounded temporary sums into Et_t datatype
  GlobalPx = tempsumPx;
  GlobalPy = tempsumPy;

  // Perform cordic sqrt, take x,y and converts to polar coordinate r,phi where
  // r=sqrt(x**2+y**2) and phi = atan(y/x)
  l1tmetemu::EtMiss EtMiss = cordicSqrt.toPolar(GlobalPx, GlobalPy);

  // Recentre phi
  l1tmetemu::METphi_t tempPhi = 0;

  if ((GlobalPx < 0) && (GlobalPy < 0))
    tempPhi = EtMiss.Phi - l1tmetemu::kMETPhiBins / 2;
  else if ((GlobalPx >= 0) && (GlobalPy >= 0))
    tempPhi = (EtMiss.Phi) + l1tmetemu::kMETPhiBins / 2;
  else if ((GlobalPx >= 0) && (GlobalPy < 0))
    tempPhi = EtMiss.Phi - l1tmetemu::kMETPhiBins / 2;
  else if ((GlobalPx < 0) && (GlobalPy >= 0))
    tempPhi = EtMiss.Phi - 3 * l1tmetemu::kMETPhiBins / 2;

  if (debug_ == 6) {

    edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
        << "====Sector Pt====\n";

    for (unsigned int i = 0; i < l1tmetemu::kNSector * 2; i++) {

      edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
        << "Sector " << i << "\n"
        << "Float Px: " << sumPx[i] * l1tmetemu::kStepPt << " | Float Py: " << sumPy[i] * l1tmetemu::kStepPt 
        << " | Integer Px: " << floor((float)sumPx[i] / (float)l1tmetemu::kGlobalPhiBins)
        << " | Integer Py: " << floor((float)sumPy[i] / (float)l1tmetemu::kGlobalPhiBins) 
        << " | Sector Totals: " << sector_totals[(int)(i/2)] << "\n";
    }

    edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
        << "====Global Pt====\n"
        << "Integer Global Px: " << GlobalPx << "| Integer Global Py: " << GlobalPy << "\n"
        << "Float Global Px: " << GlobalPx * l1tmetemu::kStepPt
        << "| Float Global Py: " << GlobalPy * l1tmetemu::kStepPt << "\n";

    edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
        << "====MET===\n"
        << "Integer MET: " << EtMiss.Et << "| Integer MET phi: " << EtMiss.Phi << "\n"
        << "Float MET: " << EtMiss.Et.to_double() 
        << "| Float MET phi: " << (float)tempPhi * l1tmetemu::kStepMETPhi - M_PI << "\n"
        << "# Tracks after Quality Cuts: " << num_quality_tracks << "\n"
        << "# Tracks Associated to Vertex: " << num_assoc_tracks << "\n"
        << "========================================================\n";
  }

  math::XYZTLorentzVector missingEt(-GlobalPx, -GlobalPy, 0, EtMiss.Et);
  EtSum L1EtSum(missingEt, EtSum::EtSumType::kMissingEt, (int)EtMiss.Et.range(), 0, (int)tempPhi, (int)num_assoc_tracks);

  METCollection->push_back(L1EtSum);

  iEvent.put(std::move(METCollection), L1MetCollectionName_);
}  // end producer

void L1TrackerEtMissEmulatorProducer::beginJob() {}

void L1TrackerEtMissEmulatorProducer::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TrackerEtMissEmulatorProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(L1TrackerEtMissEmulatorProducer);
