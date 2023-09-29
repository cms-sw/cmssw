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
#include <iomanip>
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

using namespace l1t;

class L1TrackerEtMissEmulatorProducer : public edm::stream::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;
  typedef edm::RefVector<L1TTTrackCollectionType> L1TTTrackRefCollectionType;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  explicit L1TrackerEtMissEmulatorProducer(const edm::ParameterSet&);
  ~L1TrackerEtMissEmulatorProducer() override;

private:
  virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();

  // ----------member data ---------------------------

  std::vector<l1tmetemu::cos_lut_fixed_t> cosLUT_;  // Cos LUT array
  std::vector<l1tmetemu::global_phi_t> phiQuadrants_;
  std::vector<l1tmetemu::global_phi_t> phiShifts_;

  int cordicSteps_;
  int debug_;
  bool cordicDebug_ = false;

  std::string L1MetCollectionName_;

  const edm::EDGetTokenT<L1TTTrackRefCollectionType> trackToken_;
  const edm::EDGetTokenT<L1TTTrackRefCollectionType> vtxAssocTrackToken_;
};

// constructor//
L1TrackerEtMissEmulatorProducer::L1TrackerEtMissEmulatorProducer(const edm::ParameterSet& iConfig)
    : trackToken_(consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      vtxAssocTrackToken_(
          consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<edm::InputTag>("L1TrackAssociatedInputTag"))) {
  phiQuadrants_ = l1tmetemu::generatePhiSliceLUT(l1tmetemu::kNQuadrants);
  phiShifts_ = l1tmetemu::generatePhiSliceLUT(l1tmetemu::kNSector);

  // Get Emulator config parameters
  cordicSteps_ = (int)iConfig.getParameter<int>("nCordicSteps");
  debug_ = (int)iConfig.getParameter<int>("debug");
  // Name of output ED Product
  L1MetCollectionName_ = (std::string)iConfig.getParameter<std::string>("L1MetCollectionName");

  if (debug_ == 5) {
    cordicDebug_ = true;
  }

  // Compute LUTs
  cosLUT_ = l1tmetemu::generateCosLUT();

  // Print LUTs
  if (debug_ == 1) {
    l1tmetemu::printLUT(phiQuadrants_, "L1TrackerEtMissEmulatorProducer", "phiQuadrants_");
    l1tmetemu::printLUT(phiShifts_, "L1TrackerEtMissEmulatorProducer", "phiShifts_");
    l1tmetemu::printLUT(cosLUT_, "L1TrackerEtMissEmulatorProducer", "cosLUT_");
  }

  produces<std::vector<EtSum>>(L1MetCollectionName_);
}

L1TrackerEtMissEmulatorProducer::~L1TrackerEtMissEmulatorProducer() {}

void L1TrackerEtMissEmulatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<std::vector<l1t::EtSum>> METCollection(new std::vector<l1t::EtSum>(0));

  edm::Handle<L1TTTrackRefCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken_, L1TTTrackHandle);

  edm::Handle<L1TTTrackRefCollectionType> L1TTTrackAssociatedHandle;
  iEvent.getByToken(vtxAssocTrackToken_, L1TTTrackAssociatedHandle);

  // Initialize cordic class
  Cordic cordicSqrt(cordicSteps_, cordicDebug_);

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
  int num_assoc_tracks{0};

  for (const auto& track : *L1TTTrackHandle) {
    if (std::find(L1TTTrackAssociatedHandle->begin(), L1TTTrackAssociatedHandle->end(), track) !=
        L1TTTrackAssociatedHandle->end()) {
      bool EtaSector = (track->getTanlWord() & (1 << (TTTrack_TrackWord::TrackBitWidths::kTanlSize - 1)));

      ap_uint<TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1> ptEmulationBits = track->getTrackWord()(
          TTTrack_TrackWord::TrackBitLocations::kRinvMSB - 1, TTTrack_TrackWord::TrackBitLocations::kRinvLSB);
      ap_ufixed<TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1, l1tmetemu::kPtMagSize> ptEmulation;
      ptEmulation.V = ptEmulationBits.range();

      l1tmetemu::global_phi_t globalPhi =
          l1tmetemu::localToGlobalPhi(track->getPhiWord(), phiShifts_[track->phiSector()]);

      num_assoc_tracks++;
      if (debug_ == 7) {
        edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
            << "Track to Vertex ID: " << num_assoc_tracks << "\n"
            << "Phi Sector: " << track->phiSector() << " pT: " << track->getRinvWord()
            << " Phi: " << track->getPhiWord() << " TanL: " << track->getTanlWord() << " Z0: " << track->getZ0Word()
            << " Chi2rphi: " << track->getChi2RPhiWord() << " Chi2rz: " << track->getChi2RZWord()
            << " bendChi2: " << track->getBendChi2Word() << " Emu pT " << ptEmulation.to_double() << "\n"
            << "--------------------------------------------------------------\n";
      }

      if (debug_ == 2) {
        edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
            << "========================Phi debug=================================\n"
            << "Emu pT: " << ptEmulation.to_double() << " float pT: " << track->momentum().perp() << "\n"
            << "Int Phi: " << globalPhi << " Float Phi: " << track->phi() << " Float Cos(Phi): " << cos(track->phi())
            << "  Float Sin(Phi): " << sin(track->phi())
            << " Float Px: " << track->momentum().perp() * cos(track->phi())
            << " Float Py: " << track->momentum().perp() * sin(track->phi()) << "\n";
      }

      l1tmetemu::Et_t temppx = 0;
      l1tmetemu::Et_t temppy = 0;

      // Split tracks in phi quadrants and access cosLUT_, backwards iteration
      // through cosLUT_ gives sin Sum sector Et -ve when cos or sin phi are -ve
      sector_totals[track->phiSector()] += 1;
      if (globalPhi >= phiQuadrants_[0] && globalPhi < phiQuadrants_[1]) {
        temppx = ((l1tmetemu::Et_t)ptEmulation * cosLUT_[(globalPhi) >> l1tmetemu::kCosLUTShift]);
        temppy =
            ((l1tmetemu::Et_t)ptEmulation * cosLUT_[(phiQuadrants_[1] - 1 - globalPhi) >> l1tmetemu::kCosLUTShift]);

        if (debug_ == 2) {
          edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
              << "Sector: " << track->phiSector() << " Quadrant: " << 1 << "\n"
              << "Emu Phi: " << globalPhi << " Emu Cos(Phi): " << cosLUT_[(globalPhi) >> l1tmetemu::kCosLUTShift]
              << " Emu Sin(Phi): " << cosLUT_[(phiQuadrants_[1] - 1 - globalPhi) >> l1tmetemu::kCosLUTShift] << "\n";
        }
      } else if (globalPhi >= phiQuadrants_[1] && globalPhi < phiQuadrants_[2]) {
        temppx =
            -((l1tmetemu::Et_t)ptEmulation * cosLUT_[(phiQuadrants_[2] - 1 - globalPhi) >> l1tmetemu::kCosLUTShift]);
        temppy = ((l1tmetemu::Et_t)ptEmulation * cosLUT_[(globalPhi - phiQuadrants_[1]) >> l1tmetemu::kCosLUTShift]);

        if (debug_ == 2) {
          edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
              << "Sector: " << track->phiSector() << " Quadrant: " << 2 << "\n"
              << "Emu Phi: " << globalPhi << " Emu Cos(Phi): -"
              << cosLUT_[(phiQuadrants_[2] - 1 - globalPhi) >> l1tmetemu::kCosLUTShift]
              << " Emu Sin(Phi): " << cosLUT_[(globalPhi - phiQuadrants_[1]) >> l1tmetemu::kCosLUTShift] << "\n";
        }
      } else if (globalPhi >= phiQuadrants_[2] && globalPhi < phiQuadrants_[3]) {
        temppx = -((l1tmetemu::Et_t)ptEmulation * cosLUT_[(globalPhi - phiQuadrants_[2]) >> l1tmetemu::kCosLUTShift]);
        temppy =
            -((l1tmetemu::Et_t)ptEmulation * cosLUT_[(phiQuadrants_[3] - 1 - globalPhi) >> l1tmetemu::kCosLUTShift]);

        if (debug_ == 2) {
          edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
              << "Sector: " << track->phiSector() << " Quadrant: " << 3 << "\n"
              << "Emu Phi: " << globalPhi << " Emu Cos(Phi): -"
              << cosLUT_[(globalPhi - phiQuadrants_[2]) >> l1tmetemu::kCosLUTShift] << " Emu Sin(Phi): -"
              << cosLUT_[(phiQuadrants_[3] - 1 - globalPhi) >> l1tmetemu::kCosLUTShift] << "\n";
        }

      } else if (globalPhi >= phiQuadrants_[3] && globalPhi < phiQuadrants_[4]) {
        temppx =
            ((l1tmetemu::Et_t)ptEmulation * cosLUT_[(phiQuadrants_[4] - 1 - globalPhi) >> l1tmetemu::kCosLUTShift]);
        temppy = -((l1tmetemu::Et_t)ptEmulation * cosLUT_[(globalPhi - phiQuadrants_[3]) >> l1tmetemu::kCosLUTShift]);

        if (debug_ == 2) {
          edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
              << "Sector: " << track->phiSector() << " Quadrant: " << 4 << "\n"
              << " Emu Phi: " << globalPhi
              << " Emu Cos(Phi): " << cosLUT_[(phiQuadrants_[4] - 1 - globalPhi) >> l1tmetemu::kCosLUTShift]
              << " Emu Sin(Phi): -" << cosLUT_[(globalPhi - phiQuadrants_[3]) >> l1tmetemu::kCosLUTShift] << "\n";
        }
      }

      int link_number = (track->phiSector() * 2) + ((EtaSector) ? 0 : 1);
      link_totals[link_number] += 1;
      sumPx[link_number] += temppx;
      sumPy[link_number] += temppy;

      if (debug_ == 4) {
        edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
            << std::setprecision(8) << "Sector: " << track->phiSector() << " Eta sector: " << EtaSector << "\n"
            << "Track Ref Pt: " << track->momentum().perp() << " Track Ref Px: " << track->momentum().x()
            << " Track Ref Py: " << track->momentum().y() << "\n"
            << "Track Pt: " << ptEmulation << " Track phi: " << globalPhi << " Track Px: " << temppx
            << " Track Py: " << temppy << "\n"
            << "Sector Sum Px: " << sumPx[link_number] << " Sector Sum Py: " << sumPy[link_number] << "\n";
      }
    }
  }  // end loop over tracks

  l1tmetemu::Et_t GlobalPx = 0;
  l1tmetemu::Et_t GlobalPy = 0;

  // Global Et sum as floats to emulate rounding in HW
  for (unsigned int i = 0; i < l1tmetemu::kNSector * 2; i++) {
    GlobalPx += sumPx[i];
    GlobalPy += sumPy[i];
  }

  // Perform cordic sqrt, take x,y and converts to polar coordinate r,phi where
  // r=sqrt(x**2+y**2) and phi = atan(y/x)
  l1tmetemu::EtMiss EtMiss = cordicSqrt.toPolar(-GlobalPx, -GlobalPy);

  if (debug_ == 4 || debug_ == 6) {
    edm::LogVerbatim("L1TrackerEtMissEmulatorProducer") << "====Sector Pt====\n";

    for (unsigned int i = 0; i < l1tmetemu::kNSector * 2; i++) {
      edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
          << "Sector " << i << "\n"
          << "Px: " << sumPx[i] << " | Py: " << sumPy[i] << " | Link Totals: " << link_totals[i]
          << " | Sector Totals: " << sector_totals[(int)(i / 2)] << "\n";
    }

    edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
        << "====Global Pt====\n"
        << "Global Px: " << GlobalPx << "| Global Py: " << GlobalPy << "\n";

    edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
        << "====MET===\n"
        << "MET word Et: " << EtMiss.Et.range() * l1tmetemu::kStepMETwordEt << "| MET word phi: " << EtMiss.Phi << "\n"
        << "MET: " << EtMiss.Et.to_double() << "| MET phi: " << EtMiss.Phi.to_double() * l1tmetemu::kStepMETwordPhi
        << "\n"
        << "Word MET: " << EtMiss.Et.to_string(2) << " | Word MET phi: " << EtMiss.Phi.to_string(2) << "\n"
        << "# Tracks Associated to Vertex: " << num_assoc_tracks << "\n"
        << "========================================================\n";
  }

  math::XYZTLorentzVector missingEt(-GlobalPx, -GlobalPy, 0, EtMiss.Et);
  EtSum L1EtSum(missingEt, EtSum::EtSumType::kMissingEt, EtMiss.Et.range(), 0, EtMiss.Phi, num_assoc_tracks);

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
