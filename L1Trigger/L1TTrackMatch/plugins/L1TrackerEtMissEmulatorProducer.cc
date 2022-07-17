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
  std::vector<l1tmetemu::eta_t> EtaRegionsLUT_;  // Various precomputed LUTs
  std::vector<l1tmetemu::z_t> DeltaZLUT_;
  std::vector<l1tmetemu::global_phi_t> phiQuadrants_;
  std::vector<l1tmetemu::global_phi_t> phiShifts_;

  l1tmetemu::z_t minZ0_;
  l1tmetemu::z_t maxZ0_;
  l1tmetemu::eta_t maxEta_;
  TTTrack_TrackWord::chi2rphi_t chi2rphiMax_;
  TTTrack_TrackWord::chi2rz_t chi2rzMax_;
  TTTrack_TrackWord::bendChi2_t bendChi2Max_;
  l1tmetemu::pt_t minPt_;
  l1tmetemu::nstub_t nStubsmin_;

  vector<double> z0Thresholds_;
  vector<double> etaRegions_;

  l1tmetemu::z_t deltaZ0_ = 0;

  int cordicSteps_;
  int debug_;
  bool cordicDebug_ = false;

  bool GTTinput_ = false;

  L1TkEtMissEmuTrackTransform TrackTransform;

  std::string L1MetCollectionName_;

  const edm::EDGetTokenT<L1VertexCollectionType> pvToken_;
  const edm::EDGetTokenT<L1TTTrackCollectionType> trackToken_;
};

// constructor//
L1TrackerEtMissEmulatorProducer::L1TrackerEtMissEmulatorProducer(const edm::ParameterSet& iConfig)
    : pvToken_(consumes<L1VertexCollectionType>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))),
      trackToken_(consumes<L1TTTrackCollectionType>(iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))) {
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

  // Input parameter cuts and convert to correct integer representations
  maxZ0_ = l1tmetemu::digitizeSignedValue<l1tmetemu::z_t>(
      (double)iConfig.getParameter<double>("maxZ0"), l1tmetemu::kInternalVTXWidth, l1tmetemu::kStepZ0);
  minZ0_ = (1 << TTTrack_TrackWord::TrackBitWidths::kZ0Size) - maxZ0_;

  maxEta_ = l1tmetemu::digitizeSignedValue<l1tmetemu::eta_t>(
      (double)iConfig.getParameter<double>("maxEta"), l1tmetemu::kInternalEtaWidth, l1tmetemu::kStepEta);

  chi2rphiMax_ =
      l1tmetemu::getBin((double)iConfig.getParameter<double>("chi2rphidofMax"), TTTrack_TrackWord::chi2RPhiBins);
  chi2rzMax_ = l1tmetemu::getBin((double)iConfig.getParameter<double>("chi2rzdofMax"), TTTrack_TrackWord::chi2RZBins);
  bendChi2Max_ =
      l1tmetemu::getBin((double)iConfig.getParameter<double>("bendChi2Max"), TTTrack_TrackWord::bendChi2Bins);

  minPt_ = l1tmetemu::digitizeSignedValue<l1tmetemu::pt_t>(
      (double)iConfig.getParameter<double>("minPt"), l1tmetemu::kInternalPtWidth, l1tmetemu::kStepPt);

  nStubsmin_ = (l1tmetemu::nstub_t)iConfig.getParameter<int>("nStubsmin");

  z0Thresholds_ = iConfig.getParameter<std::vector<double>>("z0Thresholds");
  etaRegions_ = iConfig.getParameter<std::vector<double>>("etaRegions");

  if (debug_ == 5) {
    cordicDebug_ = true;
  }

  // To have same bin spacing between 0 and pi/2 as between original phi
  // granularity
  int cosLUTbins = floor(l1tmetemu::kMaxCosLUTPhi / l1tmetemu::kStepPhi);

  // Compute LUTs
  cosLUT_ = l1tmetemu::generateCosLUT(cosLUTbins);
  EtaRegionsLUT_ = l1tmetemu::generateEtaRegionLUT(etaRegions_);
  DeltaZLUT_ = l1tmetemu::generateDeltaZLUT(z0Thresholds_);

  produces<std::vector<EtSum>>(L1MetCollectionName_);
}

L1TrackerEtMissEmulatorProducer::~L1TrackerEtMissEmulatorProducer() {}

void L1TrackerEtMissEmulatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<std::vector<l1t::EtSum>> METCollection(new std::vector<l1t::EtSum>(0));

  edm::Handle<L1VertexCollectionType> L1VertexHandle;
  iEvent.getByToken(pvToken_, L1VertexHandle);

  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken_, L1TTTrackHandle);

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

  // Initialize sector sums, need 0 initialization in case a sector has no
  // tracks
  l1tmetemu::Et_t sumPx[l1tmetemu::kNSector * 2] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  l1tmetemu::Et_t sumPy[l1tmetemu::kNSector * 2] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int sector_totals[l1tmetemu::kNSector] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Track counters
  int num_tracks{0};
  int num_assoc_tracks{0};
  int num_quality_tracks{0};

  // Get reference to first vertex in event vertex collection
  L1VertexType& vtx = const_cast<L1VertexType&>(L1VertexHandle->at(0));

  for (const auto& track : *L1TTTrackHandle) {
    num_tracks++;
    L1TTTrackType& track_ref = const_cast<L1TTTrackType&>(track);  // Get Reference to track to pass to TrackTransform

    // Convert to internal track representation
    InternalEtWord EtTrack = TrackTransform.transformTrack<L1TTTrackType, L1VertexType>(track_ref, vtx);

    // Parameter cuts
    if (EtTrack.pt < minPt_)
      continue;

    // Z signed so double bound
    if (EtTrack.z0 & (1 << (l1tmetemu::kInternalVTXWidth - 1))) {
      // if negative
      if (EtTrack.z0 <= maxZ0_)
        continue;
    } else {
      if (EtTrack.z0 > minZ0_)
        continue;
    }
    if (EtTrack.eta > maxEta_)
      continue;

    // Quality Cuts
    if (EtTrack.chi2rphidof >= chi2rphiMax_)
      continue;

    if (EtTrack.chi2rzdof >= chi2rzMax_)
      continue;

    if (EtTrack.bendChi2 >= bendChi2Max_)
      continue;

    if (EtTrack.nstubs < nStubsmin_)
      continue;

    num_quality_tracks++;

    // Temporary int representation to get the difference
    int tempz = l1tmetemu::unpackSignedValue(EtTrack.z0, l1tmetemu::kInternalVTXWidth);
    int temppv = l1tmetemu::unpackSignedValue(EtTrack.pV, l1tmetemu::kInternalVTXWidth);

    l1tmetemu::z_t z_diff = abs(tempz - temppv);

    // Track to vertex association, adaptive z window based on eta region
    for (unsigned int reg = 0; reg < l1tmetemu::kNEtaRegion; reg++) {
      if (EtTrack.eta >= EtaRegionsLUT_[reg] && EtTrack.eta < EtaRegionsLUT_[reg + 1]) {
        deltaZ0_ = DeltaZLUT_[reg];
        break;
      }
    }

    if (z_diff <= deltaZ0_) {
      num_assoc_tracks++;
      if (debug_ == 7) {
        edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
            << "Track to Vertex ID: " << num_tracks << "\n"
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
      if (EtTrack.EtaSector) {
        sumPx[EtTrack.Sector] = sumPx[EtTrack.Sector] + temppx;
        sumPy[EtTrack.Sector] = sumPy[EtTrack.Sector] + temppy;
      } else {
        sumPx[l1tmetemu::kNSector + EtTrack.Sector] = sumPx[l1tmetemu::kNSector + EtTrack.Sector] + temppx;
        sumPy[l1tmetemu::kNSector + EtTrack.Sector] = sumPy[l1tmetemu::kNSector + EtTrack.Sector] + temppy;
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
    std::string flpxarray[l1tmetemu::kNSector * 2];
    std::string flpyarray[l1tmetemu::kNSector * 2];

    std::string intpxarray[l1tmetemu::kNSector * 2];
    std::string intpyarray[l1tmetemu::kNSector * 2];

    std::string totalsarray[l1tmetemu::kNSector * 2];

    for (unsigned int i = 0; i < l1tmetemu::kNSector * 2; i++) {
      flpxarray[i] = to_string(sumPx[i] * l1tmetemu::kStepPt) + "|";
      flpyarray[i] = to_string(sumPy[i] * l1tmetemu::kStepPt) + "|";
      intpxarray[i] = to_string(floor((float)sumPx[i] / (float)l1tmetemu::kGlobalPhiBins)) + "|";
      intpyarray[i] = to_string(floor((float)sumPy[i] / (float)l1tmetemu::kGlobalPhiBins)) + "|";
      totalsarray[i] = to_string(sector_totals[i]) + "|";
    }

    edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
        << "====Sector Pt====\n"
        << "Float Px: " << flpxarray << "Float Py: " << flpyarray << "Integer Px: " << intpyarray
        << "Integer Px: " << intpyarray << "Sector Totals: " << totalsarray

        << "====Global Pt====\n"
        << "Integer Global Px: " << GlobalPx << "| Integer Global Py: " << GlobalPy << "\n"
        << "Float Global Px: " << GlobalPx * l1tmetemu::kStepPt
        << "| Float Global Py: " << GlobalPy * l1tmetemu::kStepPt << "\n";
  }

  if (debug_ == 6) {
    edm::LogVerbatim("L1TrackerEtMissEmulatorProducer")
        << "====MET===\n"
        << "Integer MET: " << EtMiss.Et << "| Integer MET phi: " << EtMiss.Phi << "\n"
        << "Float MET: " << (EtMiss.Et) * l1tmetemu::kStepMET
        << "| Float MET phi: " << (float)tempPhi * l1tmetemu::kStepMETPhi - M_PI << "\n"
        << "# Intial Tracks: " << num_tracks << "\n"
        << "# Tracks after Quality Cuts: " << num_quality_tracks << "\n"
        << "# Tracks Associated to Vertex: " << num_assoc_tracks << "\n"
        << "========================================================\n";
  }

  math::XYZTLorentzVector missingEt(-GlobalPx, -GlobalPy, 0, EtMiss.Et);
  EtSum L1EtSum(missingEt, EtSum::EtSumType::kMissingEt, (int)EtMiss.Et, 0, (int)tempPhi, (int)num_assoc_tracks);

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
