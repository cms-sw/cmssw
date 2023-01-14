// -*- C++ -*-
//
// Package:    FastSimulation/CTPPSFastTrackingProducer
// Class:      CTPPSFastTrackingProducer
//
/**\class CTPPSFastTrackingProducer CTPPSFastTrackingProducer.cc FastSimulation/CTPPSFastTrackingProducer/plugins/CTPPSFastTrackingProducer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Sandro Fonseca De Souza
//         Created:  Thu, 29 Sep 2016 16:13:41 GMT
//
//

#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastRecHit.h"
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastRecHitContainer.h"
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastTrack.h"
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastTrackContainer.h"
#include "FastSimulation/CTPPSFastGeometry/interface/CTPPSToFDetector.h"
#include "FastSimulation/CTPPSFastGeometry/interface/CTPPSTrkDetector.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"
#include "Utilities/PPS/interface/PPSUtilities.h"

#include "TLorentzVector.h"

// hector includes
#include "H_Parameters.h"
#include "H_BeamLine.h"
#include "H_RecRPObject.h"
#include "H_BeamParticle.h"

class CTPPSFastTrackingProducer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSFastTrackingProducer(const edm::ParameterSet&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  typedef std::vector<CTPPSFastRecHit> CTPPSFastRecHitContainer;
  edm::EDGetTokenT<CTPPSFastRecHitContainer> _recHitToken;
  void ReadRecHits(edm::Handle<CTPPSFastRecHitContainer>&);
  void FastReco(int Direction, H_RecRPObject* station);
  void Reconstruction();
  void ReconstructArm(
      H_RecRPObject* pps_station, double x1, double y1, double x2, double y2, double& tx, double& ty, double& eloss);
  void MatchCellId(int cellId, std::vector<int> vrecCellId, std::vector<double> vrecTof, bool& match, double& recTof);
  bool SearchTrack(int,
                   int,
                   int Direction,
                   double& xi,
                   double& t,
                   double& partP,
                   double& pt,
                   double& thx,
                   double& thy,
                   double& x0,
                   double& y0,
                   double& xt,
                   double& yt,
                   double& X1d,
                   double& Y1d,
                   double& X2d,
                   double& Y2d);
  void TrackerStationClear();
  void TrackerStationStarting();
  void ProjectToToF(const double x1, const double y1, const double x2, const double y2, double& xt, double& yt) {
    xt = ((fz_timing - fz_tracker2) * (x2 - x1) / (fz_tracker2 - fz_tracker1)) + x2;
    yt = ((fz_timing - fz_tracker2) * (y2 - y1) / (fz_tracker2 - fz_tracker1)) + y2;
  };
  // Hector objects
  bool SetBeamLine();

  std::unique_ptr<H_BeamLine> m_beamlineCTPPS1;
  std::unique_ptr<H_BeamLine> m_beamlineCTPPS2;
  std::unique_ptr<H_RecRPObject> pps_stationF;
  std::unique_ptr<H_RecRPObject> pps_stationB;

  std::string beam1filename;
  std::string beam2filename;

  // Defaults
  double lengthctpps;
  bool m_verbosity;
  double fBeamEnergy;
  double fBeamMomentum;
  bool fCrossAngleCorr;
  double fCrossingAngleBeam1;
  double fCrossingAngleBeam2;
  ////////////////////////////////////////////////
  std::unique_ptr<CTPPSTrkStation> TrkStation_F;  // auxiliary object with the tracker geometry
  std::unique_ptr<CTPPSTrkStation> TrkStation_B;
  std::unique_ptr<CTPPSTrkDetector> det1F;
  std::unique_ptr<CTPPSTrkDetector> det1B;
  std::unique_ptr<CTPPSTrkDetector> det2F;
  std::unique_ptr<CTPPSTrkDetector> det2B;
  std::unique_ptr<CTPPSToFDetector> detToF_F;
  std::unique_ptr<CTPPSToFDetector> detToF_B;

  std::vector<CTPPSFastTrack> theCTPPSFastTrack;

  CTPPSFastTrack track;

  std::vector<int> recCellId_F, recCellId_B;
  std::vector<double> recTof_F, recTof_B;

  double fz_tracker1, fz_tracker2, fz_timing;
  double fTrackerWidth, fTrackerHeight, fTrackerInsertion, fBeamXRMS_Trk1, fBeamXRMS_Trk2, fTrk1XOffset, fTrk2XOffset;
  std::vector<double> fToFCellWidth;
  double fToFCellHeight, fToFPitchX, fToFPitchY;
  int fToFNCellX, fToFNCellY;
  double fToFInsertion, fBeamXRMS_ToF, fToFXOffset, fTimeSigma, fImpParcut;
};
//////////////////////
// constructors and destructor
//
CTPPSFastTrackingProducer::CTPPSFastTrackingProducer(const edm::ParameterSet& iConfig)
    : m_verbosity(false), fBeamMomentum(0.), fCrossAngleCorr(false), fCrossingAngleBeam1(0.), fCrossingAngleBeam2(0.) {
  //register your products
  produces<edm::CTPPSFastTrackContainer>("CTPPSFastTrack");
  using namespace edm;
  _recHitToken = consumes<CTPPSFastRecHitContainer>(iConfig.getParameter<edm::InputTag>("recHitTag"));
  m_verbosity = iConfig.getParameter<bool>("Verbosity");
  // User definitons

  // Read beam parameters needed for Hector reconstruction
  lengthctpps = iConfig.getParameter<double>("BeamLineLengthCTPPS");
  beam1filename = iConfig.getParameter<string>("Beam1");
  beam2filename = iConfig.getParameter<string>("Beam2");
  fBeamEnergy = iConfig.getParameter<double>("BeamEnergy");  // beam energy in GeV
  fBeamMomentum = sqrt(fBeamEnergy * fBeamEnergy - PPSTools::ProtonMassSQ);
  fCrossingAngleBeam1 = iConfig.getParameter<double>("CrossingAngleBeam1");
  fCrossingAngleBeam2 = iConfig.getParameter<double>("CrossingAngleBeam2");

  if (fCrossingAngleBeam1 != 0 || fCrossingAngleBeam2 != 0)
    fCrossAngleCorr = true;
  //Read detectors positions and parameters

  fz_tracker1 = iConfig.getParameter<double>("Z_Tracker1");
  fz_tracker2 = iConfig.getParameter<double>("Z_Tracker2");
  fz_timing = iConfig.getParameter<double>("Z_Timing");
  //
  fTrackerWidth = iConfig.getParameter<double>("TrackerWidth");
  fTrackerHeight = iConfig.getParameter<double>("TrackerHeight");
  fTrackerInsertion = iConfig.getParameter<double>("TrackerInsertion");
  fBeamXRMS_Trk1 = iConfig.getParameter<double>("BeamXRMS_Trk1");
  fBeamXRMS_Trk2 = iConfig.getParameter<double>("BeamXRMS_Trk2");
  fTrk1XOffset = iConfig.getParameter<double>("Trk1XOffset");
  fTrk2XOffset = iConfig.getParameter<double>("Trk2XOffset");
  fToFCellWidth = iConfig.getUntrackedParameter<std::vector<double> >("ToFCellWidth");
  fToFCellHeight = iConfig.getParameter<double>("ToFCellHeight");
  fToFPitchX = iConfig.getParameter<double>("ToFPitchX");
  fToFPitchY = iConfig.getParameter<double>("ToFPitchY");
  fToFNCellX = iConfig.getParameter<int>("ToFNCellX");
  fToFNCellY = iConfig.getParameter<int>("ToFNCellY");
  fToFInsertion = iConfig.getParameter<double>("ToFInsertion");
  fBeamXRMS_ToF = iConfig.getParameter<double>("BeamXRMS_ToF");
  fToFXOffset = iConfig.getParameter<double>("ToFXOffset");
  fTimeSigma = iConfig.getParameter<double>("TimeSigma");
  fImpParcut = iConfig.getParameter<double>("ImpParcut");

  if (!SetBeamLine()) {
    if (m_verbosity)
      LogDebug("CTPPSFastTrackingProducer") << "CTPPSFastTrackingProducer: WARNING: lengthctpps=  " << lengthctpps;
    return;
  }

  // Create a particle to get the beam energy from the beam file
  // Take care: the z inside the station is in meters
  //
  //Tracker Detector Description
  det1F = std::make_unique<CTPPSTrkDetector>(
      fTrackerWidth, fTrackerHeight, fTrackerInsertion * fBeamXRMS_Trk1 + fTrk1XOffset);
  det2F = std::make_unique<CTPPSTrkDetector>(
      fTrackerWidth, fTrackerHeight, fTrackerInsertion * fBeamXRMS_Trk2 + fTrk2XOffset);
  det1B = std::make_unique<CTPPSTrkDetector>(
      fTrackerWidth, fTrackerHeight, fTrackerInsertion * fBeamXRMS_Trk1 + fTrk1XOffset);
  det2B = std::make_unique<CTPPSTrkDetector>(
      fTrackerWidth, fTrackerHeight, fTrackerInsertion * fBeamXRMS_Trk2 + fTrk2XOffset);

  //Timing Detector Description
  std::vector<double> vToFCellWidth;
  vToFCellWidth.reserve(8);
  for (int i = 0; i < 8; i++) {
    vToFCellWidth.push_back(fToFCellWidth[i]);
  }
  double pos_tof = fToFInsertion * fBeamXRMS_ToF + fToFXOffset;
  detToF_F = std::make_unique<CTPPSToFDetector>(
      fToFNCellX, fToFNCellY, vToFCellWidth, fToFCellHeight, fToFPitchX, fToFPitchY, pos_tof, fTimeSigma);
  detToF_B = std::make_unique<CTPPSToFDetector>(
      fToFNCellX, fToFNCellY, vToFCellWidth, fToFCellHeight, fToFPitchX, fToFPitchY, pos_tof, fTimeSigma);
  //
}

// ------------ method called to produce the data  ------------
void CTPPSFastTrackingProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  TrackerStationStarting();
  Handle<CTPPSFastRecHitContainer> recHits;
  iEvent.getByToken(_recHitToken, recHits);
  recCellId_F.clear();
  recCellId_B.clear();
  recTof_F.clear();
  recTof_B.clear();
  ReadRecHits(recHits);
  Reconstruction();
  TrackerStationClear();

  std::unique_ptr<CTPPSFastTrackContainer> output_tracks(new CTPPSFastTrackContainer);
  for (std::vector<CTPPSFastTrack>::const_iterator i = theCTPPSFastTrack.begin(); i != theCTPPSFastTrack.end(); i++) {
    output_tracks->push_back(*i);
  }

  iEvent.put(std::move(output_tracks), "CTPPSFastTrack");
}  //end

/////////////////////////
void CTPPSFastTrackingProducer::TrackerStationClear() {
  TrkStation_F->first.clear();
  TrkStation_F->second.clear();
  TrkStation_B->first.clear();
  TrkStation_B->second.clear();
}
/////////////////////////
void CTPPSFastTrackingProducer::TrackerStationStarting() {
  det1F->clear();
  det1B->clear();
  det2F->clear();
  det2B->clear();
  detToF_F->clear();
  detToF_B->clear();
}

////////////////////////////
void CTPPSFastTrackingProducer::ReadRecHits(edm::Handle<CTPPSFastRecHitContainer>& recHits) {
  // DetId codification for PSimHit taken from CTPPSPixel- It will be replaced by CTPPSDetId
  // 2014314496 -> Tracker1 zPositive
  // 2014838784 -> Tracker2 zPositive
  // 2046820352 -> Timing   zPositive
  // 2031091712 -> Tracker1 zNegative
  // 2031616000 -> Tracker2 zNegative
  // 2063597568 -> Timing   zNegative

  for (unsigned int irecHits = 0; irecHits < recHits->size(); ++irecHits) {
    const CTPPSFastRecHit* recHitDet = &(*recHits)[irecHits];
    unsigned int detlayerId = recHitDet->detUnitId();
    double x = recHitDet->entryPoint().x();
    double y = recHitDet->entryPoint().y();
    double z = recHitDet->entryPoint().z();
    float tof = recHitDet->tof();
    if (detlayerId == 2014314496)
      det1F->AddHit(detlayerId, x, y, z);
    else if (detlayerId == 2014838784)
      det2F->AddHit(detlayerId, x, y, z);
    else if (detlayerId == 2031091712)
      det1B->AddHit(detlayerId, x, y, z);
    else if (detlayerId == 2031616000)
      det2B->AddHit(detlayerId, x, y, z);
    else if (detlayerId == 2046820352) {
      detToF_F->AddHit(x, y, tof);
      recCellId_F.push_back(detToF_F->findCellId(x, y));
      recTof_F.push_back(tof);
    } else if (detlayerId == 2063597568) {
      detToF_B->AddHit(x, y, tof);
      recCellId_B.push_back(detToF_B->findCellId(x, y));
      recTof_B.push_back(tof);
    }

  }  //LOOP TRK
  //creating Stations
  TrkStation_F = std::make_unique<CTPPSTrkStation>(*det1F, *det2F);
  TrkStation_B = std::make_unique<CTPPSTrkStation>(*det1B, *det2B);
}  // end function

void CTPPSFastTrackingProducer::Reconstruction() {
  theCTPPSFastTrack.clear();
  int Direction;
  Direction = 1;  //cms positive Z / forward
  FastReco(Direction, &*pps_stationF);
  Direction = -1;  //cms negative Z / backward
  FastReco(Direction, &*pps_stationB);
}  //end Reconstruction

bool CTPPSFastTrackingProducer::SearchTrack(int i,
                                            int j,
                                            int Direction,
                                            double& xi,
                                            double& t,
                                            double& partP,
                                            double& pt,
                                            double& thx,
                                            double& thy,
                                            double& x0,
                                            double& y0,
                                            double& xt,
                                            double& yt,
                                            double& X1d,
                                            double& Y1d,
                                            double& X2d,
                                            double& Y2d) {
  // Given 1 hit in Tracker1 and 1 hit in Tracker2 try to make a track with Hector
  xi = 0;
  t = 0;
  partP = 0;
  pt = 0;
  x0 = 0.;
  y0 = 0.;
  xt = 0.;
  yt = 0.;
  X1d = 0.;
  Y1d = 0.;
  X2d = 0.;
  Y2d = 0.;
  CTPPSTrkDetector* det1 = nullptr;
  CTPPSTrkDetector* det2 = nullptr;
  H_RecRPObject* station = nullptr;
  // Separate in forward and backward stations according to direction
  if (Direction > 0) {
    det1 = &(TrkStation_F->first);
    det2 = &(TrkStation_F->second);
    station = &*pps_stationF;
  } else {
    det1 = &(TrkStation_B->first);
    det2 = &(TrkStation_B->second);
    station = &*pps_stationB;
  }
  if (det1->ppsNHits_ <= i || det2->ppsNHits_ <= j)
    return false;
  //
  double x1 = det1->ppsX_.at(i);
  double y1 = det1->ppsY_.at(i);
  double x2 = det2->ppsX_.at(j);
  double y2 = det2->ppsY_.at(j);
  double eloss;

  //thx and thy are returned in microrad
  ReconstructArm(
      station, Direction * x1, y1, Direction * x2, y2, thx, thy, eloss);  // Pass the hits in the LHC ref. frame
  thx *= -Direction;                                                      //  invert to the CMS ref frame

  // Protect for unphysical results
  if (edm::isNotFinite(eloss) || edm::isNotFinite(thx) || edm::isNotFinite(thy))
    return false;
  //

  if (m_verbosity)
    LogDebug("CTPPSFastTrackingProducer::SearchTrack:") << "thx " << thx << " thy " << thy << " eloss " << eloss;

  // Get the start point of the reconstructed track near the origin made by Hector in the CMS ref. frame
  x0 = -Direction * station->getX0() * um_to_cm;
  y0 = station->getY0() * um_to_cm;
  double ImpPar = sqrt(x0 * x0 + y0 * y0);
  if (ImpPar > fImpParcut)
    return false;
  if (eloss < 0. || eloss > fBeamEnergy)
    return false;
  //
  // Calculate the reconstructed track parameters
  double theta = sqrt(thx * thx + thy * thy) * urad;
  xi = eloss / fBeamEnergy;
  double energy = fBeamEnergy * (1. - xi);
  partP = sqrt(energy * energy - PPSTools::ProtonMassSQ);
  t = -2. * (PPSTools::ProtonMassSQ - fBeamEnergy * energy + fBeamMomentum * partP * cos(theta));
  pt = partP * theta;
  if (xi < 0. || xi > 1. || t < 0. || t > 10. || pt <= 0.) {
    xi = 0.;
    t = 0.;
    partP = 0.;
    pt = 0.;
    x0 = 0.;
    y0 = 0.;
    return false;  // unphysical values
  }
  //Try to include the timing detector in the track
  ProjectToToF(x1, y1, x2, y2, xt, yt);  // the projections is done in the CMS ref frame
  X1d = x1;
  Y1d = y1;
  X2d = x2;
  Y2d = y2;
  return true;
}  //end  SearchTrack

void CTPPSFastTrackingProducer::ReconstructArm(
    H_RecRPObject* pps_station, double x1, double y1, double x2, double y2, double& tx, double& ty, double& eloss) {
  tx = 0.;
  ty = 0.;
  eloss = 0.;
  if (!pps_station)
    return;
  x1 *= mm_to_um;
  x2 *= mm_to_um;
  y1 *= mm_to_um;
  y2 *= mm_to_um;
  pps_station->setPositions(x1, y1, x2, y2);
  double energy = pps_station->getE(AM);  // dummy call needed to calculate some Hector internal parameter
  if (edm::isNotFinite(energy))
    return;
  tx = pps_station->getTXIP();  // change orientation to CMS
  ty = pps_station->getTYIP();
  eloss = pps_station->getE();
}

void CTPPSFastTrackingProducer::MatchCellId(
    int cellId, std::vector<int> vrecCellId, std::vector<double> vrecTof, bool& match, double& recTof) {
  for (unsigned int i = 0; i < vrecCellId.size(); i++) {
    if (cellId == vrecCellId.at(i)) {
      match = true;
      recTof = vrecTof.at(i);
      continue;
    }
  }
}

void CTPPSFastTrackingProducer::FastReco(int Direction, H_RecRPObject* station) {
  double theta = 0.;
  double xi, t, partP, pt, phi, x0, y0, thx, thy, xt, yt, X1d, Y1d, X2d, Y2d;
  CTPPSTrkDetector* Trk1 = nullptr;
  CTPPSTrkDetector* Trk2 = nullptr;
  double pos_tof = fToFInsertion * fBeamXRMS_ToF + fToFXOffset;
  int cellId = 0;
  std::vector<double> vToFCellWidth;
  vToFCellWidth.reserve(8);
  for (int i = 0; i < 8; i++) {
    vToFCellWidth.push_back(fToFCellWidth[i]);
  }
  CTPPSToFDetector* ToF = new CTPPSToFDetector(
      fToFNCellX, fToFNCellY, vToFCellWidth, fToFCellHeight, fToFPitchX, fToFPitchY, pos_tof, fTimeSigma);
  if (Direction > 0) {
    Trk1 = &(TrkStation_F->first);
    Trk2 = &(TrkStation_F->second);
  } else {
    Trk1 = &(TrkStation_B->first);
    Trk2 = &(TrkStation_B->second);
  }
  // Make a track from EVERY pair of hits combining Tracker1 and Tracker2.
  // The tracks may not be independent as 1 hit may belong to more than 1 track.
  for (int i = 0; i < (int)Trk1->ppsNHits_; i++) {
    for (int j = 0; j < (int)Trk2->ppsNHits_; j++) {
      if (SearchTrack(i, j, Direction, xi, t, partP, pt, thx, thy, x0, y0, xt, yt, X1d, Y1d, X2d, Y2d)) {
        // Check if the hitted timing cell matches the reconstructed track
        cellId = ToF->findCellId(xt, yt);
        double recTof = 0.;
        bool matchCellId = false;
        if (Direction > 0) {
          theta = sqrt(thx * thx + thy * thy) * urad;
          MatchCellId(cellId, recCellId_F, recTof_F, matchCellId, recTof);
        } else if (Direction < 0) {
          theta = TMath::Pi() - sqrt(thx * thx + thy * thy) * urad;
          MatchCellId(cellId, recCellId_B, recTof_B, matchCellId, recTof);
        }
        phi = atan2(thy, thx);  // at this point, thx is already in the cms ref. frame

        double px = partP * sin(theta) * cos(phi);
        double py = partP * sin(theta) * sin(phi);
        double pz = partP * cos(theta);
        double e = sqrt(partP * partP + PPSTools::ProtonMassSQ);
        TLorentzVector p(px, py, pz, e);
        // Invert the Lorentz boost made to take into account the crossing angle during simulation
        if (fCrossAngleCorr) {
          PPSTools::LorentzBoost(p, "MC", {fCrossingAngleBeam1, fCrossingAngleBeam2, fBeamMomentum, fBeamEnergy});
        }
        //Getting the Xi and t (squared four momentum transferred) of the reconstructed track
        PPSTools::Get_t_and_xi(const_cast<TLorentzVector*>(&p), t, xi, {fBeamMomentum, fBeamEnergy});
        double pxx = p.Px();
        double pyy = p.Py();
        double pzz = p.Pz();
        math::XYZVector momentum(pxx, pyy, pzz);
        math::XYZPoint vertex(x0, y0, 0);

        track.setp(momentum);
        track.setvertex(vertex);
        track.sett(t);
        track.setxi(xi);
        track.setx1(X1d);
        track.sety1(Y1d);
        track.setx2(X2d);
        track.sety2(Y2d);
        if (matchCellId) {
          track.setcellid(cellId);
          track.settof(recTof);
        } else {
          track.setcellid(0);
          track.settof(0.);
        }
        theCTPPSFastTrack.push_back(track);
      }
    }
  }
}  //end FastReco

bool CTPPSFastTrackingProducer::SetBeamLine() {
  edm::FileInPath b1(beam1filename.c_str());
  edm::FileInPath b2(beam2filename.c_str());
  if (lengthctpps <= 0)
    return false;
  m_beamlineCTPPS1 = std::make_unique<H_BeamLine>(-1, lengthctpps + 0.1);  // (direction, length)
  m_beamlineCTPPS1->fill(b2.fullPath(), 1, "IP5");
  m_beamlineCTPPS2 = std::make_unique<H_BeamLine>(1, lengthctpps + 0.1);  //
  m_beamlineCTPPS2->fill(b1.fullPath(), 1, "IP5");
  m_beamlineCTPPS1->offsetElements(120, 0.097);
  m_beamlineCTPPS2->offsetElements(120, -0.097);
  pps_stationF = std::make_unique<H_RecRPObject>(fz_tracker1, fz_tracker2, *m_beamlineCTPPS1);
  pps_stationB = std::make_unique<H_RecRPObject>(fz_tracker1, fz_tracker2, *m_beamlineCTPPS2);
  return true;
}
//define this as a plug-in
DEFINE_FWK_MODULE(CTPPSFastTrackingProducer);
