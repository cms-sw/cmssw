/**
 * \file MillePedeMonitor.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.25 $
 *  $Date: 2013/01/07 20:21:32 $
 *  (last update by $Author: wmtan $)
 */

#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeMonitor.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableBeamSpot.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignmentParametrization/interface/FrameToFrameDerivative.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;

#include <TProfile2D.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TMath.h>

#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"

typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;

//__________________________________________________________________
MillePedeMonitor::MillePedeMonitor(const TrackerTopology* tTopo, const char *rootFileName)
  : myRootDir(0), myDeleteDir(false), trackerTopology(tTopo)
{
  myRootDir = TFile::Open(rootFileName, "recreate");
  myDeleteDir = true;

  this->init(myRootDir);
}

//__________________________________________________________________
MillePedeMonitor::MillePedeMonitor(TDirectory *rootDir, const TrackerTopology* tTopo) 
  : myRootDir(0), myDeleteDir(false), trackerTopology(tTopo)
{
  //  cout << "MillePedeMonitor using input TDirectory" << endl;

  myRootDir = rootDir;
  myDeleteDir = false;

  this->init(myRootDir);
}

//__________________________________________________________________
MillePedeMonitor::~MillePedeMonitor()
{

  myRootDir->Write();
  if (myDeleteDir) delete myRootDir; //hists are deleted with their directory
}

//__________________________________________________________________
bool MillePedeMonitor::init(TDirectory *directory)
{
  if (!directory) return false;
  TDirectory *oldDir = gDirectory;

  const int kNumBins = 20;
  double binsPt[kNumBins+1] = {0.}; // fully initialised with 0.
  if (!this->equidistLogBins(binsPt, kNumBins, 0.8, 100.)) {
//     cerr << "MillePedeMonitor::init: problem with log bins" << endl;
  }
  const int nHits = 35;

  myTrackHists1D.push_back(new TH1F("ptTrackLogBins",  "p_{t}(track);p_{t} [GeV]",
				    kNumBins, binsPt));

  myTrackHists1D.push_back(new TH1F("ptTrack",  "p_{t}(track);p_{t} [GeV]",
				    kNumBins, binsPt[0], binsPt[kNumBins]));
  myTrackHists1D.push_back(new TH1F("pTrack",  "p(track);p [GeV]",
				    kNumBins, binsPt[0], 1.3*binsPt[kNumBins]));
  myTrackHists1D.push_back(new TH1F("etaTrack", "#eta(track);#eta", 26, -2.6, 2.6));
  myTrackHists1D.push_back(new TH1F("thetaTrack", "#theta(track);#theta", 100, 0., TMath::Pi()));
  myTrackHists1D.push_back(new TH1F("phiTrack", "#phi(track);#phi", 15, -TMath::Pi(), TMath::Pi()));

  myTrackHists1D.push_back(new TH1F("nHitTrack", "N_{hit}(track);N_{hit}", 40, 5., 45.));
  myTrackHists1D.push_back(new TH1F("nHitBPIXTrack", "N_{hit, BPIX}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitFPIXTrack", "N_{hit, FPIX}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitFPIXplusTrack", "N_{hit, BPIXplus}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitFPIXminusTrack", "N_{hit, BPIXminus}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitPIXELTrack", "N_{hit, PIXEL}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitTIBTrack", "N_{hit, TIB}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitTOBTrack", "N_{hit, TOB}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitTIDplusTrack", "N_{hit, TIDplus}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitTIDminusTrack", "N_{hit, TIDminus}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitTIDTrack", "N_{hit, TID}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitTECplusTrack", "N_{hit, TECplus}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitTECminusTrack", "N_{hit, TECminus}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitTECTrack", "N_{hit, TEC}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitENDCAPplusTrack", "N_{hit, ENDCAPplus}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitENDCAPminusTrack", "N_{hit, ENDCAPminus}(track);N_{hit}", 45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitENDCAPTrack", "N_{hit, ENDCAP}(track);N_{hit}", 45, 0., 45.));
  myTrackHists2D.push_back(new TH2F("nHitENDCAPTrackMinusVsPlus", "N_{hit, ENDCAP}(track);N_{hit, plus};N_{hit, minus}",
				    45, 0., 45.,45, 0., 45.));
  myTrackHists1D.push_back(new TH1F("nHitInvalidTrack", "N_{hit, invalid}(track)", 5, 0., 5.));
  myTrackHists1D.push_back(new TH1F("r1Track", "r(1st hit);r [cm]", 20, 0., 20.));
  myTrackHists1D.push_back(new TH1F("phi1Track", "#phi(1st hit);#phi",
				    30, -TMath::Pi(), TMath::Pi()));
  myTrackHists1D.push_back(new TH1F("y1Track", "y(1st hit);y [cm]", 40, -120., +120.));
  myTrackHists1D.push_back(new TH1F("z1Track", "z(1st hit);z [cm]", 20, -50, +50));
  myTrackHists1D.push_back(new TH1F("r1TrackSigned", "r(1st hit);r_{#pm} [cm]",
				    40, -120., 120.));
  myTrackHists1D.push_back(new TH1F("z1TrackFull", "z(1st hit);z [cm]", 20, -300., +300.));
  myTrackHists1D.push_back(new TH1F("rLastTrack", "r(last hit);r [cm]", 20, 20., 120.));
  myTrackHists1D.push_back(new TH1F("phiLastTrack", "#phi(last hit);#phi",
				    30, -TMath::Pi(), TMath::Pi()));
  myTrackHists1D.push_back(new TH1F("yLastTrack", "y(last hit);y [cm]", 40, -120., +120.));
  myTrackHists1D.push_back(new TH1F("zLastTrack", "z(last hit);z [cm]", 30, -300., +300.));
  myTrackHists1D.push_back(new TH1F("chi2PerNdf", "#chi^{2}/ndf;#chi^{2}/ndf", 500, 0., 50.));
  myTrackHists1D.push_back(new TH1F("impParZ", "impact parameter in z", 20, -20., 20.));
  myTrackHists1D.push_back(new TH1F("impParErrZ", "error of impact parameter in z",
				    40, 0., 0.06));  
  myTrackHists1D.push_back(new TH1F("impParRphi", "impact parameter in r#phi", 51, -0.05, .05));
  myTrackHists1D.push_back(new TH1F("impParErrRphi", "error of impact parameter in r#phi",
				    50, 0., 0.01));

  myTrackHists2D.push_back(new TH2F("rz1TrackFull", "rz(1st hit);z [cm]; r_{#pm} [cm]",
				    40, -282., +282., 40, -115., +115.));
  myTrackHists2D.push_back(new TH2F("xy1Track", "xy(1st hit);x [cm]; y [cm]",
				    40, -115., +115., 40, -115., +115.));
  
  TDirectory *dirTracks = directory->mkdir("trackHists", "input tracks");
  this->addToDirectory(myTrackHists1D, dirTracks);
  this->addToDirectory(myTrackHists2D, dirTracks);

// used track 
  myUsedTrackHists1D = this->cloneHists(myTrackHists1D, "used", " (used tracks)");
  myUsedTrackHists2D = this->cloneHists(myTrackHists2D, "used", " (used tracks)");
  // must be after clone: index in vector!
  myUsedTrackHists1D.push_back(new TH1F("usedHitsX", "n(x-hits) transferred to pede;n(x-hits)", nHits, 0., nHits));
  myUsedTrackHists1D.push_back(new TH1F("usedHitsY", "n(y-hits) transferred to pede;n(y-hits)", 10, 0., 10));

  TDirectory *dirUsedTracks = directory->mkdir("usedTrackHists", "used tracks");
  this->addToDirectory(myUsedTrackHists1D, dirUsedTracks);
  this->addToDirectory(myUsedTrackHists2D, dirUsedTracks);

// ReferenceTrajectory
  myTrajectoryHists1D.push_back(new TH1F("validRefTraj", "validity of ReferenceTrajectory",
					 2, 0., 2.));

  myTrajectoryHists2D.push_back(new TProfile2D("profCorr",
					       "mean of |#rho|, #rho#neq0;hit x/y;hit x/y;",
					       2*nHits, 0., nHits, 2*nHits, 0., nHits));
  myTrajectoryHists2D.push_back
    (new TProfile2D("profCorrOffXy", "mean of |#rho|, #rho#neq0, no xy_{hit};hit x/y;hit x/y;",
		    2*nHits, 0., nHits, 2*nHits, 0., nHits));

  myTrajectoryHists2D.push_back(new TH2F("hitCorrOffBlock",
					 "hit correlations: off-block-diagonals;N(hit);#rho",
					 2*nHits, 0., nHits, 81, -.06, .06));
  TArrayD logBins(102);
  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-11, .1);
  myTrajectoryHists2D.push_back(new TH2F("hitCorrOffBlockLog",
					 "hit correlations: off-block-diagonals;N(hit);|#rho|",
					 2*nHits, 0., nHits, logBins.GetSize()-1, logBins.GetArray()));

  myTrajectoryHists2D.push_back(new TH2F("hitCorrXy", "hit correlations: xy;N(hit);#rho",
					 nHits, 0., nHits, 81, -.5, .5));
  myTrajectoryHists2D.push_back
    (new TH2F("hitCorrXyValid", "hit correlations: xy, 2D-det.;N(hit);#rho",
	      nHits, 0., nHits, 81, -.02, .02));
  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-10, .5);
  myTrajectoryHists2D.push_back(new TH2F("hitCorrXyLog", "hit correlations: xy;N(hit);|#rho|",
					 nHits, 0., nHits, logBins.GetSize()-1, logBins.GetArray()));
  myTrajectoryHists2D.push_back
    (new TH2F("hitCorrXyLogValid", "hit correlations: xy, 2D-det.;N(hit);|#rho|",
	      nHits, 0., nHits, logBins.GetSize()-1, logBins.GetArray()));


  myTrajectoryHists1D.push_back(new TH1F("measLocX", "local x measurements;x", 101, -6., 6.));
  myTrajectoryHists1D.push_back(new TH1F("measLocY", "local y measurements, 2D-det.;y",
					 101, -10., 10.));
  myTrajectoryHists1D.push_back(new TH1F("trajLocX", "local x trajectory;x", 101, -6., 6.));
  myTrajectoryHists1D.push_back(new TH1F("trajLocY", "local y trajectory, 2D-det.;y",
					 101, -10., 10.));

  myTrajectoryHists1D.push_back(new TH1F("residLocX", "local x residual;#Deltax", 101, -.75, .75));
  myTrajectoryHists1D.push_back(new TH1F("residLocY", "local y residual, 2D-det.;#Deltay",
					 101, -2., 2.));
  myTrajectoryHists1D.push_back(new TH1F("reduResidLocX", "local x reduced residual;#Deltax/#sigma",
					 101, -20., 20.));
  myTrajectoryHists1D.push_back
    (new TH1F("reduResidLocY", "local y reduced residual, 2D-det.;#Deltay/#sigma", 101, -20., 20.));

  // 2D vs. hit
  myTrajectoryHists2D.push_back(new TH2F("measLocXvsHit", "local x measurements;hit;x", 
					 nHits, 0., nHits, 101, -6., 6.));
  myTrajectoryHists2D.push_back(new TH2F("measLocYvsHit", "local y measurements, 2D-det.;hit;y",
					 nHits, 0., nHits, 101, -10., 10.));
  myTrajectoryHists2D.push_back(new TH2F("trajLocXvsHit", "local x trajectory;hit;x",
					 nHits, 0., nHits, 101, -6., 6.));
  myTrajectoryHists2D.push_back(new TH2F("trajLocYvsHit", "local y trajectory, 2D-det.;hit;y",
					 nHits, 0., nHits,  101, -10., 10.));

  myTrajectoryHists2D.push_back(new TH2F("residLocXvsHit", "local x residual;hit;#Deltax",
					 nHits, 0., nHits, 101, -.75, .75));
  myTrajectoryHists2D.push_back(new TH2F("residLocYvsHit", "local y residual, 2D-det.;hit;#Deltay",
					 nHits, 0., nHits, 101, -1., 1.));
  myTrajectoryHists2D.push_back
    (new TH2F("reduResidLocXvsHit", "local x reduced residual;hit;#Deltax/#sigma",
	      nHits, 0., nHits, 101, -20., 20.));
  myTrajectoryHists2D.push_back
    (new TH2F("reduResidLocYvsHit", "local y reduced residual, 2D-det.;hit;#Deltay/#sigma",
	      nHits, 0., nHits, 101, -20., 20.));


  myTrajectoryHists2D.push_back(new TProfile2D("profDerivatives",
					       "mean derivatives;hit x/y;parameter;",
					       2*nHits, 0., nHits, 10, 0., 10.));

  myTrajectoryHists2D.push_back
    (new TH2F("derivatives", "derivative;parameter;#partial(x/y)_{local}/#partial(param)",
	      10, 0., 10., 101, -20., 20.));
  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-12, 100.);
  myTrajectoryHists2D.push_back
    (new TH2F("derivativesLog", "|derivative|;parameter;|#partial(x/y)_{local}/#partial(param)|",
	      10, 0., 10., logBins.GetSize()-1, logBins.GetArray()));
  myTrajectoryHists2D.push_back
    (new TH2F("derivativesVsPhi", 
              "derivatives vs. #phi;#phi(geomDet);#partial(x/y)_{local}/#partial(param)",
	      50, -TMath::Pi(), TMath::Pi(), 101, -300., 300.));
  //  myTrajectoryHists2D.back()->SetBit(TH1::kCanRebin);

  TDirectory *dirTraject = directory->mkdir("refTrajectoryHists", "ReferenceTrajectory's");
  this->addToDirectory(myTrajectoryHists2D, dirTraject);
  this->addToDirectory(myTrajectoryHists1D, dirTraject);

// derivatives hists
  myDerivHists2D.push_back
    (new TH2F("localDerivsPar","local derivatives vs. paramater;parameter;#partial/#partial(param)",
              6, 0., 6., 101, -200., 200.));
  myDerivHists2D.push_back
    (new TH2F("localDerivsPhi","local derivatives vs. #phi(det);#phi(det);#partial/#partial(param)",
              51, -TMath::Pi(), TMath::Pi(), 101, -150., 150.));
  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-13, 150.);
  myDerivHists2D.push_back
    (new TH2F("localDerivsParLog",
              "local derivatives (#neq 0) vs. parameter;parameter;|#partial/#partial(param)|",
              6, 0., 6., logBins.GetSize()-1, logBins.GetArray()));
  myDerivHists2D.push_back
    (new TH2F("localDerivsPhiLog",
              "local derivatives (#neq 0) vs. #phi(det);#phi(det);|#partial/#partial(param)|",
              51, -TMath::Pi(), TMath::Pi(), logBins.GetSize()-1, logBins.GetArray()));
  const unsigned int maxParNum = PedeLabelerBase::theMaxNumParam;
  myDerivHists2D.push_back
    (new TH2F("globalDerivsPar",
              "global derivatives vs. paramater;parameter;#partial/#partial(param)",
              maxParNum, 0., maxParNum, 100, -200, 200));
  myDerivHists2D.push_back
    (new TH2F("globalDerivsPhi",
              "global derivatives vs. #phi(det);#phi(det);#partial/#partial(param)",
              51, -TMath::Pi(), TMath::Pi(), 102, -1.02, 1.02));
  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-7, 5.e2);
  myDerivHists2D.push_back
    (new TH2F("globalDerivsParLog",
              "global derivatives (#neq 0) vs. parameter;parameter;|#partial/#partial(param)|",
              maxParNum, 0., maxParNum, logBins.GetSize()-1, logBins.GetArray()));
  myDerivHists2D.push_back
    (new TH2F("globalDerivsPhiLog",
              "global derivatives (#neq 0) vs. #phi(det);#phi(det);|#partial/#partial(param)|",
              51, -TMath::Pi(), TMath::Pi(), logBins.GetSize()-1, logBins.GetArray()));
//   this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.e-40, 1.E-35);
//   myDerivHists2D.push_back
//     (new TH2F("globalDerivsPhiLog2",
//               "global derivatives (#neq 0) vs. #phi(det);#phi(det);|#partial/#partial(param)|",
//               51, -TMath::Pi(), TMath::Pi(), logBins.GetSize()-1, logBins.GetArray()));

  TDirectory *dirDerivs = directory->mkdir("derivatives", "derivatives etc.");
  this->addToDirectory(myDerivHists2D, dirDerivs);

// residual hists
  myResidHists2D.push_back(new TH2F("residPhi","residuum vs. #phi(det);#phi(det);residuum[cm]",
                                    51, -TMath::Pi(), TMath::Pi(), 101, -.5, .5));
  myResidHists2D.push_back(new TH2F("sigmaPhi","#sigma vs. #phi(det);#phi(det);#sigma[cm]",
                                    51, -TMath::Pi(), TMath::Pi(), 101, .0, .1));
  myResidHists2D.push_back(new TH2F("reduResidPhi",
                                    "residuum/#sigma vs. #phi(det);#phi(det);residuum/#sigma",
                                    51, -TMath::Pi(), TMath::Pi(), 101, -14., 14.));
  
//   myResidHists2D.push_back(new TProfile2D("residXProfXy",
// 					  "mean |residuum| (u);x [cm];y [cm];#LT|residuum|#GT",
// 					  25, -110., 110., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("residXProfZr",
// 					  "mean |residuum| (u);z [cm];r_{#pm} [cm];#LT|residuum|#GT",
// 					  25, -275., 275., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("residYProfXy",
// 					  "mean |residuum| (v);x [cm];y [cm];#LT|residuum|#GT",
// 					  25, -110., 110., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("residYProfZr",
// 					  "mean |residuum| (v);z [cm];r_{#pm} [cm];#LT|residuum|#GT",
// 					  25, -275., 275., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("reduResidXProfXy",
// 					  "mean |residuum/#sigma| (u);x [cm];y [cm];#LT|res./#sigma|#GT",
// 					  25, -110., 110., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("reduResidXProfZr",
// 					  "mean |residuum/#sigma| (u);z [cm];r_{#pm} [cm];#LT|res./#sigma|#GT",
// 					  25, -275., 275., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("reduResidYProfXy",
// 					  "mean |residuum/#sigma| (v);x [cm];y [cm];#LT|res./#sigma|#GT",
// 					  25, -110., 110., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("reduResidYProfZr",
// 					  "mean |residuum/#sigma| (v);z [cm];r_{#pm} [cm];#LT|res./#sigma|#GT",
// 					  25, -275., 275., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("sigmaXProfXy",
// 					  "mean sigma (u);x [cm];y [cm];#LT#sigma#GT",
// 					  25, -110., 110., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("sigmaXProfZr",
// 					  "mean sigma (u);z [cm];r_{#pm} [cm];#LT#sigma#GT",
// 					  25, -275., 275., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("sigmaYProfXy",
// 					  "mean sigma (v);x [cm];y [cm];#LT#sigma#GT",
// 					  25, -110., 110., 25, -110., 110.));
//   myResidHists2D.push_back(new TProfile2D("sigmaYProfZr",
// 					  "mean sigma (v);z [cm];r_{#pm} [cm];#LT#sigma#GT",
// 					  25, -275., 275., 25, -110., 110.));
  
  TDirectory *dirResid = directory->mkdir("residuals", "hit residuals, sigma,...");
  this->addToDirectory(myResidHists2D, dirResid);

// Residuum, hit sigma and res./sigma for all sensor/track angles and separated for large and
// small angles with respect to the sensor normal in sensitive direction. 
  // Here for x-measurements:
  std::vector<TH1*> allResidHistsX;
  allResidHistsX.push_back(new TH1F("resid", "hit residuals;residuum [cm]", 101,-.5,.5));//51,-.05, .05));
  //allResidHistsX.back()->SetBit(TH1::kCanRebin);
  allResidHistsX.push_back(new TH1F("sigma", "hit uncertainties;#sigma [cm]", 100,0.,1.));//50, 0., .02));
  //allResidHistsX.back()->SetBit(TH1::kCanRebin);
  allResidHistsX.push_back(new TH1F("reduResid", "reduced hit residuals;res./#sigma",
				    101, -10., 10.));//51, -3., 3.));
  //  allResidHistsX.back()->SetBit(TH1::kCanRebin);
  allResidHistsX.push_back(new TH1F("angle", "#phi_{tr} wrt normal (sens. plane);#phi_{n}^{sens}",
				    50, 0., TMath::PiOver2()));
  allResidHistsX.push_back(new TH2F("residVsAngle",
				    "residuum vs. #phi_{tr} wrt normal (sens. plane);#phi_{n}^{sens};residuum [cm]",
				    50, 0., TMath::PiOver2(), 51, -1., 1.));
  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-6, 100.);
  allResidHistsX.push_back(new TH2F("sigmaVsAngle",
				    "#sigma vs. #phi_{tr} wrt normal (sens. plane);#phi_{n}^{sens};#sigma [cm]",
				    50, 0., TMath::PiOver2(), logBins.GetSize()-1, logBins.GetArray()));
  allResidHistsX.push_back(new TH2F("reduResidVsAngle",
				    "reduced residuum vs. #phi_{tr} wrt normal (sens. plane);#phi_{n}^{sens};res./#sigma",
				    50, 0., TMath::PiOver2(), 51, -15., 15.));

  allResidHistsX.push_back(new TH1F("residGt45",
				    "hit residuals (#phi_{n}^{sens}>45#circ);residuum [cm]",
				    101, -.5, .5));//51, -.05, .05));
  // allResidHistsX.back()->SetBit(TH1::kCanRebin);
  allResidHistsX.push_back(new TH1F("sigmaGt45",
				    "hit uncertainties(#phi_{n}^{sens}>45#circ);#sigma [cm]",
				     100, 0., 1.));//50, 0., .02));
  // allResidHistsX.back()->SetBit(TH1::kCanRebin);
  allResidHistsX.push_back(new TH1F("reduResidGt45",
				    "reduced hit residuals(#phi_{n}^{sens}>45#circ);res./#sigma",
                                    101, -10., 10.));//51,-3.,3.));
  // allResidHistsX.back()->SetBit(TH1::kCanRebin);
  allResidHistsX.push_back(new TH1F("residLt45",
				    "hit residuals (#phi_{n}^{sens}<45#circ);residuum [cm]",
				    101, -.5, .5));//51, -.15, .15));
  // allResidHistsX.back()->SetBit(TH1::kCanRebin);
  allResidHistsX.push_back(new TH1F("sigmaLt45",
				    "hit uncertainties(#phi_{n}^{sens}<45#circ);#sigma [cm]",
				    100, 0., 1.));//50, 0., .01));
  // allResidHistsX.back()->SetBit(TH1::kCanRebin);
  allResidHistsX.push_back(new TH1F("reduResidLt45",
				    "reduced hit residuals(#phi_{n}^{sens}<45#circ);res./#sigma",
				    101, -10., 10.));//51,-3.,3.));
  myResidHistsVec1DX.push_back(allResidHistsX); // at [0] for all subdets together...
  // ... then separately - indices/order like DetId.subdetId() in tracker:
  myResidHistsVec1DX.push_back(this->cloneHists(allResidHistsX, "TPB", " x-coord. in pixel barrel"));
  myResidHistsVec1DX.push_back(this->cloneHists(allResidHistsX, "TPE", " x-coord. in pixel discs"));
  myResidHistsVec1DX.push_back(this->cloneHists(allResidHistsX, "TIB", " x-coord. in TIB"));
  myResidHistsVec1DX.push_back(this->cloneHists(allResidHistsX, "TID", " x-coord. in TID"));
  myResidHistsVec1DX.push_back(this->cloneHists(allResidHistsX, "TOB", " x-coord. in TOB"));
  myResidHistsVec1DX.push_back(this->cloneHists(allResidHistsX, "TEC", " x-coord. in TEC"));
  // finally, differential in hit number (but subdet independent)
  for (unsigned int iHit = 0; iHit < 30; ++iHit) { // 4: for each hit fill only angle independent plots
    for (unsigned int iHist = 0; iHist < 4 && iHist < allResidHistsX.size(); ++iHist) {
      TH1 *h = allResidHistsX[iHist];
      myResidHitHists1DX.push_back(static_cast<TH1*>(h->Clone(Form("%s_%d", h->GetName(), iHit))));
      myResidHitHists1DX.back()->SetTitle(Form("%s, hit %d", h->GetTitle(), iHit));
    }
  }

  TDirectory *dirResidX =
    (dirResid ? dirResid : directory)->mkdir("X", "hit residuals etc. for x measurements");
  this->addToDirectory(myResidHitHists1DX, dirResidX);
  for (std::vector<std::vector<TH1*> >::iterator vecIter = myResidHistsVec1DX.begin(),
	 vecIterEnd = myResidHistsVec1DX.end(); vecIter != vecIterEnd; ++vecIter) {
    this->addToDirectory(*vecIter, dirResidX);
  }

  // Now clone the same as above for y-ccordinate:
  myResidHistsVec1DY.push_back(this->cloneHists(allResidHistsX, "", " y-coord."));// all subdets
  std::vector<TH1*> &yHists1D = allResidHistsX;//myResidHistsVec1DY.back(); crashes? ROOT?
  myResidHistsVec1DY.push_back(this->cloneHists(yHists1D, "TPB", " y-coord. in pixel barrel"));
  myResidHistsVec1DY.push_back(this->cloneHists(yHists1D, "TPE", " y-coord. in pixel discs"));
  myResidHistsVec1DY.push_back(this->cloneHists(yHists1D, "TIB", " y-coord. in TIB"));
  myResidHistsVec1DY.push_back(this->cloneHists(yHists1D, "TID", " y-coord. in TID"));
  myResidHistsVec1DY.push_back(this->cloneHists(yHists1D, "TOB", " y-coord. in TOB"));
  myResidHistsVec1DY.push_back(this->cloneHists(yHists1D, "TEC", " y-coord. in TEC"));
  myResidHitHists1DY = this->cloneHists(myResidHitHists1DX, "", " y-coord.");// diff. in nHit

  TDirectory *dirResidY = 
    (dirResid ? dirResid : directory)->mkdir("Y", "hit residuals etc. for y measurements");
  this->addToDirectory(myResidHitHists1DY, dirResidY);
  for (std::vector<std::vector<TH1*> >::iterator vecIter = myResidHistsVec1DY.begin(),
	 vecIterEnd = myResidHistsVec1DY.end(); vecIter != vecIterEnd; ++vecIter) {
    this->addToDirectory(*vecIter, dirResidY);
  }

  // farme-to-frame derivatives
  myFrame2FrameHists2D.push_back(new TProfile2D("frame2frame",
                                                "mean frame to frame derivatives;col;row",
                                                6, 0., 6., 6, 0., 6.));
  myFrame2FrameHists2D.push_back(new TProfile2D("frame2frameAbs",
                                                "mean |frame to frame derivatives|, #neq0;col;row",
                                                6, 0., 6., 6, 0., 6.));

  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-36, 100.);
  for (unsigned int i = 0; i < 6; ++i) {
    for (unsigned int j = 0; j < 6; ++j) {
      myFrame2FrameHists2D.push_back
        (new TH2F(Form("frame2framePhi%d%d", i, j),
                  Form("frame to frame derivatives, %d%d;#phi(aliDet);deriv",i,j),
                  51, -TMath::Pi(), TMath::Pi(), 10, 0., 1.));
      myFrame2FrameHists2D.back()->SetBit(TH1::kCanRebin);
      myFrame2FrameHists2D.push_back
        (new TH2F(Form("frame2frameR%d%d", i, j),
                  Form("frame to frame derivatives, %d%d;r(aliDet);deriv",i,j),
                  51, 0., 110., 10, 0., 1.));
      myFrame2FrameHists2D.back()->SetBit(TH1::kCanRebin);

      myFrame2FrameHists2D.push_back
        (new TH2F(Form("frame2framePhiLog%d%d", i, j),
                  Form("frame to frame |derivatives|, %d%d, #neq0;#phi(aliDet);deriv",i,j),
                  51, -TMath::Pi(), TMath::Pi(), logBins.GetSize()-1, logBins.GetArray()));
      myFrame2FrameHists2D.push_back
        (new TH2F(Form("frame2frameRLog%d%d", i, j),
                  Form("frame to frame |derivatives|, %d%d, #neq0;r(aliDet);deriv",i,j),
                  51, 0., 110., logBins.GetSize()-1, logBins.GetArray()));
    }
  }

  TDirectory *dirF2f = directory->mkdir("frame2FrameHists", "derivatives etc.");
  this->addToDirectory(myFrame2FrameHists2D, dirF2f);


  myCorrHists.push_back(new TH1F("xyCorrTPB", "#rho_{xy} in pixel barrel", 50, -.5, .5));
  myCorrHists.push_back(new TH1F("xyCorrTPE", "#rho_{xy} in forward pixel", 50, -.5, .5));
  myCorrHists.push_back(new TH1F("xyCorrTIB", "#rho_{xy} in TIB", 50, -.5, .5));
  myCorrHists.push_back(new TH1F("xyCorrTID", "#rho_{xy} in TID", 50, -1., 1.));
  myCorrHists.push_back(new TH1F("xyCorrTOB", "#rho_{xy} in TOB", 50, -.5, .5));
  myCorrHists.push_back(new TH1F("xyCorrTEC", "#rho_{xy} in TEC", 50, -1., 1.));
  TDirectory *dirCorr = directory->mkdir("hitCorrelationHists", "correlations");
  this->addToDirectory( myCorrHists, dirCorr);

  // PXB Survey
  myPxbSurveyHists.push_back(new TH1F("PxbSurvChi2_lo", "#chi^{2} from PXB survey", 25, 0, 25));
  myPxbSurveyHists.push_back(new TH1F("PxbSurvChi2_md", "#chi^{2} from PXB survey", 25, 0, 500));
  myPxbSurveyHists.push_back(new TH1F("PxbSurvChi2_hi", "#chi^{2} from PXB survey", 25, 0, 50000));
  myPxbSurveyHists.push_back(new TH1F("PxbSurvChi2prob", "Math::Prob(#chi^{2},4) from PXB survey", 25, 0, 1));
  myPxbSurveyHists.push_back(new TH1F("PxbSurv_a0", "a_{0} from PXB survey", 100, -3000, 3000));
  myPxbSurveyHists.push_back(new TH1F("PxbSurv_a0_abs", "fabs(a_{0}) from PXB survey", 100, 0, 3000));
  myPxbSurveyHists.push_back(new TH1F("PxbSurv_a1", "a_{1} from PXB survey", 100, -6000, 6000));
  myPxbSurveyHists.push_back(new TH1F("PxbSurv_a1_abs", "fabs(a_{1}) from PXB survey", 100, 0, 6000));
  myPxbSurveyHists.push_back(new TH1F("PxbSurv_scale", "scale (#sqrt{a_{2}^{2}+a_{3}^{2}}) from PXB survey", 100, 0, 1500));
  myPxbSurveyHists.push_back(new TH1F("PxbSurv_phi", "angle(#atan{a_{3}/a_{4}}) from PXB survey", 100, -.05, .05));
  TDirectory *dirPxbSurvey = directory->mkdir("PxbSurveyHists", "PxbSurvey");
  this->addToDirectory( myPxbSurveyHists, dirPxbSurvey);

  oldDir->cd();
  return true;
}


//__________________________________________________________________
bool MillePedeMonitor::equidistLogBins(double* bins, int nBins, 
					double first, double last) const
{
  // Filling 'bins' with borders of 'nBins' bins between 'first' and 'last'
  // that are equidistant when viewed in log scale,
  // so 'bins' must have length nBins+1;
  // If 'first', 'last' or 'nBins' are not positive, failure is reported.

  if (nBins < 1 || first <= 0. || last <= 0.) return false;

  bins[0] = first;
  bins[nBins] = last;
  const double firstLog = TMath::Log10(bins[0]);
  const double lastLog  = TMath::Log10(bins[nBins]);
  for (int i = 1; i < nBins; ++i) {
    bins[i] = TMath::Power(10., firstLog + i*(lastLog-firstLog)/(nBins));
  }

  return true;
}

//__________________________________________________________________
void MillePedeMonitor::fillTrack(const reco::Track *track)
{
  this->fillTrack(track, myTrackHists1D, myTrackHists2D);
}

//__________________________________________________________________
void MillePedeMonitor::fillUsedTrack(const reco::Track *track, unsigned int nHitX,
				     unsigned int nHitY)
{
  // these hist exist only for 'used track' hists:
  static const int iUsedX = this->GetIndex(myUsedTrackHists1D, "usedHitsX");
  myUsedTrackHists1D[iUsedX]->Fill(nHitX);
  static const int iUsedY = this->GetIndex(myUsedTrackHists1D, "usedHitsY");
  myUsedTrackHists1D[iUsedY]->Fill(nHitY);

  if (!track) return;
  this->fillTrack(track, myUsedTrackHists1D, myUsedTrackHists2D);
}

//__________________________________________________________________
void MillePedeMonitor::fillTrack(const reco::Track *track, std::vector<TH1*> &trackHists1D,
				 std::vector<TH2*> &trackHists2D)
{
  if (!track) return;

  const reco::TrackBase::Vector p(track->momentum());

  static const int iPtLog = this->GetIndex(trackHists1D, "ptTrackLogBins");
  trackHists1D[iPtLog]->Fill(track->pt());
  static const int iPt = this->GetIndex(trackHists1D, "ptTrack");
  trackHists1D[iPt]->Fill(track->pt());
  static const int iP = this->GetIndex(trackHists1D, "pTrack");
  trackHists1D[iP]->Fill(p.R());
  static const int iEta = this->GetIndex(trackHists1D, "etaTrack");
  trackHists1D[iEta]->Fill(p.Eta());
  static const int iTheta = this->GetIndex(trackHists1D, "thetaTrack");
  trackHists1D[iTheta]->Fill(p.Theta());
  static const int iPhi = this->GetIndex(trackHists1D, "phiTrack");
  trackHists1D[iPhi]->Fill(p.Phi());

  static const int iNhit = this->GetIndex(trackHists1D, "nHitTrack");
  trackHists1D[iNhit]->Fill(track->numberOfValidHits());
  static const int iNhitInvalid = this->GetIndex(trackHists1D, "nHitInvalidTrack");
  trackHists1D[iNhitInvalid]->Fill(track->numberOfLostHits());

  int nhitinTIB = 0, nhitinTOB = 0, nhitinTID = 0;
  int nhitinTEC = 0, nhitinBPIX = 0, nhitinFPIX = 0, nhitinPIXEL=0;
  int nhitinENDCAP = 0, nhitinENDCAPplus = 0, nhitinENDCAPminus = 0;
  int nhitinTIDplus = 0, nhitinTIDminus = 0;
  int nhitinFPIXplus = 0, nhitinFPIXminus = 0;
  int nhitinTECplus = 0, nhitinTECminus = 0;
  unsigned int thishit = 0;

  for (trackingRecHit_iterator iHit = track->recHitsBegin(); iHit != track->recHitsEnd(); ++iHit) {
    thishit++;
    const DetId detId((*iHit)->geographicalId());
    const int subdetId = detId.subdetId(); 

    if (!(*iHit)->isValid()) continue; // only real hits count as in track->numberOfValidHits()
    if (detId.det() != DetId::Tracker) {
      edm::LogError("DetectorMismatch") << "@SUB=MillePedeMonitor::fillTrack"
                                        << "DetId.det() != DetId::Tracker (=" << DetId::Tracker
                                        << "), but " << detId.det() << ".";
    }

    if      (SiStripDetId::TIB == subdetId) ++nhitinTIB;
    else if (SiStripDetId::TOB == subdetId) ++nhitinTOB;
    else if (SiStripDetId::TID == subdetId) {
      ++nhitinTID;
      ++nhitinENDCAP;
      
      if (trackerTopology->tidIsZMinusSide(detId)) {
        ++nhitinTIDminus;
        ++nhitinENDCAPminus;
      }
      else if (trackerTopology->tidIsZPlusSide(detId)) {
        ++nhitinTIDplus;
        ++nhitinENDCAPplus;
      }
    }
    else if (SiStripDetId::TEC == subdetId) {
      ++nhitinTEC;
      ++nhitinENDCAP;
      
      if (trackerTopology->tecIsZMinusSide(detId)) {
        ++nhitinTECminus;
        ++nhitinENDCAPminus;
      }
      else if (trackerTopology->tecIsZPlusSide(detId)) {
        ++nhitinTECplus;
        ++nhitinENDCAPplus;
      }
    }
    else if (            kBPIX == subdetId) {++nhitinBPIX;++nhitinPIXEL;}
    else if (            kFPIX == subdetId) {
      ++nhitinFPIX;
      ++nhitinPIXEL;
      
      if (trackerTopology->pxfSide(detId)==1) ++nhitinFPIXminus;
      else if (trackerTopology->pxfSide(detId)==2) ++nhitinFPIXplus;
    }

  } // end loop on hits

  static const int iNhit01 = this->GetIndex(trackHists1D, "nHitBPIXTrack");
  trackHists1D[iNhit01]->Fill(nhitinBPIX);
  static const int iNhit02 = this->GetIndex(trackHists1D, "nHitFPIXplusTrack");
  trackHists1D[iNhit02]->Fill(nhitinFPIXplus);
  static const int iNhit03 = this->GetIndex(trackHists1D, "nHitFPIXminusTrack");
  trackHists1D[iNhit03]->Fill(nhitinFPIXminus);
  static const int iNhit04 = this->GetIndex(trackHists1D, "nHitFPIXTrack");
  trackHists1D[iNhit04]->Fill(nhitinFPIX);
  static const int iNhit05 = this->GetIndex(trackHists1D, "nHitPIXELTrack");
  trackHists1D[iNhit05]->Fill(nhitinPIXEL);
  static const int iNhit06 = this->GetIndex(trackHists1D, "nHitTIBTrack");
  trackHists1D[iNhit06]->Fill(nhitinTIB);
  static const int iNhit07 = this->GetIndex(trackHists1D, "nHitTOBTrack");
  trackHists1D[iNhit07]->Fill(nhitinTOB);
  static const int iNhit08 = this->GetIndex(trackHists1D, "nHitTIDplusTrack");
  trackHists1D[iNhit08]->Fill(nhitinTIDplus);
  static const int iNhit09 = this->GetIndex(trackHists1D, "nHitTIDminusTrack");
  trackHists1D[iNhit09]->Fill(nhitinTIDminus);
  static const int iNhit10 = this->GetIndex(trackHists1D, "nHitTIDTrack");
  trackHists1D[iNhit10]->Fill(nhitinTID);
  static const int iNhit11 = this->GetIndex(trackHists1D, "nHitTECplusTrack");
  trackHists1D[iNhit11]->Fill(nhitinTECplus);
  static const int iNhit12 = this->GetIndex(trackHists1D, "nHitTECminusTrack");
  trackHists1D[iNhit12]->Fill(nhitinTECminus);
  static const int iNhit13 = this->GetIndex(trackHists1D, "nHitTECTrack");
  trackHists1D[iNhit13]->Fill(nhitinTEC);
  static const int iNhit14 = this->GetIndex(trackHists1D, "nHitENDCAPplusTrack");
  trackHists1D[iNhit14]->Fill(nhitinENDCAPplus);
  static const int iNhit15 = this->GetIndex(trackHists1D, "nHitENDCAPminusTrack");
  trackHists1D[iNhit15]->Fill(nhitinENDCAPminus);
  static const int iNhit16 = this->GetIndex(trackHists1D, "nHitENDCAPTrack");
  trackHists1D[iNhit16]->Fill(nhitinENDCAP);
  static const int iNhit17 = this->GetIndex(trackHists2D, "nHitENDCAPTrackMinusVsPlus");
  trackHists2D[iNhit17]->Fill(nhitinENDCAPplus,nhitinENDCAPminus);

  if (track->innerOk()) {
    const reco::TrackBase::Point firstPoint(track->innerPosition());
    static const int iR1 = this->GetIndex(trackHists1D, "r1Track");
    trackHists1D[iR1]->Fill(firstPoint.Rho());
    const double rSigned1 = (firstPoint.y() > 0 ? firstPoint.Rho() : -firstPoint.Rho());
    static const int iR1Signed = this->GetIndex(trackHists1D, "r1TrackSigned");
    trackHists1D[iR1Signed]->Fill(rSigned1);
    static const int iZ1 = this->GetIndex(trackHists1D, "z1Track");
    trackHists1D[iZ1]->Fill(firstPoint.Z());
    static const int iZ1Full = this->GetIndex(trackHists1D, "z1TrackFull");
    trackHists1D[iZ1Full]->Fill(firstPoint.Z());
    static const int iY1 = this->GetIndex(trackHists1D, "y1Track");
    trackHists1D[iY1]->Fill(firstPoint.Y());
    static const int iPhi1 = this->GetIndex(trackHists1D, "phi1Track");
    trackHists1D[iPhi1]->Fill(firstPoint.phi());
    static const int iRz1Full = this->GetIndex(trackHists2D, "rz1TrackFull");
    trackHists2D[iRz1Full]->Fill(firstPoint.Z(), rSigned1);
    static const int iXy1 = this->GetIndex(trackHists2D, "xy1Track");
    trackHists2D[iXy1]->Fill(firstPoint.X(), firstPoint.Y());
  }

  if (track->outerOk()) {
    const reco::TrackBase::Point lastPoint(track->outerPosition());
    static const int iRlast = this->GetIndex(trackHists1D, "rLastTrack");
    trackHists1D[iRlast]->Fill(lastPoint.Rho());
    static const int iZlast = this->GetIndex(trackHists1D, "zLastTrack");
    trackHists1D[iZlast]->Fill(lastPoint.Z());
    static const int iYlast = this->GetIndex(trackHists1D, "yLastTrack");
    trackHists1D[iYlast]->Fill(lastPoint.Y());
    static const int iPhiLast = this->GetIndex(trackHists1D, "phiLastTrack");
    trackHists1D[iPhiLast]->Fill(lastPoint.phi());
  }

  static const int iChi2Ndf = this->GetIndex(trackHists1D, "chi2PerNdf");
  trackHists1D[iChi2Ndf]->Fill(track->normalizedChi2());

  static const int iImpZ = this->GetIndex(trackHists1D, "impParZ");
  trackHists1D[iImpZ]->Fill(track->dz());
  static const int iImpZerr = this->GetIndex(trackHists1D, "impParErrZ");
  trackHists1D[iImpZerr]->Fill(track->dzError());


  static const int iImpRphi = this->GetIndex(trackHists1D, "impParRphi");
  trackHists1D[iImpRphi]->Fill(track->d0());
  static const int iImpRphiErr = this->GetIndex(trackHists1D, "impParErrRphi");
  trackHists1D[iImpRphiErr]->Fill(track->d0Error());

}

//__________________________________________________________________
void MillePedeMonitor::fillRefTrajectory(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr)
{

  static const int iValid = this->GetIndex(myTrajectoryHists1D, "validRefTraj");
  if (refTrajPtr->isValid()) {
    myTrajectoryHists1D[iValid]->Fill(1.);
  } else {
    myTrajectoryHists1D[iValid]->Fill(0.);
    return;
  }

  const AlgebraicSymMatrix &covMeasLoc = refTrajPtr->measurementErrors();
  const AlgebraicVector &measurementsLoc = refTrajPtr->measurements();
  const AlgebraicVector &trajectoryLoc = refTrajPtr->trajectoryPositions();
  const TransientTrackingRecHit::ConstRecHitContainer &recHits = refTrajPtr->recHits();
  const AlgebraicMatrix &derivatives = refTrajPtr->derivatives();
  
// CHK
  const int nRow = refTrajPtr->numberOfHitMeas(); 

  for (int iRow = 0; iRow < nRow; ++iRow) {
    const double residuum = measurementsLoc[iRow] - trajectoryLoc[iRow];
    const double covMeasLocRow = covMeasLoc[iRow][iRow];
    const bool is2DhitRow = (!recHits[iRow/2]->detUnit() // FIXME: as in MillePedeAlignmentAlgorithm::is2D()
	 		     || recHits[iRow/2]->detUnit()->type().isTrackerPixel());
    //the GeomDetUnit* is zero for composite hits (matched hits in the tracker,...). 
    if (TMath::Even(iRow)) { // local x
      static const int iMeasLocX = this->GetIndex(myTrajectoryHists1D, "measLocX");
      myTrajectoryHists1D[iMeasLocX]->Fill(measurementsLoc[iRow]);
      static const int iTrajLocX = this->GetIndex(myTrajectoryHists1D, "trajLocX");
      myTrajectoryHists1D[iTrajLocX]->Fill(trajectoryLoc[iRow]);
      static const int iResidLocX = this->GetIndex(myTrajectoryHists1D, "residLocX");
      myTrajectoryHists1D[iResidLocX]->Fill(residuum);

      static const int iMeasLocXvsHit = this->GetIndex(myTrajectoryHists2D, "measLocXvsHit");
      myTrajectoryHists2D[iMeasLocXvsHit]->Fill(iRow/2, measurementsLoc[iRow]);
      static const int iTrajLocXvsHit = this->GetIndex(myTrajectoryHists2D, "trajLocXvsHit");
      myTrajectoryHists2D[iTrajLocXvsHit]->Fill(iRow/2, trajectoryLoc[iRow]);
      static const int iResidLocXvsHit = this->GetIndex(myTrajectoryHists2D, "residLocXvsHit");
      myTrajectoryHists2D[iResidLocXvsHit]->Fill(iRow/2, residuum);

      if (covMeasLocRow > 0.) { 
	static const int iReduResidLocX = this->GetIndex(myTrajectoryHists1D, "reduResidLocX");
	myTrajectoryHists1D[iReduResidLocX]->Fill(residuum / TMath::Sqrt(covMeasLocRow));
	static const int iReduResidLocXvsHit = 
	  this->GetIndex(myTrajectoryHists2D, "reduResidLocXvsHit");
	myTrajectoryHists2D[iReduResidLocXvsHit]->Fill(iRow/2, residuum/TMath::Sqrt(covMeasLocRow));
      }
    } else if (is2DhitRow) { // local y, 2D detectors only

      static const int iMeasLocY = this->GetIndex(myTrajectoryHists1D, "measLocY");
      myTrajectoryHists1D[iMeasLocY]->Fill(measurementsLoc[iRow]);
      static const int iTrajLocY = this->GetIndex(myTrajectoryHists1D, "trajLocY");
      myTrajectoryHists1D[iTrajLocY]->Fill(trajectoryLoc[iRow]);
      static const int iResidLocY = this->GetIndex(myTrajectoryHists1D, "residLocY");
      myTrajectoryHists1D[iResidLocY]->Fill(residuum);

      static const int iMeasLocYvsHit = this->GetIndex(myTrajectoryHists2D, "measLocYvsHit");
      myTrajectoryHists2D[iMeasLocYvsHit]->Fill(iRow/2, measurementsLoc[iRow]);
      static const int iTrajLocYvsHit = this->GetIndex(myTrajectoryHists2D, "trajLocYvsHit");
      myTrajectoryHists2D[iTrajLocYvsHit]->Fill(iRow/2, trajectoryLoc[iRow]);
      static const int iResidLocYvsHit = this->GetIndex(myTrajectoryHists2D, "residLocYvsHit");
      myTrajectoryHists2D[iResidLocYvsHit]->Fill(iRow/2, residuum);

      if (covMeasLocRow > 0.) {
	static const int iReduResidLocY = this->GetIndex(myTrajectoryHists1D, "reduResidLocY");
	myTrajectoryHists1D[iReduResidLocY]->Fill(residuum / TMath::Sqrt(covMeasLocRow));
	static const int iReduResidLocYvsHit = this->GetIndex(myTrajectoryHists2D,
							      "reduResidLocYvsHit");
	myTrajectoryHists2D[iReduResidLocYvsHit]->Fill(iRow/2, residuum/TMath::Sqrt(covMeasLocRow));
      }
    }

    float nHitRow = iRow/2; // '/2' not '/2.'!
    if (TMath::Odd(iRow)) nHitRow += 0.5; // y-hit gets 0.5
    // correlations
    for (int iCol = iRow+1; iCol < nRow; ++iCol) {
      double rho = TMath::Sqrt(covMeasLocRow + covMeasLoc[iCol][iCol]);
      rho = (0. == rho ? -2 : covMeasLoc[iRow][iCol] / rho);
      float nHitCol = iCol/2; //cf. comment nHitRow
      if (TMath::Odd(iCol)) nHitCol += 0.5; // dito
      //      myProfileCorr->Fill(nHitRow, nHitCol, TMath::Abs(rho));
      if (0. == rho) continue;
      static const int iProfileCorr = this->GetIndex(myTrajectoryHists2D, "profCorr");
      myTrajectoryHists2D[iProfileCorr]->Fill(nHitRow, nHitCol, TMath::Abs(rho));
      if (iRow+1 == iCol && TMath::Even(iRow)) { // i.e. if iRow is x and iCol the same hit's y
//  	static const int iProfileCorrXy = this->GetIndex(myTrajectoryHists2D,"profCorrOffXy");
// 	myTrajectoryHists2D[iProfileCorrOffXy]->Fill(iRow/2, rho);
	static const int iHitCorrXy = this->GetIndex(myTrajectoryHists2D, "hitCorrXy");
	myTrajectoryHists2D[iHitCorrXy]->Fill(iRow/2, rho);
	static const int iHitCorrXyLog = this->GetIndex(myTrajectoryHists2D, "hitCorrXyLog");
	myTrajectoryHists2D[iHitCorrXyLog]->Fill(iRow/2, TMath::Abs(rho));
	if (is2DhitRow) {
	  static const int iHitCorrXyValid = this->GetIndex(myTrajectoryHists2D, "hitCorrXyValid");
	  myTrajectoryHists2D[iHitCorrXyValid]->Fill(iRow/2, rho); // nhitRow??
	  static const int iHitCorrXyLogValid = this->GetIndex(myTrajectoryHists2D,
							       "hitCorrXyLogValid");
	  myTrajectoryHists2D[iHitCorrXyLogValid]->Fill(iRow/2, TMath::Abs(rho)); // nhitRow??
	}
      } else {
	static const int iProfCorrOffXy = this->GetIndex(myTrajectoryHists2D, "profCorrOffXy");
	myTrajectoryHists2D[iProfCorrOffXy]->Fill(nHitRow, nHitCol, TMath::Abs(rho));
	static const int iHitCorrOffBlock = this->GetIndex(myTrajectoryHists2D, "hitCorrOffBlock");
	myTrajectoryHists2D[iHitCorrOffBlock]->Fill(nHitRow, rho);
	static const int iHitCorOffBlkLg = this->GetIndex(myTrajectoryHists2D,"hitCorrOffBlockLog");
	myTrajectoryHists2D[iHitCorOffBlkLg]->Fill(nHitRow, TMath::Abs(rho));
      }
    } // end loop on columns of covariance
    
    // derivatives
    for (int iCol = 0; iCol < derivatives.num_col(); ++iCol) {
      static const int iProfDerivatives = this->GetIndex(myTrajectoryHists2D, "profDerivatives");
      myTrajectoryHists2D[iProfDerivatives]->Fill(nHitRow, iCol, derivatives[iRow][iCol]);
      static const int iDerivatives = this->GetIndex(myTrajectoryHists2D, "derivatives");
      myTrajectoryHists2D[iDerivatives]->Fill(iCol, derivatives[iRow][iCol]);
      static const int iDerivativesLog = this->GetIndex(myTrajectoryHists2D, "derivativesLog");
      myTrajectoryHists2D[iDerivativesLog]->Fill(iCol, TMath::Abs(derivatives[iRow][iCol]));
      static const int iDerivativesPhi = this->GetIndex(myTrajectoryHists2D, "derivativesVsPhi");
      myTrajectoryHists2D[iDerivativesPhi]->Fill(recHits[iRow/2]->det()->position().phi(),
                                                 derivatives[iRow][iCol]);
    }
  } // end loop on rows of covarianvce

}

//____________________________________________________________________
void MillePedeMonitor::fillDerivatives(const ConstRecHitPointer &recHit,
				       const float *localDerivs, unsigned int nLocal,
				       const float *globalDerivs, unsigned int nGlobal,
				       const int *labels)
{
  const double phi = recHit->det()->position().phi();

  static const int iLocPar = this->GetIndex(myDerivHists2D, "localDerivsPar");
  static const int iLocPhi = this->GetIndex(myDerivHists2D, "localDerivsPhi");
  static const int iLocParLog = this->GetIndex(myDerivHists2D, "localDerivsParLog");
  static const int iLocPhiLog = this->GetIndex(myDerivHists2D, "localDerivsPhiLog");
  for (unsigned int i = 0; i < nLocal; ++i) {
    myDerivHists2D[iLocPar]->Fill(i, localDerivs[i]);
    myDerivHists2D[iLocPhi]->Fill(phi, localDerivs[i]);
    if (localDerivs[i]) {
      myDerivHists2D[iLocParLog]->Fill(i, TMath::Abs(localDerivs[i]));
      myDerivHists2D[iLocPhiLog]->Fill(phi, TMath::Abs(localDerivs[i]));
    }
  }

  static const int iGlobPar = this->GetIndex(myDerivHists2D, "globalDerivsPar");
  static const int iGlobPhi = this->GetIndex(myDerivHists2D, "globalDerivsPhi");
  static const int iGlobParLog = this->GetIndex(myDerivHists2D, "globalDerivsParLog");
  static const int iGlobPhiLog = this->GetIndex(myDerivHists2D, "globalDerivsPhiLog");
//   static const int iGlobPhiLog2 = this->GetIndex(myDerivHists2D, "globalDerivsPhiLog2");
  for (unsigned int i = 0; i < nGlobal; ++i) {
    const unsigned int parNum = (labels ? (labels[i]%PedeLabelerBase::theMaxNumParam)-1 : i);
    myDerivHists2D[iGlobPar]->Fill(parNum, globalDerivs[i]);
    myDerivHists2D[iGlobPhi]->Fill(phi, globalDerivs[i]);
    if (globalDerivs[i]) {
      myDerivHists2D[iGlobParLog]->Fill(parNum, TMath::Abs(globalDerivs[i]));
      myDerivHists2D[iGlobPhiLog]->Fill(phi, TMath::Abs(globalDerivs[i]));
//       myDerivHists2D[iGlobPhiLog2]->Fill(phi, TMath::Abs(globalDerivs[i]));
    }
  }

}


//____________________________________________________________________
void MillePedeMonitor::fillResiduals(const ConstRecHitPointer &recHit,
				     const TrajectoryStateOnSurface &tsos,
				     unsigned int nHit, float residuum, float sigma, bool isY)
{
  // isY == false: x-measurements
  // isY == true:  y-measurements
  const GlobalPoint detPos(recHit->det()->position());
  const double phi = detPos.phi();
//  const double rSigned = (detPos.y() > 0. ? detPos.perp() : -detPos.perp());

  static const int iResPhi = this->GetIndex(myResidHists2D, "residPhi");
  myResidHists2D[iResPhi]->Fill(phi, residuum);
  static const int iSigPhi = this->GetIndex(myResidHists2D, "sigmaPhi");
  myResidHists2D[iSigPhi]->Fill(phi, sigma);
  if (sigma) {
    static const int iReduResPhi = this->GetIndex(myResidHists2D, "reduResidPhi");
    myResidHists2D[iReduResPhi]->Fill(phi, residuum/sigma);
  }

//   if (isY) {
//     static const int iResYXy = this->GetIndex(myResidHists2D, "residYProfXy");
//     myResidHists2D[iResYXy]->Fill(detPos.x(), detPos.y(), TMath::Abs(residuum));
//     static const int iResYZr = this->GetIndex(myResidHists2D, "residYProfZr");
//     myResidHists2D[iResYZr]->Fill(detPos.z(), rSigned, TMath::Abs(residuum));
//     static const int iSigmaYXy = this->GetIndex(myResidHists2D, "sigmaYProfXy");
//     myResidHists2D[iSigmaYXy]->Fill(detPos.x(), detPos.y(), sigma);
//     static const int iSigmaYZr = this->GetIndex(myResidHists2D, "sigmaYProfZr");
//     myResidHists2D[iSigmaYZr]->Fill(detPos.z(), rSigned, sigma);
//     if (sigma) {
//       static const int iReduResYXy = this->GetIndex(myResidHists2D, "reduResidYProfXy");
//       myResidHists2D[iReduResYXy]->Fill(detPos.x(), detPos.y(), TMath::Abs(residuum/sigma));
//       static const int iReduResYZr = this->GetIndex(myResidHists2D, "reduResidYProfZr");
//       myResidHists2D[iReduResYZr]->Fill(detPos.z(), rSigned, TMath::Abs(residuum/sigma));
//     }
//   } else {
//     static const int iResXXy = this->GetIndex(myResidHists2D, "residXProfXy");
//     myResidHists2D[iResXXy]->Fill(detPos.x(), detPos.y(), TMath::Abs(residuum));
//     static const int iResXZr = this->GetIndex(myResidHists2D, "residXProfZr");
//     myResidHists2D[iResXZr]->Fill(detPos.z(), rSigned, TMath::Abs(residuum));
//     static const int iSigmaXXy = this->GetIndex(myResidHists2D, "sigmaXProfXy");
//     myResidHists2D[iSigmaXXy]->Fill(detPos.x(), detPos.y(), sigma);
//     static const int iSigmaXZr = this->GetIndex(myResidHists2D, "sigmaXProfZr");
//     myResidHists2D[iSigmaXZr]->Fill(detPos.z(), rSigned, sigma);
//     if (sigma) {
//       static const int iReduResXXy = this->GetIndex(myResidHists2D, "reduResidXProfXy");
//       myResidHists2D[iReduResXXy]->Fill(detPos.x(), detPos.y(), TMath::Abs(residuum/sigma));
//       static const int iReduResXZr = this->GetIndex(myResidHists2D, "reduResidXProfZr");
//       myResidHists2D[iReduResXZr]->Fill(detPos.z(), rSigned, TMath::Abs(residuum/sigma));
//     }
//   }

  const LocalVector mom(tsos.localDirection()); // mom.z()==0. impossible for TSOS:
  const float phiSensToNorm = TMath::ATan(TMath::Abs((isY ? mom.y() : mom.x())/mom.z()));

  std::vector<std::vector<TH1*> > &histArrayVec = (isY ? myResidHistsVec1DY : myResidHistsVec1DX);
  std::vector<TH1*> &hitHists =                   (isY ? myResidHitHists1DY : myResidHitHists1DX);

  // call with histArrayVec[0] first: 'static' inside is referring to "subdet-less" names (X/Y irrelevant)
  this->fillResidualHists(histArrayVec[0], phiSensToNorm, residuum, sigma);
  this->fillResidualHitHists(hitHists, phiSensToNorm, residuum, sigma, nHit);

  const DetId detId(recHit->det()->geographicalId());
  if (detId.det() == DetId::Tracker) {
    //   const GeomDet::SubDetector subDetNum = recHit->det()->subDetector();
    unsigned int subDetNum = detId.subdetId();
    if (subDetNum < histArrayVec.size() && subDetNum > 0) {
      this->fillResidualHists(histArrayVec[subDetNum], phiSensToNorm, residuum, sigma);
    } else {
      if (detId!=AlignableBeamSpot::detId())
	edm::LogWarning("Alignment") << "@SUB=MillePedeMonitor::fillResiduals"
				     << "Expect subDetNum from 1 to 6, got " << subDetNum;
    }
  }
}

//____________________________________________________________________
void MillePedeMonitor::fillResidualHists(const std::vector<TH1*> &hists,
					 float phiSensToNorm, float residuum, float sigma)
{
  // Histogram indices are calculated at first call, so make sure the vector of hists at the first
  // call has the correct names (i.e. without subdet extension) and all later calls have the
  // same order of hists...
  
  static const int iRes = this->GetIndex(hists, "resid");
  hists[iRes]->Fill(residuum);
  static const int iSigma = this->GetIndex(hists, "sigma");
  hists[iSigma]->Fill(sigma);
  static const int iSigmaVsAngle = this->GetIndex(hists, "sigmaVsAngle");
  hists[iSigmaVsAngle]->Fill(phiSensToNorm, sigma);
  static const int iResidVsAngle = this->GetIndex(hists, "residVsAngle");
  hists[iResidVsAngle]->Fill(phiSensToNorm, residuum);
  static const int iReduRes = this->GetIndex(hists, "reduResid");
  static const int iReduResidVsAngle = this->GetIndex(hists, "reduResidVsAngle");
  if (sigma) {
    hists[iReduRes]->Fill(residuum/sigma);
    hists[iReduResidVsAngle]->Fill(phiSensToNorm, residuum/sigma);
  }
  static const int iAngle = this->GetIndex(hists, "angle");
  hists[iAngle]->Fill(phiSensToNorm);
  
  if (phiSensToNorm > TMath::DegToRad()*45.) {
    static const int iResGt45 = this->GetIndex(hists, "residGt45");
    hists[iResGt45]->Fill(residuum);
    static const int iSigmaGt45 = this->GetIndex(hists, "sigmaGt45");
    hists[iSigmaGt45]->Fill(sigma);
    static const int iReduResGt45 = this->GetIndex(hists, "reduResidGt45");
    if (sigma) hists[iReduResGt45]->Fill(residuum/sigma);
  } else {
    static const int iResLt45 = this->GetIndex(hists, "residLt45");
    hists[iResLt45]->Fill(residuum);
    static const int iSigmaLt45 = this->GetIndex(hists, "sigmaLt45");
    hists[iSigmaLt45]->Fill(sigma);
    static const int iReduResLt45 = this->GetIndex(hists, "reduResidLt45");
    if (sigma) hists[iReduResLt45]->Fill(residuum/sigma);
  }
}

//____________________________________________________________________
void MillePedeMonitor::fillResidualHitHists(const std::vector<TH1*> &hists, float angle,
					    float residuum, float sigma, unsigned int nHit)
{
  // Histogram indices are calculated at first call, so make sure the vector of hists at the first
  // call has the correct names (i.e. without subdet extension) and all later calls have the
  // same order of hists...
  
  static const unsigned int maxNhit = 29;  // 0...29 foreseen in initialisation...
  static int iResHit[maxNhit+1] = {-1};
  static int iSigmaHit[maxNhit+1] = {-1};
  static int iReduResHit[maxNhit+1] = {-1};
  static int iAngleHit[maxNhit+1] = {-1};
  if (iResHit[0] == -1) { // first event only...
    for (unsigned int i = 0; i <= maxNhit; ++i) {
      iResHit[i] = this->GetIndex(hists, Form("resid_%d", i));
      iSigmaHit[i] = this->GetIndex(hists, Form("sigma_%d", i));
      iReduResHit[i] = this->GetIndex(hists, Form("reduResid_%d", i));
      iAngleHit[i] = this->GetIndex(hists, Form("angle_%d", i));
    }
  }
  if (nHit > maxNhit) nHit = maxNhit; // limit of hists

  hists[iResHit[nHit]]->Fill(residuum);
  hists[iSigmaHit[nHit]]->Fill(sigma);
  if (sigma) {
    hists[iReduResHit[nHit]]->Fill(residuum/sigma);
  }
  hists[iAngleHit[nHit]]->Fill(angle);
}

//____________________________________________________________________
void MillePedeMonitor::fillFrameToFrame(const AlignableDetOrUnitPtr &aliDet, const Alignable *ali)
{
  // get derivative of higher level structure w.r.t. det
  FrameToFrameDerivative ftfd;
  const AlgebraicMatrix frameDeriv = ftfd.frameToFrameDerivative(aliDet, ali);
  //const AlgebraicMatrix frameDeriv = ftfd.frameToFrameDerivativeAtOrgRot(aliDet, ali);

  static const int iF2f = this->GetIndex(myFrame2FrameHists2D, "frame2frame");
  static const int iF2fAbs = this->GetIndex(myFrame2FrameHists2D, "frame2frameAbs");
  static int iF2fIjPhi[6][6], iF2fIjPhiLog[6][6], iF2fIjR[6][6], iF2fIjRLog[6][6];
  static bool first = true;
  if (first) {
    for (unsigned int i = 0; i < 6; ++i) {
      for (unsigned int j = 0; j < 6; ++j) {
        iF2fIjPhi[i][j] = this->GetIndex(myFrame2FrameHists2D, Form("frame2framePhi%d%d", i, j));
        iF2fIjPhiLog[i][j]=this->GetIndex(myFrame2FrameHists2D, Form("frame2framePhiLog%d%d",i,j));
        iF2fIjR[i][j] = this->GetIndex(myFrame2FrameHists2D, Form("frame2frameR%d%d", i, j));
        iF2fIjRLog[i][j]=this->GetIndex(myFrame2FrameHists2D, Form("frame2frameRLog%d%d",i,j));
      }
    }
    first = false;
  }
  
  const double phi = aliDet->globalPosition().phi(); // after misalignment...
  const double r = aliDet->globalPosition().perp(); // after misalignment...
  for (unsigned int i = 0; i < 6; ++i) {
    for (unsigned int j = 0; j < 6; ++j) {
      myFrame2FrameHists2D[iF2f]->Fill(i, j, frameDeriv[i][j]);
      myFrame2FrameHists2D[iF2fIjPhi[i][j]]->Fill(phi, frameDeriv[i][j]);
      myFrame2FrameHists2D[iF2fIjR[i][j]]->Fill(r, frameDeriv[i][j]);
      if (frameDeriv[i][j]) {
        myFrame2FrameHists2D[iF2fAbs]->Fill(i, j, TMath::Abs(frameDeriv[i][j]));
        myFrame2FrameHists2D[iF2fIjPhiLog[i][j]]->Fill(phi, TMath::Abs(frameDeriv[i][j]));
        myFrame2FrameHists2D[iF2fIjRLog[i][j]]->Fill(r, TMath::Abs(frameDeriv[i][j]));
      }
    }
  }
}

//____________________________________________________________________
void MillePedeMonitor::fillCorrelations2D(float corr, const ConstRecHitPointer &recHit)
{
  const DetId detId(recHit->det()->geographicalId());
  if (detId.det() != DetId::Tracker) return;

  if ((detId.subdetId() < 1 || detId.subdetId() > 6) &&
      detId!=AlignableBeamSpot::detId()) {
    edm::LogWarning("Alignment") << "@SUB=MillePedeMonitor::fillCorrelations2D"
                                 << "Expect subdetId from 1 to 6, got " << detId.subdetId();
    return;
  }

  myCorrHists[detId.subdetId()-1]->Fill(corr);
}

//____________________________________________________________________
void MillePedeMonitor::fillPxbSurveyHistsChi2(const float &chi2)
{
  static const int iPxbSurvChi2_lo = this->GetIndex(myPxbSurveyHists,"PxbSurvChi2_lo");
  myPxbSurveyHists[iPxbSurvChi2_lo]->Fill(chi2);
  static const int iPxbSurvChi2_md = this->GetIndex(myPxbSurveyHists,"PxbSurvChi2_md");
  myPxbSurveyHists[iPxbSurvChi2_md]->Fill(chi2);
  static const int iPxbSurvChi2_hi = this->GetIndex(myPxbSurveyHists,"PxbSurvChi2_hi");
  myPxbSurveyHists[iPxbSurvChi2_hi]->Fill(chi2);
  static const int iPxbSurvChi2prob = this->GetIndex(myPxbSurveyHists,"PxbSurvChi2prob");
  myPxbSurveyHists[iPxbSurvChi2prob]->Fill(TMath::Prob(chi2,4));
}

//____________________________________________________________________
void MillePedeMonitor::fillPxbSurveyHistsLocalPars(const float &a0, const float &a1, const float &S, const float &phi)
{
  static const int iPxbSurv_a0 = this->GetIndex(myPxbSurveyHists,"PxbSurv_a0");
  myPxbSurveyHists[iPxbSurv_a0]->Fill(a0);
  static const int iPxbSurv_a0_abs = this->GetIndex(myPxbSurveyHists,"PxbSurv_a0_abs");
  myPxbSurveyHists[iPxbSurv_a0_abs]->Fill(fabs(a0));
  static const int iPxbSurv_a1 = this->GetIndex(myPxbSurveyHists,"PxbSurv_a1");
  myPxbSurveyHists[iPxbSurv_a1]->Fill(a1);
  static const int iPxbSurv_a1_abs = this->GetIndex(myPxbSurveyHists,"PxbSurv_a1_abs");
  myPxbSurveyHists[iPxbSurv_a1_abs]->Fill(fabs(a1));
  static const int iPxbSurv_scale = this->GetIndex(myPxbSurveyHists,"PxbSurv_scale");
  myPxbSurveyHists[iPxbSurv_scale]->Fill(S);
  static const int iPxbSurv_phi = this->GetIndex(myPxbSurveyHists,"PxbSurv_phi");
  myPxbSurveyHists[iPxbSurv_phi]->Fill(phi);
}


