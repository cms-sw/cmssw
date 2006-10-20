/**
 * \file MillePedeMonitor.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.11 $
 *  $Date$
 *  (last update by $Author$)
 */

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeMonitor.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>

#include <TH1.h>
#include <TH2.h>
#include <TProfile2D.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TArrayF.h>

//__________________________________________________________________
MillePedeMonitor::MillePedeMonitor(const char *rootFileName)
  : myRootDir(0), myDeleteDir(false)
{
  myRootDir = TFile::Open(rootFileName, "recreate");
  myDeleteDir = true;

  this->init(myRootDir);
}

//__________________________________________________________________
MillePedeMonitor::MillePedeMonitor(TDirectory *rootDir) 
  : myRootDir(0), myDeleteDir(false)
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
//   directory->cd();
  TDirectory *dirTracks = directory->mkdir("trackHists", "input tracks");
  if (dirTracks) dirTracks->cd(); // else...

  const int kNumBins = 20;
  double binsPt[kNumBins+1] = {0.}; // fully initialised with 0.
  if (!this->equidistLogBins(binsPt, kNumBins, 0.8, 100.)) {
//     cerr << "MillePedeMonitor::init: problem with log bins" << endl;
  }

  myTrackHists1D.push_back(new TH1F("ptTrackLogBins",  "p_{t}(track)", kNumBins, binsPt));

  myTrackHists1D.push_back(new TH1F("ptTrack",  "p_{t}(track)",
				    kNumBins, binsPt[0], binsPt[kNumBins]));
  myTrackHists1D.push_back(new TH1F("pTrack",  "p(track)",
				    kNumBins, binsPt[0], 1.3*binsPt[kNumBins]));
  myTrackHists1D.push_back(new TH1F("etaTrack", "#eta(track)", 26, -2.6, 2.6));
  myTrackHists1D.push_back(new TH1F("phiTrack", "#phi(track)", 15, -TMath::Pi(), TMath::Pi()));

  myTrackHists1D.push_back(new TH1F("nHitTrack", "N_{hit}(track)", 18, 5., 23.));
  myTrackHists1D.push_back(new TH1F("nHitInvalidTrack", "N_{hit, invalid}(track)", 5, 0., 5.));
  myTrackHists1D.push_back(new TH1F("r1Track", "r(1st hit)", 20, 0., 20.));
  myTrackHists1D.push_back(new TH1F("z1Track", "z(1st hit)", 20, -50, +50));
  myTrackHists1D.push_back(new TH1F("rLastTrack", "r(last hit)", 20, 20., 120.));
  myTrackHists1D.push_back(new TH1F("zLastTrack", "z(last hit)", 30, -300., +300.));
  myTrackHists1D.push_back(new TH1F("chi2PerNdf", "#chi^{2}/ndf", 50, 0., 50.));

  myTrackHists1D.push_back(new TH1F("impParZ", "impact parameter in z", 20, -20., 20.));
  myTrackHists1D.push_back(new TH1F("impParErrZ", "error of impact parameter in z",
				    40, 0., 0.06));  
  myTrackHists1D.push_back(new TH1F("impParRphi", "impact parameter in r#phi", 51, -0.05, .05));
  myTrackHists1D.push_back(new TH1F("impParErrRphi", "error of impact parameter in r#phi",
				    50, 0., 0.01));

// ReferenceTrajectory

  TDirectory *dirTraject = directory->mkdir("refTrajectoryHists", "ReferenceTrajectory's");
  if (dirTraject) dirTraject->cd();

  myTrajectoryHists1D.push_back(new TH1F("validRefTraj", "validity of ReferenceTrajectory",
					 2, 0., 2.));

  myTrajectoryHists2D.push_back(new TProfile2D("profCorr",
					       "mean of |#rho|, #rho#neq0;hit x/y;hit x/y;",
					       34, 0., 17., 34, 0., 17.));
  myTrajectoryHists2D.push_back
    (new TProfile2D("profCorrOffXy", "mean of |#rho|, #rho#neq0, no xy_{hit};hit x/y;hit x/y;",
		    34, 0., 17., 34, 0., 17.));

  myTrajectoryHists2D.push_back(new TH2F("hitCorrOffBlock",
					 "hit correlations: off-block-diagonals;N(hit);#rho",
					 34, 0., 17., 81, -.06, .06));
  TArrayD logBins(102);
  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-11, .1);
  myTrajectoryHists2D.push_back(new TH2F("hitCorrOffBlockLog",
					 "hit correlations: off-block-diagonals;N(hit);|#rho|",
					 34, 0., 17., logBins.GetSize()-1, logBins.GetArray()));

  myTrajectoryHists2D.push_back(new TH2F("hitCorrXy", "hit correlations: xy;N(hit);#rho",
					 17, 0., 17., 81, -.5, .5));
  myTrajectoryHists2D.push_back
    (new TH2F("hitCorrXyValid", "hit correlations: xy, 2D-det.;N(hit);#rho",
	      17, 0., 17., 81, -.02, .02));
  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-10, .5);
  myTrajectoryHists2D.push_back(new TH2F("hitCorrXyLog", "hit correlations: xy;N(hit);|#rho|",
					 17, 0., 17., logBins.GetSize()-1, logBins.GetArray()));
  myTrajectoryHists2D.push_back
    (new TH2F("hitCorrXyLogValid", "hit correlations: xy, 2D-det.;N(hit);|#rho|",
	      17, 0., 17., logBins.GetSize()-1, logBins.GetArray()));


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
					 17, 0., 17., 101, -6., 6.));
  myTrajectoryHists2D.push_back(new TH2F("measLocYvsHit", "local y measurements, 2D-det.;hit;y",
					 17, 0., 17., 101, -10., 10.));
  myTrajectoryHists2D.push_back(new TH2F("trajLocXvsHit", "local x trajectory;hit;x",
					 17, 0., 17., 101, -6., 6.));
  myTrajectoryHists2D.push_back(new TH2F("trajLocYvsHit", "local y trajectory, 2D-det.;hit;y",
					 17, 0., 17.,  101, -10., 10.));

  myTrajectoryHists2D.push_back(new TH2F("residLocXvsHit", "local x residual;hit;#Deltax",
					 17, 0., 17., 101, -.75, .75));
  myTrajectoryHists2D.push_back(new TH2F("residLocYvsHit", "local y residual, 2D-det.;hit;#Deltay",
					 17, 0., 17., 101, -1., 1.));
  myTrajectoryHists2D.push_back
    (new TH2F("reduResidLocXvsHit", "local x reduced residual;hit;#Deltax/#sigma",
	      17, 0., 17., 101, -20., 20.));
  myTrajectoryHists2D.push_back
    (new TH2F("reduResidLocYvsHit", "local y reduced residual, 2D-det.;hit;#Deltay/#sigma",
	      17, 0., 17., 101, -20., 20.));


  myTrajectoryHists2D.push_back(new TProfile2D("profDerivatives",
					       "mean derivatives;hit x/y;parameter;",
					       34, 0., 17., 10, 0., 10.));

  myTrajectoryHists2D.push_back
    (new TH2F("derivatives", "derivative;parameter;#partial(x/y)_{local}/#partial(param)",
	      10, 0., 10., 101, -20., 20.));
  this->equidistLogBins(logBins.GetArray(), logBins.GetSize()-1, 1.E-12, 100.);
  myTrajectoryHists2D.push_back
    (new TH2F("derivativesLog", "|derivative|;parameter;|#partial(x/y)_{local}/#partial(param)|",
	      10, 0., 10., logBins.GetSize()-1, logBins.GetArray()));

  directory->cd();

  myDetLabelBitHist = new TH1F("globalDetLabelBits", "bits of detId", 32,0.,32.);
  myDetLabelHist = new TH1F("globalDetLabels", "detLabels", 1000, 0., 1.);
  myDetLabelHist->SetBit(TH1::kCanRebin);

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
void MillePedeMonitor::fillTrack(const reco::Track *track, const Trajectory *traj)
{
  if (!track) return;

  const reco::TrackBase::Vector p(track->momentum());

  static const int iPtLog = this->GetIndex(myTrackHists1D, "ptTrackLogBins");
  myTrackHists1D[iPtLog]->Fill(track->pt());
  static const int iPt = this->GetIndex(myTrackHists1D, "ptTrack");
  myTrackHists1D[iPt]->Fill(track->pt());
  static const int iP = this->GetIndex(myTrackHists1D, "pTrack");
  myTrackHists1D[iP]->Fill(p.R());
  static const int iEta = this->GetIndex(myTrackHists1D, "etaTrack");
  myTrackHists1D[iEta]->Fill(p.Eta());
  static const int iPhi = this->GetIndex(myTrackHists1D, "phiTrack");
  myTrackHists1D[iPhi]->Fill(p.Phi());

  static const int iNhit = this->GetIndex(myTrackHists1D, "nHitTrack");
  static const int iNhitInvalid = this->GetIndex(myTrackHists1D, "nHitInvalidTrack");
  static const int iR1 = this->GetIndex(myTrackHists1D, "r1Track");
  static const int iZ1 = this->GetIndex(myTrackHists1D, "z1Track");
  static const int iRlast = this->GetIndex(myTrackHists1D, "rLastTrack");
  static const int iZlast = this->GetIndex(myTrackHists1D, "zLastTrack");
  if (traj) {
    myTrackHists1D[iNhit]->Fill(traj->foundHits());
    myTrackHists1D[iNhitInvalid]->Fill(traj->lostHits());
    const TrajectoryMeasurement inner(traj->direction() == alongMomentum ? 
				      traj->firstMeasurement() : traj->lastMeasurement());
    const GlobalPoint firstPoint(inner.recHit()->globalPosition());
    myTrackHists1D[iR1]->Fill(firstPoint.perp());
    myTrackHists1D[iZ1]->Fill(firstPoint.z());

    const TrajectoryMeasurement outer(traj->direction() == oppositeToMomentum ? 
				      traj->firstMeasurement() : traj->lastMeasurement());
    const GlobalPoint lastPoint(outer.recHit()->globalPosition());
    myTrackHists1D[iRlast]->Fill(lastPoint.perp());
    myTrackHists1D[iZlast]->Fill(lastPoint.z());
  } else {
    myTrackHists1D[iNhit]->Fill(track->numberOfValidHits());
    myTrackHists1D[iNhitInvalid]->Fill(track->numberOfLostHits());
//    if (track->innerOk()) {
//   const reco::TrackBase::Point firstPoint(track->innerPosition());
//     myHistFirstR->Fill(firstPoint.Rho());
//     myHistFirstZ->Fill(firstPoint.Z());
//    }
//    if (track->outerOk()) {
//   const reco::TrackBase::Point lastPoint(track->outerPosition());
//     myHistLastR->Fill(lastPoint.Rho());
//     myHistLastZ->Fill(lastPoint.Z());
//    }
  }

  static const int iChi2Ndf = this->GetIndex(myTrackHists1D, "chi2PerNdf");
  myTrackHists1D[iChi2Ndf]->Fill(track->normalizedChi2());

  static const int iImpZ = this->GetIndex(myTrackHists1D, "impParZ");
  myTrackHists1D[iImpZ]->Fill(track->dz());
  static const int iImpZerr = this->GetIndex(myTrackHists1D, "impParErrZ");
  myTrackHists1D[iImpZerr]->Fill(track->dzError());


  static const int iImpRphi = this->GetIndex(myTrackHists1D, "impParRphi");
  myTrackHists1D[iImpRphi]->Fill(track->d0());
  static const int iImpRphiErr = this->GetIndex(myTrackHists1D, "impParErrRphi");
  myTrackHists1D[iImpRphiErr]->Fill(track->d0Error());

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

  for (int iRow = 0; iRow < covMeasLoc.num_row(); ++iRow) {
    const double residuum = measurementsLoc[iRow] - trajectoryLoc[iRow];
    const double covMeasLocRow = covMeasLoc[iRow][iRow];
    const bool is2DhitRow = (!recHits[iRow/2]->detUnit() 
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
    for (int iCol = iRow+1; iCol < covMeasLoc.num_col(); ++iCol) {
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
    } // end loop on columns of covarianvce
    
    // derivatives
    for (int iCol = 0; iCol < derivatives.num_col(); ++iCol) {
      static const int iProfDerivatives = this->GetIndex(myTrajectoryHists2D, "profDerivatives");
      myTrajectoryHists2D[iProfDerivatives]->Fill(nHitRow, iCol, derivatives[iRow][iCol]);
      static const int iDerivatives = this->GetIndex(myTrajectoryHists2D, "derivatives");
      myTrajectoryHists2D[iDerivatives]->Fill(iCol, derivatives[iRow][iCol]);
      static const int iDerivativesLog = this->GetIndex(myTrajectoryHists2D, "derivativesLog");
      myTrajectoryHists2D[iDerivativesLog]->Fill(iCol, TMath::Abs(derivatives[iRow][iCol]));
    }
  } // end loop on rows of covarianvce

}

//____________________________________________________________________
void MillePedeMonitor::fillDetLabel(unsigned int globalDetLabel)
{

  for (unsigned int i = 0; i < 32; ++i) {
    if (globalDetLabel & (1 << i)) myDetLabelBitHist->Fill(i);
  }

  myDetLabelHist->Fill(globalDetLabel);


}



