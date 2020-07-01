///=== This is the global Linear Regression for 4 helix parameters track fit algorithm.

///=== Written by: Davide Cieri

#include "L1Trigger/TrackFindingTMTT/interface/SimpleLR4.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <vector>
#include <set>
#include <algorithm>
#include <limits>

using namespace std;

namespace tmtt {

  SimpleLR4::SimpleLR4(const Settings* settings) : TrackFitGeneric(settings) {
    // Initialize digitization parameters
    phiMult_ = pow(2., settings_->phiSBits()) / settings_->phiSRange();
    rTMult_ = pow(2., settings_->rtBits()) / settings_->rtRange();
    zMult_ = pow(2., settings_->zBits()) / settings_->zRange();
    z0Mult_ = pow(2., settings_->slr_z0Bits()) / settings_->slr_z0Range();
    phiTMult_ = pow(2., settings_->slr_phi0Bits()) / settings_->slr_phi0Range();

    qOverPtMult_ = pow(2., settings_->slr_oneOver2rBits()) / settings_->slr_oneOver2rRange();
    tanLambdaMult_ = pow(2., settings_->slr_tanlambdaBits()) / settings_->slr_tanlambdaRange();
    chi2Mult_ = pow(2., settings_->slr_chisquaredBits()) / settings_->slr_chisquaredRange();

    numeratorPtMult_ = rTMult_ * phiMult_;
    numeratorPhiMult_ = rTMult_ * rTMult_ * phiMult_;
    numeratorZ0Mult_ = rTMult_ * rTMult_ * z0Mult_;
    numeratorLambdaMult_ = rTMult_ * z0Mult_;
    denominatorMult_ = rTMult_ * rTMult_;
    resMult_ = rTMult_ * qOverPtMult_;

    dividerBitsHelix_ = settings_->dividerBitsHelix();
    dividerBitsHelixZ_ = settings_->dividerBitsHelixZ();
    shiftingBitsDenRPhi_ = settings_->ShiftingBitsDenRPhi();
    shiftingBitsDenRZ_ = settings_->ShiftingBitsDenRZ();
    shiftingBitsPhi_ = settings_->ShiftingBitsPhi();
    shiftingBitsz0_ = settings_->ShiftingBitsZ0();
    shiftingBitsPt_ = settings_->ShiftingBitsPt();
    shiftingBitsLambda_ = settings_->ShiftingBitsLambda();
    digitize_ = settings_->digitizeSLR() and settings_->enableDigitize();

    phiSectorWidth_ = 2. * M_PI / float(settings_->numPhiSectors());
    phiNonantWidth_ = 2. * M_PI / float(settings_->numPhiNonants());

    chi2cut_ = settings_->slr_chi2cut();
    chosenRofPhi_ = settings_->chosenRofPhi();
    if (digitize_)
      chosenRofPhi_ = floor(chosenRofPhi_ * rTMult_) / rTMult_;

    debug_ = false;  // Enable debug printout.
  };

  static bool pair_compare(std::pair<const Stub*, float> a, std::pair<const Stub*, float> b) {
    return (a.second < b.second);
  }

  L1fittedTrack SimpleLR4::fit(const L1track3D& l1track3D) {
    if (debug_)
      PrintL1trk() << "=============== FITTING SimpleLR TRACK ====================";

    minStubLayersRed_ = Utility::numLayerCut(Utility::AlgoStep::FIT,
                                             settings_,
                                             l1track3D.iPhiSec(),
                                             l1track3D.iEtaReg(),
                                             std::abs(l1track3D.qOverPt()),
                                             l1track3D.eta());

    invPtToDPhi_ = -settings_->invPtToDphi();

    double phiCentreSec0 = -0.5 * phiNonantWidth_ + 0.5 * phiSectorWidth_;
    phiSectorCentre_ = phiSectorWidth_ * double(l1track3D.iPhiSec()) - phiCentreSec0;

    if (digitize_)
      phiSectorCentre_ = floor(phiSectorCentre_ * phiTMult_) / phiTMult_;

    // Inizialise track fit parameters
    double qOverPt = 0.;
    double phiT = 0.;
    double phi0 = 0.;
    double z0 = 0.;
    double zT = 0.;
    double tanLambda = 0.;

    // Inizialise Sums
    double SumRPhi = 0.;
    double SumR = 0.;
    double SumPhi = 0.;
    double SumR2 = 0.;
    double SumRZ = 0.;
    double SumZ = 0.;

    unsigned int numStubs = 0;
    // Calc helix parameters on Rphi Plane (STEP 1)
    // This loop calculates the sums needed to calculate the numerators and the denominator to compute the helix parameters in the R-Phi plane (q/pT, phiT)
    for (Stub* stub : l1track3D.stubs()) {
      numStubs++;

      if (digitize_) {
        const DigitalStub* digiStub = stub->digitalStub();

        SumRPhi = SumRPhi + digiStub->rt_SF_TF() * digiStub->phiS();
        SumR = SumR + digiStub->rt_SF_TF();
        SumPhi = SumPhi + digiStub->phiS();
        SumR2 = SumR2 + digiStub->rt_SF_TF() * digiStub->rt_SF_TF();
        if (debug_)
          PrintL1trk() << "Input stub (digi): phiS " << digiStub->iDigi_PhiS() << " rT " << digiStub->iDigi_Rt()
                       << " z " << digiStub->iDigi_Z();
      } else {
        float phi = 0;
        if (l1track3D.iPhiSec() == 0 and stub->phi() > 0) {
          phi = stub->phi() - 2 * M_PI;
        } else if (l1track3D.iPhiSec() == settings_->numPhiSectors() and stub->phi() < 0) {
          phi = stub->phi() + 2 * M_PI;
        } else {
          phi = stub->phi();
        }
        SumRPhi = SumRPhi + stub->r() * phi;
        SumR = SumR + stub->r();
        SumPhi = SumPhi + phi;
        SumR2 = SumR2 + stub->r() * stub->r();
        if (debug_)
          PrintL1trk() << "InputStub (float): phi " << phi << " r " << stub->r() << " z " << stub->z();
      }
    }

    double numeratorPt, digiNumeratorPt;
    double denominator, digiDenominator;
    double numeratorPhi, digiNumeratorPhi;
    double reciprocal, digiReciprocal;
    double numeratorZ0;
    double numeratorLambda;

    digiNumeratorPt = (numStubs * SumRPhi - SumR * SumPhi);
    digiDenominator = (numStubs * SumR2 - SumR * SumR);
    digiNumeratorPhi = (SumR2 * SumPhi - SumR * SumRPhi);

    if (!digitize_) {
      qOverPt = (numStubs * SumRPhi - SumR * SumPhi) / (numStubs * SumR2 - SumR * SumR);
      phi0 = (SumR2 * SumPhi - SumR * SumRPhi) / (numStubs * SumR2 - SumR * SumR);
    } else {
      digiNumeratorPt /= pow(2., shiftingBitsPt_);
      digiNumeratorPt = floor(digiNumeratorPt * numeratorPtMult_);
      numeratorPt = digiNumeratorPt / numeratorPtMult_;

      digiNumeratorPhi /= pow(2., shiftingBitsPhi_);
      digiNumeratorPhi = floor(digiNumeratorPhi * numeratorPhiMult_);
      numeratorPhi = digiNumeratorPhi / numeratorPhiMult_;

      digiDenominator /= pow(2., shiftingBitsDenRPhi_);
      digiDenominator = (floor(digiDenominator * denominatorMult_) + 0.5);
      denominator = digiDenominator / denominatorMult_;
      digiReciprocal = (pow(2., dividerBitsHelix_) - 1) / (denominator);  // To be moved
      digiReciprocal = floor(digiReciprocal / denominatorMult_);
      reciprocal = digiReciprocal * denominatorMult_;

      qOverPt = numeratorPt * reciprocal / pow(2., dividerBitsHelix_ + shiftingBitsDenRPhi_ - shiftingBitsPt_);
      phiT = numeratorPhi * reciprocal / pow(2., dividerBitsHelix_ + shiftingBitsDenRPhi_ - shiftingBitsPhi_);

      qOverPt = floor(qOverPt * qOverPtMult_) / (qOverPtMult_);
      phiT = floor(phiT * phiTMult_) / phiTMult_;
    }

    if (debug_) {
      if (digitize_) {
        PrintL1trk() << setw(10) << "Input helix (digi): qOverPt = " << qOverPt << " (" << floor(qOverPt * qOverPtMult_)
                     << "), phiT = " << phiT << " (" << floor(phiT * phiTMult_) << ") ";
      } else {
        PrintL1trk() << "Input Helix (float): qOverPt = " << qOverPt << " phi0 " << phi0;
      }
    }

    // ================== RESIDUAL CALCULATION ON RPHI ========================
    std::vector<std::pair<Stub*, double> > vRes;
    unsigned int psStubs = 0;
    for (Stub* stub : l1track3D.stubs()) {
      if (stub->psModule())
        psStubs++;
      double ResPhi;

      if (digitize_) {
        const DigitalStub* digiStub = stub->digitalStub();

        ResPhi =
            digiStub->iDigi_PhiS() * pow(2., shiftingBitsDenRPhi_ - shiftingBitsPt_) -
            floor(phiT * phiTMult_) *
                pow(2., shiftingBitsDenRPhi_ - shiftingBitsPt_ - settings_->slr_phi0Bits() + settings_->phiSBits()) -
            floor(qOverPt * qOverPtMult_) * digiStub->iDigi_Rt();

        ResPhi = floor(ResPhi) / resMult_;
      }

      else {
        ResPhi = reco::deltaPhi(stub->phi(), phi0 + qOverPt * stub->r());
      }

      double Res = std::abs(ResPhi);

      std::pair<Stub*, double> ResStubPair(stub, Res);
      vRes.push_back(ResStubPair);
      if (debug_) {
        if (stub->assocTP() != nullptr)
          PrintL1trk() << " Stub rphi residual " << Res << " TP " << stub->assocTP()->index();
        else
          PrintL1trk() << " Stub rphi residual " << Res << " TP nullptr";
      }
    }

    double largestResidual = 9999.;
    // Find largest residuals
    while (vRes.size() > minStubLayersRed_ and largestResidual > settings_->ResidualCut()) {
      std::vector<std::pair<Stub*, double> >::iterator maxResIt = max_element(vRes.begin(), vRes.end(), pair_compare);
      largestResidual = (*maxResIt).second;
      if (debug_)
        PrintL1trk() << "Largest Residual " << largestResidual;

      if (largestResidual > settings_->ResidualCut()) {
        if ((*maxResIt).first->psModule()) {
          if (psStubs > 2) {
            if (debug_)
              PrintL1trk() << "removing PS residual " << (*maxResIt).second;
            vRes.erase(maxResIt);
            psStubs--;
          } else {
            if (debug_)
              PrintL1trk() << "residual " << (*maxResIt).second << " set to -1. ";
            (*maxResIt).second = -1.;
          }
        } else {
          vRes.erase(maxResIt);
          if (debug_)
            PrintL1trk() << "removing residual " << (*maxResIt).second;
        }
      }
    }

    std::vector<Stub*> fitStubs;
    fitStubs.reserve(vRes.size());

        for (std::pair<Stub*, double> ResStubPair : vRes) {
      fitStubs.push_back(ResStubPair.first);
    }

    phiT = 0.;
    zT = 0.;

    SumRPhi = 0.;
    SumR = 0.;
    SumPhi = 0.;
    SumR2 = 0.;
    SumRZ = 0.;
    SumZ = 0.;
    double SumR_ps = 0.;
    double SumR2_ps = 0.;

    numStubs = 0;
    psStubs = 0;

    for (const Stub* stub : fitStubs) {
      if (stub->psModule())
        psStubs++;

      numStubs++;
      if (digitize_) {
        const DigitalStub* digiStub = stub->digitalStub();
        SumRPhi += digiStub->rt_SF_TF() * digiStub->phiS();
        SumR += digiStub->rt_SF_TF();
        SumPhi += digiStub->phiS();
        SumR2 += digiStub->rt_SF_TF() * digiStub->rt_SF_TF();
        if (stub->psModule()) {
          SumRZ += digiStub->rt_SF_TF() * digiStub->z();
          SumZ += digiStub->z();
          SumR_ps += digiStub->rt_SF_TF();
          SumR2_ps += digiStub->rt_SF_TF() * digiStub->rt_SF_TF();
        }
        if (debug_) {
          PrintL1trk() << "phiS " << digiStub->iDigi_PhiS() << " rT " << digiStub->iDigi_Rt() << " z "
                       << digiStub->iDigi_Z();
        }
      } else {
        float phi = 0;
        if (l1track3D.iPhiSec() == 0 and stub->phi() > 0) {
          phi = stub->phi() - 2 * M_PI;
        } else if (l1track3D.iPhiSec() == settings_->numPhiSectors() and stub->phi() < 0) {
          phi = stub->phi() + 2 * M_PI;
        } else {
          phi = stub->phi();
        }

        SumRPhi += stub->r() * phi;
        SumR += stub->r();
        SumPhi += phi;
        SumR2 += stub->r() * stub->r();
        if (stub->psModule()) {
          SumRZ += stub->r() * stub->z();
          SumZ += stub->z();
          SumR_ps += stub->r();
          SumR2_ps += stub->r() * stub->r();
        }
        if (debug_)
          PrintL1trk() << "phi " << phi << " r " << stub->r() << " z " << stub->z();
      }
    }

    numeratorZ0 = (SumR2_ps * SumZ - SumR_ps * SumRZ);
    numeratorLambda = (psStubs * SumRZ - SumR_ps * SumZ);
    numeratorPt = (numStubs * SumRPhi - SumR * SumPhi);
    denominator = (numStubs * SumR2 - SumR * SumR);
    double denominatorZ = (psStubs * SumR2_ps - SumR_ps * SumR_ps);
    numeratorPhi = (SumR2 * SumPhi - SumR * SumRPhi);
    double reciprocalZ;
    if (!digitize_) {
      z0 = numeratorZ0 / denominatorZ;
      tanLambda = numeratorLambda / denominatorZ;
      qOverPt = (numStubs * SumRPhi - SumR * SumPhi) / (numStubs * SumR2 - SumR * SumR);
      phi0 = (SumR2 * SumPhi - SumR * SumRPhi) / (numStubs * SumR2 - SumR * SumR);
    } else {
      numeratorPt /= pow(2., shiftingBitsPt_);
      numeratorPt = floor(numeratorPt * numeratorPtMult_) / numeratorPtMult_;

      numeratorPhi /= pow(2., shiftingBitsPhi_);
      numeratorPhi = floor(numeratorPhi * numeratorPhiMult_) / numeratorPhiMult_;

      numeratorLambda /= pow(2., shiftingBitsLambda_);
      numeratorLambda = floor(numeratorLambda * numeratorLambdaMult_) / numeratorLambdaMult_;

      numeratorZ0 /= pow(2., shiftingBitsz0_);
      numeratorZ0 = floor(numeratorZ0 * numeratorZ0Mult_) / numeratorZ0Mult_;

      denominator /= pow(2., shiftingBitsDenRPhi_);
      denominator = (floor(denominator * denominatorMult_) + 0.5) / denominatorMult_;
      reciprocal = (pow(2., dividerBitsHelix_) - 1) / (denominator);
      reciprocal = floor(reciprocal / denominatorMult_) * denominatorMult_;

      denominatorZ /= pow(2., shiftingBitsDenRZ_);
      denominatorZ = (floor(denominatorZ * denominatorMult_) + 0.5) / denominatorMult_;
      reciprocalZ = (pow(2., dividerBitsHelixZ_) - 1) / (denominatorZ);
      reciprocalZ = floor(reciprocalZ / denominatorMult_) * denominatorMult_;

      qOverPt = numeratorPt * reciprocal / pow(2., dividerBitsHelix_ + shiftingBitsDenRPhi_ - shiftingBitsPt_);
      phiT = numeratorPhi * reciprocal / pow(2., dividerBitsHelix_ + shiftingBitsDenRPhi_ - shiftingBitsPhi_);

      tanLambda =
          numeratorLambda * reciprocalZ / pow(2., dividerBitsHelixZ_ + shiftingBitsDenRZ_ - shiftingBitsLambda_);
      zT = numeratorZ0 * reciprocalZ / pow(2., dividerBitsHelixZ_ + shiftingBitsDenRZ_ - shiftingBitsz0_);

      phi0 = phiSectorCentre_ + phiT - qOverPt * settings_->chosenRofPhi();
      z0 = zT - tanLambda * settings_->chosenRofPhi();

      qOverPt = floor(qOverPt * qOverPtMult_) / qOverPtMult_;
      phiT = floor(phiT * phiTMult_) / phiTMult_;
    }

    if (debug_ and digitize_) {
      PrintL1trk() << "HT mbin " << int(l1track3D.cellLocationHT().first) - 16 << " cbin "
                   << int(l1track3D.cellLocationHT().second) - 32 << " iPhi " << l1track3D.iPhiSec() << " iEta "
                   << l1track3D.iEtaReg();
      PrintL1trk() << "Second Helix variables: numeratorPt = " << numeratorPt << ", numeratorPhi = " << numeratorPhi
                   << ", numeratorZ0 = " << numeratorZ0 << " numeratorLambda = " << numeratorLambda
                   << " denominator =  " << denominator << " reciprocal = " << reciprocal
                   << " denominatorZ =  " << denominatorZ << " reciprocalZ = " << reciprocalZ;
      PrintL1trk() << setw(10) << "Final Helix parameters: qOverPt = " << qOverPt << " ("
                   << floor(qOverPt * qOverPtMult_) << "), phiT = " << phiT << " (" << floor(phiT * phiTMult_)
                   << "), zT = " << zT << " (" << floor(zT * z0Mult_) << "), tanLambda = " << tanLambda << " ("
                   << floor(tanLambda * tanLambdaMult_) << ")"
                   << " z0 " << z0;
    } else if (debug_) {
      PrintL1trk() << setw(10) << "Final Helix parameters: qOverPt = " << qOverPt << ", phi0 = " << phi0
                   << ", z0 = " << z0 << ", tanLambda = " << tanLambda;
    }

    double chi2_phi = 0.;
    double chi2_z = 0.;

    for (const Stub* stub : fitStubs) {
      double ResPhi = 0.;
      double ResZ = 0.;
      if (digitize_) {
        const DigitalStub* digiStub = stub->digitalStub();
        ResPhi = digiStub->phiS() - phiT - qOverPt * digiStub->rt_SF_TF();
        ResZ = digiStub->z() - zT - tanLambda * digiStub->rt_SF_TF();
      } else {
        ResPhi = reco::deltaPhi(stub->phi(), phi0 + qOverPt * stub->r());
        ResZ = stub->z() - z0 - tanLambda * stub->r();
      }

      double RPhiSigma = 0.0002;
      float RZSigma = stub->sigmaZ() + std::abs(tanLambda) * stub->sigmaR();

      if (not stub->barrel())
        RPhiSigma = 0.0004;

      if (digitize_) {
        RPhiSigma = floor(RPhiSigma * phiMult_) / phiMult_;
      }

      ResPhi /= RPhiSigma;
      ResZ /= RZSigma;

      chi2_phi += std::abs(ResPhi * ResPhi);
      chi2_z += std::abs(ResZ * ResZ);
      if (debug_) {
        PrintL1trk() << "Stub ResPhi " << ResPhi * RPhiSigma << " ResSigma " << RPhiSigma << " Res " << ResPhi
                     << " chi2 " << chi2_phi;
        PrintL1trk() << "Stub ResZ " << ResZ * RZSigma << " ResSigma " << RZSigma << " Res " << ResZ << " chi2 "
                     << chi2_z;
      }
    }
    qOverPt /= invPtToDPhi_;

    bool accepted = false;

    //double chi2 = chi2_phi;  // Ignore r-z residuals due to poor 2S resolution?
    double chi2 = chi2_phi + chi2_z;
    if (digitize_)
      chi2 = floor(chi2 * chi2Mult_) / chi2Mult_;

    constexpr unsigned int nHelixPar = 4;
    float dof = 2 * fitStubs.size() - nHelixPar;
    float chi2dof = chi2 / dof;
    if (chi2 < chi2cut_)
      accepted = true;

    if (debug_)
      PrintL1trk() << "qOverPt " << qOverPt << " phiT " << phiT;

    // This condition can only happen if cfg param TrackFitCheat = True.
    if (fitStubs.size() < minStubLayersRed_)
      accepted = false;

    // Kinematic cuts -- NOT YET IN FIRMWARE!!!
    constexpr float tolerance = 0.1;
    if (std::abs(qOverPt) > 1. / (settings_->houghMinPt() - tolerance))
      accepted = false;
    if (std::abs(z0) > 20.)
      accepted = false;

    if (accepted) {
      // Create the L1fittedTrack object
      const unsigned int hitPattern = 0;  // FIX: Needs setting
      L1fittedTrack fitTrk(
          settings_, &l1track3D, fitStubs, hitPattern, qOverPt, 0., phi0, z0, tanLambda, chi2_phi, chi2_z, nHelixPar);

      if (settings_->enableDigitize())
        fitTrk.digitizeTrack("SimpleLR4");

      if (debug_ and digitize_) {
        PrintL1trk() << "Digitized parameters ";
        PrintL1trk() << "HT mbin " << int(l1track3D.cellLocationHT().first) - 16 << " cbin "
                     << int(l1track3D.cellLocationHT().second) - 32 << " iPhi " << l1track3D.iPhiSec() << " iEta "
                     << l1track3D.iEtaReg();
        PrintL1trk() << setw(10) << "First Helix parameters: qOverPt = " << fitTrk.qOverPt() << " oneOver2r "
                     << fitTrk.digitaltrack()->oneOver2r() << " ("
                     << floor(fitTrk.digitaltrack()->oneOver2r() * qOverPtMult_)
                     << "), phi0 = " << fitTrk.digitaltrack()->phi0() << " (" << fitTrk.digitaltrack()->iDigi_phi0rel()
                     << "), zT = " << zT << " (" << floor(zT * z0Mult_) << "), tanLambda = " << tanLambda << " ("
                     << floor(tanLambda * tanLambdaMult_) << ")";
      }

      if (debug_) {
        PrintL1trk() << "FitTrack helix parameters " << int(fitTrk.cellLocationFit().first) - 16 << ", "
                     << int(fitTrk.cellLocationFit().second) - 32 << " HT parameters "
                     << int(fitTrk.cellLocationHT().first) - 16 << ", " << int(fitTrk.cellLocationHT().second) - 32;

        if (fitTrk.matchedTP() != nullptr) {
          PrintL1trk() << "True track: chi2/ndf " << chi2dof;
          PrintL1trk() << "TP qOverPt " << fitTrk.matchedTP()->qOverPt() << " phi0 " << fitTrk.matchedTP()->phi0();
          if (!accepted)
            PrintL1trk() << "Track rejected " << chi2 << " chi2/ndof " << chi2dof;
        } else {
          PrintL1trk() << "Fake track!!! " << chi2 << " chi2/ndof " << chi2dof;
        }
        PrintL1trk() << "layers in track " << fitTrk.numLayers();
      }

      return fitTrk;

    } else {
      L1fittedTrack rejectedTrk;
      return rejectedTrk;
    }
  }

}  // namespace tmtt
