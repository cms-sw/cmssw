// TopRecoilHook.h is a part of the PYTHIA event generator.
// Copyright (C) 2020 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Includes a user hook that corrects emission in top decay for dipole
// from gluon to W, to instead be from gluon to top.

// Important: the top mass shift analysis encoded here is very primitive,
// does not perform well at all, and should not be taken seriously.
// The important part is that you see how the different scenarios
// should be set up to operate as intended.

#include "Pythia8/Pythia.h"
namespace Pythia8 {

  //==========================================================================

  // Write own derived UserHooks class for modified emission in top decay.

  class TopRecoilHook : public UserHooks {
  public:
    // Constructor.
    //  doTopRecoil : eikonal correction in GW dipole on/off when no MEC applied.
    //  useOldDipole  : in GW dipole, use partons before or after branching.
    //  doList        : diagnostic output; set false for production runs.
    TopRecoilHook(bool doTopRecoilIn = true, bool useOldDipoleIn = false, bool doListIn = false) {
      doTopRecoil = doTopRecoilIn;
      useOldDipole = useOldDipoleIn;
      doList = doListIn;
      // Constructor also creates some histograms for analysis inside User Hook.
      wtCorr = new Hist("corrective weight", 100, 0., 2.);
    }

    // Destructor prints histogram.
     ~TopRecoilHook() override {
      if (doTopRecoil)
        ;
      delete wtCorr;
    }

    // Initialise. Only use hook for simple showers with recoilToColoured = off.
    bool initAfterBeams() override {
      // Switch off if recoilToColoured = on.
      bool recoilToColoured = settingsPtr->flag("TimeShower:recoilToColoured");
      if (recoilToColoured)
        doTopRecoil = false;
      // Flag if W mass term is already accounted for (true) or not (false).
      recoilDeadCone = settingsPtr->flag("TimeShower:recoilDeadCone");
      // All ok.
      return true;
    }

    // Allow a veto after an FSR emission
    bool canVetoFSREmission() override { return doTopRecoil; }

    // Access the event after an FSR emission, specifically inside top decay.
    bool doVetoFSREmission(int sizeOld, const Event& event, int iSys, bool inResonance) override {
      // Check that we are inside a resonance decay.
      if (!inResonance)
        return false;

      // Check that it is a top decay.
      int iTop = partonSystemsPtr->getInRes(iSys);
      if (iTop == 0 || event[iTop].idAbs() != 6)
        return false;

      // Skip first emission, where ME corrections are already made.
      int sizeOut = partonSystemsPtr->sizeOut(iSys);
      if (sizeOut == 2)
        return false;

      // Location of trial new particles: radiator, emitted, recoiler.
      int iRad = sizeOld;
      int iEmt = sizeOld + 1;
      int iRec = sizeOld + 2;

      // The above partons are after emission;
      // alternatively use the ones before.
      if (useOldDipole) {
        iRad = event[iRad].mother1();
        iRec = event[iRec].mother1();
      }

      // Check if newly emitted gluon matches (anti)top colour line.
      if (event[iEmt].id() != 21)
        return false;
      if (event[iTop].id() == 6) {
        if (event[iEmt].col() != event[iTop].col())
          return false;
      } else {
        if (event[iEmt].acol() != event[iTop].acol())
          return false;
      }

      // Recoiler should now be a W, else something is wrong.
      if (event[iRec].idAbs() != 24) {
        return false;
      }

      // Denominator: eikonal weight with W as recoiler.
      double pRadRec = event[iRad].p() * event[iRec].p();
      double pRadEmt = event[iRad].p() * event[iEmt].p();
      double pRecEmt = event[iRec].p() * event[iEmt].p();
      double wtW = 2. * pRadRec / (pRadEmt * pRecEmt) - pow2(event[iRad].m() / pRadEmt);
      // If recoilDeadCone = on, include W mass term in denominator.
      if (recoilDeadCone)
        wtW -= pow2(event[iRec].m() / pRecEmt);

      // Numerator: eikonal weight with top as recoiler.
      double pRadTop = event[iRad].p() * event[iTop].p();
      double pTopEmt = event[iTop].p() * event[iEmt].p();
      double wtT =
          2. * pRadTop / (pRadEmt * pTopEmt) - pow2(event[iRad].m() / pRadEmt) - pow2(event[iTop].m() / pTopEmt);

      // Histogram weight ratio.
      wtCorr->fill(wtT / wtW);

      // List relevant properties.
      if (doList) {
        partonSystemsPtr->list();
        event.list();
      }
      
      // Accept/reject emission. Smooth suppression or step function.
      return (wtT < wtW * rndmPtr->flat());
    }

  private:
    // Options and Histograms.
    bool doTopRecoil, useOldDipole, doList, recoilDeadCone;
    Hist* wtCorr;
  };

}  // namespace Pythia8
