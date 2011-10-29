#include "GeneratorInterface/Pythia8Interface/interface/EmissionVetoHook.h"

  // Use VetoMIStep to analyse the incoming LHEF event and
  // extract the veto scale
  bool EmissionVetoHook::doVetoMIStep(int, const Pythia8::Event &e) {
    // Check that partons 5 and 6 are the t/tbar pair
    if (e[5].id() != 6 || e[6].id() != -6) {
      cout << "Error: could not find t/tbar pair" << endl;
      e.list();
      exit(1);
    }

    // Some events may not have radiation from POWHEG
    switch (e.size()) {
    case 7:
      // No radiation - veto scale is given by the factorisation scale
      pTpowheg = -1.;
      pTveto   = infoPtr->QFac();
      noRad    = true;

      // If this is the first no radiation event, then print scale
      if (firstNoRad) {
        cout << "Info: no POWHEG radiation, Q = " << pTveto
             << " GeV" << endl;
        firstNoRad = false;
      }
      break;

    case 8:
      // Radiation is parton 7 - first check that it is as expected
      if (e[7].id() != 21 && e[7].idAbs() > 5) {
        cout << "Error: jet is not quark/gluon" << endl;
        e.list();
        exit(1);
      }
      // Veto scale is given by jet pT
      pTveto = pTpowheg = e[7].pT();
      noRad  = false;
      noRad  = false;
      break;
    }

    // Initialise other variables
    nISRveto = nFSRveto = 0;
    pTshower = -1.;

    // Do not veto the event
    return false;
  }

  // For subsequent ISR/FSR emissions, find the pT of the shower
  // emission and veto as necessary
  bool EmissionVetoHook::doVetoISREmission(int, const Pythia8::Event &e) {
    // ISR - next shower emission is given status 43
    int i;
    for (i = e.size() - 1; i > 6; i--)
      if (e[i].isFinal() && e[i].status() == 43) break;
    if (i == 6) {
      cout << "Error: couldn't find ISR emission" << endl;
      e.list();
      exit(1);
    }
     
    // Veto if above the POWHEG scale
    if (e[i].pT() > pTveto) {
      nISRveto++;
      return true; 
    }
    // Store the first shower emission pT
    if (pTshower < 0.) pTshower = e[i].pT();
   
    return false;
  }

  bool EmissionVetoHook::doVetoFSREmission(int, const Pythia8::Event &e) {
    // FSR - shower emission will have status 51 and not be t/tbar
    int i;
    for (i = e.size() - 1; i > 6; i--)
      if (e[i].isFinal() && e[i].status() == 51 &&
          e[i].idAbs() != 6) break;
    if (i == 6) {
      cout << "Error: couldn't find FSR emission" << endl;
      e.list();
      exit(1);
    }

    // Veto if above the POWHEG scale
    if (e[i].pT() > pTveto) {
      nFSRveto++;  
      return true;
    }
    // Store the first shower emission pT   
    if (pTshower < 0.) pTshower = e[i].pT();

    return false;
  }

