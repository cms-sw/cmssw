// GeneratorInput.h is a part of the PYTHIA event generator.
// Copyright (C) 2014 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Primary Author: Richard Corke
// Secondary Author: Stephen Mrenna
// This file provides the following classes:
//   AlpgenPar:   a class for parsing ALPGEN parameter files
//                and reading back out the values
//   LHAupAlpgen: an LHAup derived class for reading in ALPGEN
//                format event files
//   AlpgenHooks: a UserHooks derived class for providing
//                'Alpgen:*' user options
//   MadgraphPar: a class for parsing MadGraph parameter files
//                and reading back out the values
// Example usage is shown in main32.cc, and further details
// can be found in the 'Jet Matching Style' manual page.
// Minor changes were made by the secondary author for integration
// with Madgraph-style matching, and Madgraph input was added.

#ifndef Pythia8_GeneratorInput_H
#define Pythia8_GeneratorInput_H

// Includes and namespace
#include "Pythia8/Pythia.h"
using namespace Pythia8;

//==========================================================================

// AlpgenPar: Class to parse ALPGEN parameter files and make them
//            available through a simple interface

class AlpgenPar {

public:

  // Constructor
  AlpgenPar(Info *infoPtrIn = NULL) : infoPtr(infoPtrIn) {}

  // Parse as incoming ALPGEN parameter file (passed as string)
  bool parse(const string paramStr);

  // Parse an incoming parameter line
  void extractRunParam(string line);

  // Check if a parameter exists
  bool haveParam(const string &paramIn) {
    return (params.find(paramIn) == params.end()) ? false : true; }

  // Get a parameter as a double or integer.
  // Caller should have already checked existance of the parameter.
  double getParam(const string &paramIn) {
    return (haveParam(paramIn)) ? params[paramIn] : 0.; }
  int    getParamAsInt(const string &paramIn) {
    return (haveParam(paramIn)) ? int(params[paramIn]) : 0.; }

  // Print parameters read from the '.par' file
  void printParams();

private:

  // Warn if a parameter is going to be overwriten
  void warnParamOverwrite(const string &paramIn, double val);

  // Simple string trimmer
  static string trim(string s);

  // Storage for parameters
  map<string,double> params;

  // Info pointer if provided
  Info* infoPtr;

  // Constants
  static const double ZEROTHRESHOLD;

};

//==========================================================================

// LHAupAlpgen: LHAup derived class for reading in ALPGEN format
//              event files.

class LHAupAlpgen : public LHAup {

public:

  // Constructor and destructor.
  LHAupAlpgen(const char *baseFNin, Info *infoPtrIn = NULL);
  ~LHAupAlpgen() { closeFile(isUnw, ifsUnw); }

  // Override fileFound routine from LHAup.
  bool fileFound() { return (isUnw != NULL); }

  // Override setInit/setEvent routines from LHAup.
  bool setInit();
  bool setEvent(int, double);

  // Print list of particles; mainly intended for debugging
  void printParticles();

private:

  // Add resonances to incoming event.
  bool addResonances();

  // Rescale momenta to remove any imbalances.
  bool rescaleMomenta();

  // Class variables
  string    baseFN, parFN, unwFN;  // Incoming filenames
  AlpgenPar alpgenPar;             // Parameter database
  int      lprup;                  // Process code
  double   ebmupA, ebmupB;         // Beam energies
  int      ihvy1, ihvy2;           // Heavy flavours for certain processes
  double   mb;                     // Bottom mass
  ifstream  ifsUnw;                // Input file stream for 'unw' file
  istream*  isUnw;                 // Input stream for 'unw' file
  vector<LHAParticle> myParticles; // Local storage for particles

  // Constants
  static const bool   LHADEBUG, LHADEBUGRESCALE;
  static const double ZEROTHRESHOLD, EWARNTHRESHOLD, PTWARNTHRESHOLD,
                      INCOMINGMIN;

};

//==========================================================================

// AlpgenHooks: provides Alpgen file reading options.
// Note that it is defined with virtual inheritance, so that it can
// be combined with other UserHooks classes, see e.g. main32.cc.

class AlpgenHooks : virtual public UserHooks {

public:

  // Constructor and destructor
  AlpgenHooks(Pythia &pythia);
  ~AlpgenHooks() { if (LHAagPtr) delete LHAagPtr; }

  // Override initAfterBeams routine from UserHooks
  bool initAfterBeams();

private:

  // LHAupAlpgen pointer
  LHAupAlpgen* LHAagPtr;

};

//==========================================================================

// MadgraphPar: Class to parse the Madgraph header parameters and
//               make them available through a simple interface

class MadgraphPar {

public:

  // Constructor
  MadgraphPar(Info *infoPtrIn = NULL) : infoPtr(infoPtrIn) {}

  // Parse an incoming Madgraph parameter file string
  bool parse(const string paramStr);

  // Parse an incoming parameter line
  void extractRunParam(string line);

  // Check if a parameter exists
  bool haveParam(const string &paramIn) {
    return (params.find(paramIn) == params.end()) ? false : true; }

  // Get a parameter as a double or integer.
  // Caller should have already checked existance of the parameter.
  double getParam(const string &paramIn) {
    return (haveParam(paramIn)) ? params[paramIn] : 0.; }
  int    getParamAsInt(const string &paramIn) {
    return (haveParam(paramIn)) ? int(params[paramIn]) : 0.; }

  // Print parameters read from the '.par' file
  void printParams();

private:

  // Warn if a parameter is going to be overwriten
  void warnParamOverwrite(const string &paramIn, double val);

  // Simple string trimmer
  static string trim(string s);

  // Storage for parameters
  map<string,double> params;

  // Info pointer if provided
  Info *infoPtr;

  // Constants
  static const double ZEROTHRESHOLD;

};

//==========================================================================

// Main implementation of AlpgenPar class.
// This may be split out to a separate C++ file if desired,
// but currently included here for ease of use.

//--------------------------------------------------------------------------

// Constants: could be changed here if desired, but normally should not.
// These are of technical nature, as described for each.

// A zero threshold value for double comparisons.
const double AlpgenPar::ZEROTHRESHOLD = 1e-10;

//--------------------------------------------------------------------------

// Warn if e/pT imbalance greater than these values
// Parse an incoming Alpgen parameter file string

bool AlpgenPar::parse(const string paramStr) {

  // Read par file in blocks:
  //   0 - process information
  //   1 - run parameters
  //   2 - cross sections
  int block = 0;

  // Loop over incoming lines
  stringstream paramStream(paramStr);
  string line;
  while (getline(paramStream, line)) {

    // Change to 'run parameters' block
    if        (line.find("run parameters") != string::npos) {
      block = 1;

    // End of 'run parameters' block
    } else if (line.find("end parameters") != string::npos) {
      block = 2;

    // Do not extract anything from block 0 so far
    } else if (block == 0) {

    // Block 1 or 2: extract parameters
    } else {
      extractRunParam(line);

    }
  } // while (getline(paramStream, line))

  return true;
}

//--------------------------------------------------------------------------

// Parse an incoming parameter line

void AlpgenPar::extractRunParam(string line) {

  // Extract information to the right of the final '!' character
  size_t idx = line.rfind("!");
  if (idx == string::npos) return;
  string paramName = trim(line.substr(idx + 1));
  string paramVal  = trim(line.substr(0, idx));
  istringstream iss(paramVal);

  // Special case: 'hard process code' - single integer input
  double val;
  if (paramName == "hard process code") {
    iss >> val;
    warnParamOverwrite("hpc", val);
    params["hpc"] = val;

  // Special case: 'Crosssection +- error (pb)' - two double values
  } else if (paramName.find("Crosssection") == 0) {
    double xerrup;
    iss >> val >> xerrup;
    warnParamOverwrite("xsecup", val);
    warnParamOverwrite("xerrup", val);
    params["xsecup"] = val;
    params["xerrup"] = xerrup;

  // Special case: 'unwtd events, lum (pb-1)' - integer and double values
  } else if (paramName.find("unwtd events") == 0) {
    int nevent;
    iss >> nevent >> val;
    warnParamOverwrite("nevent", val);
    warnParamOverwrite("lum", val);
    params["nevent"] = nevent;
    params["lum"]    = val;

  // Special case: 'mc,mb,...' - split on ',' for name and ' ' for values
  } else if (paramName.find(",") != string::npos) {
    
    // Simple tokeniser
    string        paramNameNow;
    istringstream issName(paramName);
    while (getline(issName, paramNameNow, ',')) {
      iss >> val;
      warnParamOverwrite(paramNameNow, val);
      params[paramNameNow] = val;
    }

  // Default case: assume integer and double on the left
  } else {
    int paramIdx;
    iss >> paramIdx >> val;
    warnParamOverwrite(paramName, val);
    params[paramName] = val;
  }
}

//--------------------------------------------------------------------------

// Print parameters read from the '.par' file

void AlpgenPar::printParams() {

  // Loop over all stored parameters and print
  cout << fixed << setprecision(3) << endl
       << " *-------  Alpgen parameters  -------*" << endl;
  for (map < string, double >::iterator it = params.begin();
       it != params.end(); ++it)
    cout << " |  " << left << setw(13) << it->first
         << "  |  " << right << setw(13) << it->second
         << "  |" << endl;
  cout << " *-----------------------------------*" << endl;
}

//--------------------------------------------------------------------------

// Warn if a parameter is going to be overwriten

void AlpgenPar::warnParamOverwrite(const string &paramIn, double val) {

  // Check if present and if new value is different
  if (haveParam(paramIn) &&
      abs(getParam(paramIn) - val) > ZEROTHRESHOLD) {
    if (infoPtr) infoPtr->errorMsg("Warning in LHAupAlpgen::"
        "warnParamOverwrite: overwriting existing parameter", paramIn);
  }
}

//--------------------------------------------------------------------------

// Simple string trimmer

string AlpgenPar::trim(string s) {

  // Remove whitespace in incoming string
  size_t i;
  if ((i = s.find_last_not_of(" \t\r\n")) != string::npos)
    s = s.substr(0, i + 1);
  if ((i = s.find_first_not_of(" \t\r\n")) != string::npos)
    s = s.substr(i);
  return s;
}

//==========================================================================

// Main implementation of LHAupAlpgen class.
// This may be split out to a separate C++ file if desired,
// but currently included here for ease of use.

// ----------------------------------------------------------------------

// Constants: could be changed here if desired, but normally should not.
// These are of technical nature, as described for each.

// Debug flag to print all particles in each event.
const bool LHAupAlpgen::LHADEBUG        = false;

// Debug flag to print particles when an e/p imbalance is found.
const bool LHAupAlpgen::LHADEBUGRESCALE = false;

// A zero threshold value for double comparisons.
const double LHAupAlpgen::ZEROTHRESHOLD   = 1e-10;

// Warn if e/pT imbalance greater than these values
const double LHAupAlpgen::EWARNTHRESHOLD  = 3e-3;
const double LHAupAlpgen::PTWARNTHRESHOLD = 1e-3;

// If incoming e/pZ is 0, it is reset to this value
const double LHAupAlpgen::INCOMINGMIN     = 1e-3;

// ----------------------------------------------------------------------

// Constructor. Opens parameter file and parses then opens event file.

LHAupAlpgen::LHAupAlpgen(const char* baseFNin, Info* infoPtrIn)
  : baseFN(baseFNin), alpgenPar(infoPtrIn), isUnw(NULL) {

  // Set the info pointer if given
  if (infoPtrIn) setPtr(infoPtrIn);

  // Read in '_unw.par' file to get parameters
  ifstream  ifsPar;
  istream*  isPar = NULL;

  // Try gzip file first then normal file afterwards
#ifdef GZIPSUPPORT
  parFN = baseFN + "_unw.par.gz";
  isPar = openFile(parFN.c_str(), ifsPar);
  if (!ifsPar.is_open()) closeFile(isPar, ifsPar);
#endif
  if (isPar == NULL) {
    parFN = baseFN + "_unw.par";
    isPar = openFile(parFN.c_str(), ifsPar);
    if (!ifsPar.is_open()) {
      if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::LHAupAlpgen: "
          "cannot open parameter file", parFN);
      closeFile(isPar, ifsPar);
      return;
    }
  }

  // Read entire contents into string and close file
  string paramStr((istreambuf_iterator<char>(isPar->rdbuf())),
                   istreambuf_iterator<char>());

  // Make sure we reached EOF and not other error
  if (ifsPar.bad()) {
    if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::LHAupAlpgen: "
        "cannot read parameter file", parFN);
    return;
  }
  closeFile(isPar, ifsPar);

  // Parse file and set LHEF header
  alpgenPar.parse(paramStr);
  if (infoPtr) setInfoHeader("AlpgenPar", paramStr);

  // Open '.unw' events file (with possible gzip support)
#ifdef GZIPSUPPORT
  unwFN = baseFN + ".unw.gz";
  isUnw = openFile(unwFN.c_str(), ifsUnw);
  if (!ifsUnw.is_open()) closeFile(isUnw, ifsUnw);
#endif
  if (isUnw == NULL) {
    unwFN = baseFN + ".unw";
    isUnw = openFile(unwFN.c_str(), ifsUnw);
    if (!ifsUnw.is_open()) {
      if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::LHAupAlpgen: "
          "cannot open event file", unwFN);
      closeFile(isUnw, ifsUnw);
    }
  }
}

// ----------------------------------------------------------------------

// setInit is a virtual method that must be finalised here.
// Sets up beams, strategy and processes.

bool LHAupAlpgen::setInit() {

  // Check that all required parameters are present
  if (!alpgenPar.haveParam("ih2") || !alpgenPar.haveParam("ebeam")  ||
      !alpgenPar.haveParam("hpc") || !alpgenPar.haveParam("xsecup") ||
      !alpgenPar.haveParam("xerrup")) {
    if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::setInit: "
        "missing input parameters");
    return false;
  }

  // Beam IDs
  int ih2 = alpgenPar.getParamAsInt("ih2");
  int idbmupA = 2212;
  int idbmupB = (ih2 == 1) ? 2212 : -2212;

  // Beam energies
  double ebeam = alpgenPar.getParam("ebeam");
  ebmupA = ebeam;
  ebmupB = ebmupA;

  // PDF group and set (at the moment not set)
  int pdfgupA = 0, pdfsupA = 0;
  int pdfgupB = 0, pdfsupB = 0;

  // Strategy is for unweighted events and xmaxup not important
  int    idwtup = 3;
  double xmaxup = 0.;

  // Get hard process code
  lprup = alpgenPar.getParamAsInt("hpc");

  // Check for unsupported processes
  if (lprup == 7 || lprup == 8 || lprup == 13) {
    if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::setInit: "
        "process not implemented");
    return false;
  }

  // Depending on the process code, get heavy flavour information:
  //    6 = QQbar           + jets
  //    7 = QQbar + Q'Qbar' + jets
  //    8 = QQbar + Higgs   + jets
  //   16 = QQbar + gamma   + jets
  if (lprup == 6 || lprup == 7 || lprup == 8 || lprup == 16) {
    if (!alpgenPar.haveParam("ihvy")) {
      if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::setInit: "
          "heavy flavour information not present");
      return false;
    }
    ihvy1 = alpgenPar.getParamAsInt("ihvy");

  } else ihvy1 = -1;
  if (lprup == 7) {
    if (!alpgenPar.haveParam("ihvy2")) {
      if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::setInit: "
          "heavy flavour information not present");
      return false;
    }
    ihvy2 = alpgenPar.getParamAsInt("ihvy2");
  } else ihvy2 = -1;
  // For single top (process 13), get b mass to set incoming
  mb = -1.;
  if (lprup == 13) {
    if (!alpgenPar.haveParam("mb")) {
      if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::setInit: "
          "heavy flavour information not present");
      return false;
    }
    mb = alpgenPar.getParam("mb");
  }

  // Set the beams
  setBeamA(idbmupA, ebmupA, pdfgupA, pdfsupA);
  setBeamB(idbmupB, ebmupB, pdfgupB, pdfsupB);
  setStrategy(idwtup);

  // Add the process
  double xsecup = alpgenPar.getParam("xsecup");
  double xerrup = alpgenPar.getParam("xerrup");
  addProcess(lprup, xsecup, xerrup, xmaxup);
  xSecSumSave = xsecup;
  xErrSumSave = xerrup;

  // All okay
  return true;
}

// ----------------------------------------------------------------------

// setEvent is a virtual method that must be finalised here.
// Read in an event from the 'unw' file and setup.

bool LHAupAlpgen::setEvent(int, double) {

  // Read in the first line of the event
  int    nEvent, iProc, nParton;
  double Swgt, Sq;
  string line;
  if (!getline(*isUnw, line)) {
    // Read was bad
    if (ifsUnw.bad()) {
      if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::setEvent: "
          "could not read events from file");
      return false;
    }
    // End of file reached
    if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::setEvent: "
        "end of file reached");
    return false;
  }
  istringstream iss1(line);
  iss1 >> nEvent >> iProc >> nParton >> Swgt >> Sq;

  // Set the process details (ignore alphaQED and alphaQCD parameters)
  double wgtT = Swgt, scaleT = Sq;
  setProcess(lprup, wgtT, scaleT);

  // Incoming flavour and x information for later
  int    id1T, id2T;
  double x1T, x2T;
  // Temporary storage for read in parton information
  int    idT, statusT, mother1T, mother2T, col1T, col2T;
  double pxT, pyT, pzT, eT, mT;
  // Leave tau and spin as default values
  double tauT = 0., spinT = 9.;

  // Store particles locally at first so that resonances can be added
  myParticles.clear();

  // Now read in partons
  for (int i = 0; i < nParton; i++) {
    // Get the next line
    if (!getline(*isUnw, line)) {
      if (infoPtr) infoPtr->errorMsg("Error in LHAupAlpgen::setEvent: "
          "could not read events from file");
      return false;
    }
    istringstream iss2(line);

    // Incoming (flavour, colour, anticolour, pz)
    if (i < 2) {
      // Note that mothers will be set automatically by Pythia, and LHA
      // status -1 is for an incoming parton
      iss2 >> idT >> col1T >> col2T >> pzT;
      statusT  = -1;
      mother1T = mother2T = 0;
      pxT = pyT = mT = 0.;
      eT  = abs(pzT);

      // Adjust when zero pz/e
      if (pzT == 0.) {
        pzT = (i == 0) ? INCOMINGMIN : -INCOMINGMIN;
        eT  = INCOMINGMIN;
      }

    // Outgoing (flavour, colour, anticolour, px, py, pz, mass)
    } else {
      // Note that mothers 1 and 2 corresport to the incoming partons,
      // as set above, and LHA status +1 is for outgoing final state
      iss2 >> idT >> col1T >> col2T >> pxT >> pyT >> pzT >> mT;
      statusT  = 1;
      mother1T = 1;
      mother2T = 2;
      eT = sqrt(max(0., pxT*pxT + pyT*pyT + pzT*pzT + mT*mT));
    }

    // Add particle
    myParticles.push_back(LHAParticle(
        idT, statusT, mother1T, mother2T, col1T, col2T,
        pxT, pyT, pzT, eT, mT, tauT, spinT,-1.));
  }

  // Add resonances if required
  if (!addResonances()) return false;

  // Rescale momenta if required (must be done after full event
  // reconstruction in addResonances)
  if (!rescaleMomenta()) return false;

  // Pass particles on to Pythia
  for (size_t i = 0; i < myParticles.size(); i++)
    addParticle(myParticles[i]);

  // Set incoming flavour/x information and done
  id1T = myParticles[0].idPart;
  x1T  = myParticles[0].ePart / ebmupA;
  id2T = myParticles[1].idPart;
  x2T  = myParticles[1].ePart / ebmupA;
  setIdX(id1T, id2T, x1T, x2T);
  setPdf(id1T, id2T, x1T, x2T, 0., 0., 0., false);
  return true;
}

// ----------------------------------------------------------------------

// Print list of particles; mainly intended for debugging

void LHAupAlpgen::printParticles() {

  cout << endl << "---- LHAupAlpgen particle listing begin ----" << endl;
  cout << scientific << setprecision(6);
  for (int i = 0; i < int(myParticles.size()); i++) {
    cout << setw(5)  << i
         << setw(5)  << myParticles[i].idPart
         << setw(5)  << myParticles[i].statusPart
         << setw(15) << myParticles[i].pxPart
         << setw(15) << myParticles[i].pyPart
         << setw(15) << myParticles[i].pzPart
         << setw(15) << myParticles[i].ePart
         << setw(15) << myParticles[i].mPart
         << setw(5)  << myParticles[i].mother1Part - 1
         << setw(5)  << myParticles[i].mother2Part - 1
         << setw(5)  << myParticles[i].col1Part
         << setw(5)  << myParticles[i].col2Part
         << endl;
  }
  cout << "----  LHAupAlpgen particle listing end  ----" << endl;
}

// ----------------------------------------------------------------------

// Routine to add resonances to an incoming event based on the
// hard process code (now stored in lprup).

bool LHAupAlpgen::addResonances() {

  // Temporary storage for resonance information
  int    idT, statusT, mother1T, mother2T, col1T, col2T;
  double pxT, pyT, pzT, eT, mT;
  // Leave tau and spin as default values
  double tauT = 0., spinT = 9.;

  // Alpgen process dependent parts. Processes:
  //    1 = W        + QQbar         + jets
  //    2 = Z/gamma* + QQbar         + jets
  //    3 = W                        + jets
  //    4 = Z/gamma*                 + jets
  //   10 = W        + c             + jets
  //   14 = W        + gamma         + jets
  //   15 = W        + QQbar + gamma + jets
  // When QQbar = ttbar, tops are not decayed in these processes.
  // Explicitly reconstruct W/Z resonances; assumption is that the
  // decay products are the last two particles.
  if (lprup <= 4 || lprup == 10 || lprup == 14 || lprup == 15) {
    // Decay products are the last two entries
    int i1 = myParticles.size() - 1, i2 = i1 - 1;

    // Follow 'alplib/alpsho.f' procedure to get ID
    if (myParticles[i1].idPart + myParticles[i2].idPart == 0)
      idT = 0;
    else
      idT = - (myParticles[i1].idPart % 2) - (myParticles[i2].idPart % 2);
    idT = (idT > 0) ? 24 : (idT < 0) ? -24 : 23;

    // Check that we get the expected resonance type; Z/gamma*
    if (lprup == 2 || lprup == 4) {
      if (idT != 23) {
        if (infoPtr) infoPtr->errorMsg("Error in "
            "LHAupAlpgen::addResonances: wrong resonance type in event");
        return false;
      }

    // W's
    } else {
      if (abs(idT) != 24) {
        if (infoPtr) infoPtr->errorMsg("Error in "
            "LHAupAlpgen::addResonances: wrong resonance type in event");
        return false;
      }
    }

    // Remaining input
    statusT  = 2;
    mother1T = 1;
    mother2T = 2;
    col1T = col2T = 0;
    pxT = myParticles[i1].pxPart + myParticles[i2].pxPart;
    pyT = myParticles[i1].pyPart + myParticles[i2].pyPart;
    pzT = myParticles[i1].pzPart + myParticles[i2].pzPart;
    eT  = myParticles[i1].ePart  + myParticles[i2].ePart;
    mT  = sqrt(eT*eT - pxT*pxT - pyT*pyT - pzT*pzT);
    myParticles.push_back(LHAParticle(
        idT, statusT, mother1T, mother2T, col1T, col2T,
        pxT, pyT, pzT, eT, mT, tauT, spinT, -1.));

    // Update decay product mothers (note array size as if from 1)
    myParticles[i1].mother1Part = myParticles[i2].mother1Part =
        myParticles.size();
    myParticles[i1].mother2Part = myParticles[i2].mother2Part = 0;

  // Processes:
  //    5 = nW + mZ + j gamma + lH + jets
  //    6 = QQbar         + jets    (QQbar = ttbar)
  //    8 = QQbar + Higgs + jets    (QQbar = ttbar)
  //   13 = top   + q               (topprc = 1)
  //   13 = top   + b               (topprc = 2)
  //   13 = top   + W     + jets    (topprc = 3)
  //   13 = top   + W     + b       (topprc = 4)
  //   16 = QQbar + gamma + jets    (QQbar = ttbar)
  //
  // When tops are present, they are decayed to Wb (both the W and b
  // are not given), with this W also decaying (decay products given).
  // The top is marked intermediate, the (intermediate) W is
  // reconstructed from its decay products, and the decay product mothers
  // updated. The final-state b is reconstructed from (top - W).
  //
  // W/Z resonances are given, as well as their decay products. The
  // W/Z is marked intermediate, and the decay product mothers updated.
  //
  // It is always assumed that decay products are at the end.
  // For processes 5 and 13, it is also assumed that the decay products
  // are in the same order as the resonances.
  // For processes 6, 8 and 16, the possibility of the decay products
  // being out-of-order is also taken into account.
  } else if ( ((lprup == 6 || lprup == 8 || lprup == 16) && ihvy1 == 6) ||
              lprup == 5 || lprup == 13) {

    // Go backwards through the particles looking for W/Z/top
    int idx = myParticles.size() - 1;
    for (int i = myParticles.size() - 1; i > -1; i--) {

      // W or Z
      if (myParticles[i].idPart == 23 ||
          abs(myParticles[i].idPart) == 24) {

        // Check that decay products and resonance match up
        int flav;
        if (myParticles[idx].idPart + myParticles[idx - 1].idPart == 0)
          flav = 0;
        else
          flav = - (myParticles[idx].idPart % 2)
                 - (myParticles[idx - 1].idPart % 2);
        flav = (flav > 0) ? 24 : (flav < 0) ? -24 : 23;
        if (flav != myParticles[i].idPart) {
          if (infoPtr)
            infoPtr->errorMsg("Error in LHAupAlpgen::addResonance: "
                "resonance does not match decay products");
          return false;
        }

        // Update status/mothers
        myParticles[i].statusPart      = 2;
        myParticles[idx  ].mother1Part = i + 1;
        myParticles[idx--].mother2Part = 0;
        myParticles[idx  ].mother1Part = i + 1;
        myParticles[idx--].mother2Part = 0;

      // Top
      } else if (abs(myParticles[i].idPart) == 6) {

        // Check that decay products and resonance match up
        int flav;
        if (myParticles[idx].idPart + myParticles[idx - 1].idPart == 0)
          flav = 0;
        else
          flav = - (myParticles[idx].idPart % 2)
                 - (myParticles[idx - 1].idPart % 2);
        flav = (flav > 0) ? 24 : (flav < 0) ? -24 : 23;

        bool outOfOrder = false, wrongFlavour = false;;
        if ( abs(flav) != 24 ||
             (flav ==  24 && myParticles[i].idPart !=  6) ||
             (flav == -24 && myParticles[i].idPart != -6) ) {

          // Processes 5 and 13, order should always be correct
          if (lprup == 5 || lprup == 13) {
            wrongFlavour = true;

          // Processes 6, 8 and 16, can have out of order decay products
          } else {

            // Go back two decay products and retry
            idx -= 2;
            if (myParticles[idx].idPart + myParticles[idx - 1].idPart == 0)
              flav = 0;
            else
              flav = - (myParticles[idx].idPart % 2)
                     - (myParticles[idx - 1].idPart % 2);
            flav = (flav > 0) ? 24 : (flav < 0) ? -24 : 23;

            // If still the wrong flavour then error
            if ( abs(flav) != 24 ||
                 (flav ==  24 && myParticles[i].idPart !=  6) ||
                 (flav == -24 && myParticles[i].idPart != -6) )
              wrongFlavour = true;
            else outOfOrder = true;
          }

          // Error if wrong flavour
          if (wrongFlavour) {
            if (infoPtr)
              infoPtr->errorMsg("Error in LHAupAlpgen::addResonance: "
                  "resonance does not match decay products");
            return false;
          }
        }

        // Mark t/tbar as now intermediate
        myParticles[i].statusPart = 2;

        // New intermediate W+/W-
        idT      = flav;
        statusT  = 2;
        mother1T = i + 1;
        mother2T = 0;
        col1T = col2T = 0;
        pxT = myParticles[idx].pxPart + myParticles[idx - 1].pxPart;
        pyT = myParticles[idx].pyPart + myParticles[idx - 1].pyPart;
        pzT = myParticles[idx].pzPart + myParticles[idx - 1].pzPart;
        eT  = myParticles[idx].ePart  + myParticles[idx - 1].ePart;
        mT  = sqrt(eT*eT - pxT*pxT - pyT*pyT - pzT*pzT);
        myParticles.push_back(LHAParticle(
            idT, statusT, mother1T, mother2T, col1T, col2T,
            pxT, pyT, pzT, eT, mT, tauT, spinT, -1.));

        // Update the decay product mothers
        myParticles[idx  ].mother1Part = myParticles.size();
        myParticles[idx--].mother2Part = 0;
        myParticles[idx  ].mother1Part = myParticles.size();
        myParticles[idx--].mother2Part = 0;

        // New final-state b/bbar
        idT     = (flav == 24) ? 5 : -5;
        statusT = 1;
        // Colour from top
        col1T   = myParticles[i].col1Part;
        col2T   = myParticles[i].col2Part;
        // Momentum from (t/tbar - W+/W-)
        pxT     = myParticles[i].pxPart - myParticles.back().pxPart;
        pyT     = myParticles[i].pyPart - myParticles.back().pyPart;
        pzT     = myParticles[i].pzPart - myParticles.back().pzPart;
        eT      = myParticles[i].ePart  - myParticles.back().ePart;
        mT      = sqrt(eT*eT - pxT*pxT - pyT*pyT - pzT*pzT);
        myParticles.push_back(LHAParticle(
            idT, statusT, mother1T, mother2T, col1T, col2T,
            pxT, pyT, pzT, eT, mT, tauT, spinT, -1.));

        // If decay products were out of order, reset idx to point
        // at correct decay products
        if (outOfOrder) idx += 4;

      } // if (abs(myParticles[i].idPart) == 6)
    } // for (i)


  // Processes:
  //    7 = QQbar + Q'Qbar' + jets (tops are not decayed)
  //    9 =                   jets
  //   11 = gamma           + jets
  //   12 = Higgs           + jets
  } else if (lprup == 7 || lprup == 9 || lprup == 11 || lprup == 12) {
    // Nothing to do for these processes
  }

  // For single top, set incoming b mass if necessary
  if (lprup == 13) for (int i = 0; i < 2; i++)
    if (abs(myParticles[i].idPart) == 5) {
      myParticles[i].mPart = mb;
      myParticles[i].ePart = sqrt(pow2(myParticles[i].pzPart) + pow2(mb));
    }

  // Debug output and done.
  if (LHADEBUG) printParticles();
  return true;

}

// ----------------------------------------------------------------------

// Routine to rescale momenta to remove any imbalances. The routine
// assumes that any imbalances are due to decimal output/rounding
// effects, and are therefore small.
//
// First any px/py imbalances are fixed by adjusting all outgoing
// particles px/py and also updating their energy so mass is fixed.
// Because incoming pT is zero, changes should be limited to ~0.001.
//
// Second, any pz/e imbalances are fixed by scaling the incoming beams
// (again, no changes to masses required). Because incoming pz/e is not
// zero, effects can be slightly larger ~0.002/0.003.

bool LHAupAlpgen::rescaleMomenta() {

  // Total momenta in/out
  int  nOut = 0;
  Vec4 pIn, pOut;
  for (int i = 0; i < int(myParticles.size()); i++) {
    Vec4 pNow = Vec4(myParticles[i].pxPart, myParticles[i].pyPart,
                     myParticles[i].pzPart, myParticles[i].ePart);
    if (i < 2) pIn += pNow;
    else if (myParticles[i].statusPart == 1) {
      nOut++;
      pOut += pNow;
    }
  }

  // pT out to match pT in. Split any imbalances over all outgoing
  // particles, and scale energies also to keep m^2 fixed.
  if (abs(pOut.pT() - pIn.pT()) > ZEROTHRESHOLD) {
    // Differences in px/py
    double pxDiff = (pOut.px() - pIn.px()) / nOut,
           pyDiff = (pOut.py() - pIn.py()) / nOut;

    // Warn if resulting changes above warning threshold
    if (pxDiff > PTWARNTHRESHOLD || pyDiff > PTWARNTHRESHOLD) {
      if (infoPtr) infoPtr->errorMsg("Warning in LHAupAlpgen::setEvent: "
          "large pT imbalance in incoming event");

      // Debug printout
      if (LHADEBUGRESCALE) {
        printParticles();
        cout << "pxDiff = " << pxDiff << ", pyDiff = " << pyDiff << endl;
      }
    }

    // Adjust all final-state outgoing
    pOut.reset();
    for (int i = 2; i < int(myParticles.size()); i++) {
      if (myParticles[i].statusPart != 1) continue;
      myParticles[i].pxPart -= pxDiff;
      myParticles[i].pyPart -= pyDiff;
      myParticles[i].ePart   = sqrt(max(0., pow2(myParticles[i].pxPart) +
          pow2(myParticles[i].pyPart) + pow2(myParticles[i].pzPart) +
          pow2(myParticles[i].mPart)));
      pOut += Vec4(myParticles[i].pxPart, myParticles[i].pyPart,
                   myParticles[i].pzPart, myParticles[i].ePart);
    }
  }

  // Differences in E/pZ and scaling factors
  double de = (pOut.e()  - pIn.e());
  double dp = (pOut.pz() - pIn.pz());
  double a  = 1 + (de + dp) / 2. / myParticles[0].ePart;
  double b  = 1 + (de - dp) / 2. / myParticles[1].ePart;

  // Warn if resulting energy changes above warning threshold.
  // Change in pz less than or equal to change in energy (incoming b
  // quark can have mass at this stage for process 13). Note that for
  // very small incoming momenta, the relative adjustment may be large,
  // but still small in absolute terms.
  if (abs(a - 1.) * myParticles[0].ePart > EWARNTHRESHOLD ||
      abs(b - 1.) * myParticles[1].ePart > EWARNTHRESHOLD) {
    if (infoPtr) infoPtr->errorMsg("Warning in LHAupAlpgen::setEvent: "
        "large rescaling factor");

    // Debug printout
    if (LHADEBUGRESCALE) {
      printParticles();
      cout << "de = " << de << ", dp = " << dp
           << ", a = " << a << ", b = " << b << endl
           << "Absolute energy change for incoming 0 = "
           << abs(a - 1.) * myParticles[0].ePart << endl
           << "Absolute energy change for incoming 1 = "
           << abs(b - 1.) * myParticles[1].ePart << endl;
    }
  }
  myParticles[0].ePart  *= a;
  myParticles[0].pzPart *= a;
  myParticles[1].ePart  *= b;
  myParticles[1].pzPart *= b;

  // Recalculate resonance four vectors
  for (int i = 0; i < int(myParticles.size()); i++) {
    if (myParticles[i].statusPart != 2) continue;

    // Only mothers stored in LHA, so go through all
    Vec4 resVec;
    for (int j = 0; j < int(myParticles.size()); j++) {
      if (myParticles[j].mother1Part - 1 != i) continue;
      resVec += Vec4(myParticles[j].pxPart, myParticles[j].pyPart,
                     myParticles[j].pzPart, myParticles[j].ePart);
    }

    myParticles[i].pxPart = resVec.px();
    myParticles[i].pyPart = resVec.py();
    myParticles[i].pzPart = resVec.pz();
    myParticles[i].ePart  = resVec.e();
  }

  return true;
}

//==========================================================================

// Main implementation of AlpgenHooks class.
// This may be split out to a separate C++ file if desired,
// but currently included here for ease of use.

// ----------------------------------------------------------------------

// Constructor: provides the 'Alpgen:file' option by directly
//              changing the Pythia 'Beams' settings

AlpgenHooks::AlpgenHooks(Pythia &pythia) : LHAagPtr(NULL) {

  // If LHAupAlpgen needed, construct and pass to Pythia
  string agFile = pythia.settings.word("Alpgen:file");
  if (agFile != "void") {
    LHAagPtr = new LHAupAlpgen(agFile.c_str(), &pythia.info);
    pythia.settings.mode("Beams:frameType", 5);
    pythia.setLHAupPtr(LHAagPtr);
  }
}

// ----------------------------------------------------------------------

// Initialisation routine which is called by pythia.init().
// This happens after the local pointers have been assigned and after
// Pythia has processed the Beam information (and therefore LHEF header
// information has been read in), but before any other internal
// initialisation. Provides the remaining 'Alpgen:*' options.

bool AlpgenHooks::initAfterBeams() {

  // Read in ALPGEN specific configuration variables
  bool setMasses = settingsPtr->flag("Alpgen:setMasses");
  bool setNjet   = settingsPtr->flag("Alpgen:setNjet");
  bool setMLM    = settingsPtr->flag("Alpgen:setMLM");

  // If ALPGEN parameters are present, then parse in AlpgenPar object
  AlpgenPar par(infoPtr);
  string parStr = infoPtr->header("AlpgenPar");
  if (!parStr.empty()) {
    par.parse(parStr);
    par.printParams();
  }

  // Set masses if requested
  if (setMasses) {
    if (par.haveParam("mc")) particleDataPtr->m0(4,  par.getParam("mc"));
    if (par.haveParam("mb")) particleDataPtr->m0(5,  par.getParam("mb"));
    if (par.haveParam("mt")) particleDataPtr->m0(6,  par.getParam("mt"));
    if (par.haveParam("mz")) particleDataPtr->m0(23, par.getParam("mz"));
    if (par.haveParam("mw")) particleDataPtr->m0(24, par.getParam("mw"));
    if (par.haveParam("mh")) particleDataPtr->m0(25, par.getParam("mh"));
  }

  // Set MLM:nJets if requested
  if (setNjet) {
    if (par.haveParam("njets"))
      settingsPtr->mode("JetMatching:nJet", par.getParamAsInt("njets"));
    else
      infoPtr->errorMsg("Warning in AlpgenHooks:init: "
          "no ALPGEN nJet parameter found");
  }

  // Set MLM merging parameters if requested
  if (setMLM) {
    if (par.haveParam("ptjmin") && par.haveParam("drjmin") &&
        par.haveParam("etajmax")) {
      double ptjmin = par.getParam("ptjmin");
      ptjmin = max(ptjmin + 5., 1.2 * ptjmin);
      settingsPtr->parm("JetMatching:eTjetMin",   ptjmin);
      settingsPtr->parm("JetMatching:coneRadius", par.getParam("drjmin"));
      settingsPtr->parm("JetMatching:etaJetMax",  par.getParam("etajmax"));

    // Warn if setMLM requested, but parameters not present
    } else {
      infoPtr->errorMsg("Warning in AlpgenHooks:init: "
          "no ALPGEN merging parameters found");
    }
  }

  // Initialisation complete.
  return true;
}

//==========================================================================

// Main implementation of MadgraphPar class.
// This may be split out to a separate C++ file if desired,
// but currently included here for ease of use.

//--------------------------------------------------------------------------

// Constants: could be changed here if desired, but normally should not.
// These are of technical nature, as described for each.

// A zero threshold value for double comparisons.
const double MadgraphPar::ZEROTHRESHOLD = 1e-10;

//--------------------------------------------------------------------------

// Parse an incoming Madgraph parameter file string

bool MadgraphPar::parse(const string paramStr) {

  // Loop over incoming lines
  stringstream paramStream(paramStr);
  string line;
  while ( getline(paramStream, line) ) extractRunParam(line);
  return true;
  
}

//--------------------------------------------------------------------------

// Parse an incoming parameter line

void MadgraphPar::extractRunParam(string line) {

  // Extract information to the right of the final '!' character
  size_t idz = line.find("#");
  if ( !(idz == string::npos) ) return;
  size_t idx = line.find("=");
  size_t idy = line.find("!");
  if (idy == string::npos) idy = line.size();
  if (idx == string::npos) return;
  string paramName = trim( line.substr( idx + 1, idy - idx - 1) );
  string paramVal  = trim( line.substr( 0, idx) );
  replace( paramVal.begin(), paramVal.end(), 'd', 'e');

  // Simple tokeniser
  istringstream iss(paramVal);
  double val;
  if (paramName.find(",") != string::npos) {
    string        paramNameNow;
    istringstream issName( paramName);
    while ( getline(issName, paramNameNow, ',') ) {
      iss >> val;
      warnParamOverwrite( paramNameNow, val);
      params[paramNameNow] = val;
    }

  // Default case: assume integer and double on the left
  } else {
    iss >> val;
    warnParamOverwrite( paramName, val);
    params[paramName] = val;
  }
}

//--------------------------------------------------------------------------

// Print parameters read from the '.par' file

void MadgraphPar::printParams() {

  // Loop over all stored parameters and print
  cout << endl
       << " *--------  Madgraph parameters  --------*" << endl;
  for (map<string,double>::iterator it = params.begin();
       it != params.end(); ++it)
    cout << " |  " << left << setw(15) << it->first
         << "  |  " << right << setw(15) << it->second
         << "  |" << endl;
  cout << " *---------------------------------------*" << endl;
}

//--------------------------------------------------------------------------

// Warn if a parameter is going to be overwriten

void MadgraphPar::warnParamOverwrite(const string &paramIn, double val) {

  // Check if present and if new value is different
  if (haveParam(paramIn) &&
      abs(getParam(paramIn) - val) > ZEROTHRESHOLD) {
    if (infoPtr) infoPtr->errorMsg("Warning in LHAupAlpgen::"
        "warnParamOverwrite: overwriting existing parameter", paramIn);
  }
}

//--------------------------------------------------------------------------

// Simple string trimmer

string MadgraphPar::trim(string s) {

  // Remove whitespace in incoming string
  size_t i;
  if ( (i = s.find_last_not_of(" \t\r\n")) != string::npos)
    s = s.substr(0, i + 1);
  if ( (i = s.find_first_not_of(" \t\r\n")) != string::npos)
    s = s.substr(i);
  return s;
}

//==========================================================================

#endif //  Pythia8_GeneratorInput_H
