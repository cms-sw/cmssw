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

//==========================================================================

// AlpgenPar: Class to parse ALPGEN parameter files and make them
//            available through a simple interface

class AlpgenPar {

public:

  // Constructor
  AlpgenPar(Pythia8::Info *infoPtrIn = NULL) : infoPtr(infoPtrIn) {}

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
  Pythia8::Info* infoPtr;

  // Constants
  static const double ZEROTHRESHOLD;

};

//==========================================================================

// LHAupAlpgen: LHAup derived class for reading in ALPGEN format
//              event files.

class LHAupAlpgen : public Pythia8::LHAup {

public:

  // Constructor and destructor.
  LHAupAlpgen(const char *baseFNin, Pythia8::Info *infoPtrIn = NULL);
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
  vector<Pythia8::LHAParticle> myParticles; // Local storage for particles

  // Constants
  static const bool   LHADEBUG, LHADEBUGRESCALE;
  static const double ZEROTHRESHOLD, EWARNTHRESHOLD, PTWARNTHRESHOLD,
                      INCOMINGMIN;

};

//==========================================================================

// AlpgenHooks: provides Alpgen file reading options.
// Note that it is defined with virtual inheritance, so that it can
// be combined with other UserHooks classes, see e.g. main32.cc.

class AlpgenHooks : virtual public Pythia8::UserHooks {

public:

  // Constructor and destructor
  AlpgenHooks(Pythia8::Pythia &pythia);
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
  MadgraphPar(Pythia8::Info *infoPtrIn = NULL) : infoPtr(infoPtrIn) {}

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
  Pythia8::Info *infoPtr;

  // Constants
  static const double ZEROTHRESHOLD;

};
#endif //  Pythia8_GeneratorInput_H
