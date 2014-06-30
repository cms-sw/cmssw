// JetMatching.h is a part of the PYTHIA event generator.
// Copyright (C) 2014 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Richard Corke (implementation of MLM matching as
// in Alpgen for Alpgen input)
// and Stephen Mrenna (implementation of MLM-style matching as
// in Madgraph for Alpgen or Madgraph 5 input.)
// and Simon de Visscher and Stefan Prestel (implementation of shower-kT
// MLM-style matching and flavour treatment for Madgraph input, and FxFx NLO
// jet matching with aMC@NLO.)
// This file provides the classes to perform MLM matching of
// Alpgen or MadGraph 5 input.
// Example usage is shown in main32.cc, and further details
// can be found in the 'Jet Matching Style' manual page.

#ifndef Pythia8_JetMatching_H
#define Pythia8_JetMatching_H

// Includes
#include "Pythia8/Pythia.h"
#include "GeneratorInput.h"

//==========================================================================

// Declaration of main JetMatching class to perform MLM matching.
// Note that it is defined with virtual inheritance, so that it can
// be combined with other UserHooks classes, see e.g. main33.cc.

class JetMatching : virtual public Pythia8::UserHooks {

public:

  // Constructor and destructor
 JetMatching() : cellJet(NULL), slowJet(NULL), slowJetHard(NULL) {}
  ~JetMatching() {
    if (cellJet) delete cellJet;
    if (slowJet) delete slowJet;
    if (slowJetHard) delete slowJetHard;
  }

  // Initialisation
  virtual bool initAfterBeams() = 0;

  // Process level vetos
  bool canVetoProcessLevel() { return doMerge; }
  bool doVetoProcessLevel(Pythia8::Event& process) {
    eventProcessOrig = process;
    return false;
  }

  // Parton level vetos (before beam remnants and resonance decays)
  bool canVetoPartonLevelEarly() { return doMerge; }
  bool doVetoPartonLevelEarly(const Pythia8::Event& event);

  // Shower step vetoes (after the first emission, for Shower-kT scheme)
  int  numberVetoStep() {return 1;}
  bool canVetoStep() { return true; }
  bool doVetoStep(int,  int, int, const Pythia8::Event& ) { return false; }

protected:

  // Constants to be changed for debug printout or extra checks.
  static const bool MATCHINGDEBUG, MATCHINGCHECK;

  // Different steps of the matching algorithm.
  virtual void sortIncomingProcess(const Pythia8::Event &)=0;
  virtual void jetAlgorithmInput(const Pythia8::Event &, int)=0;
  virtual void runJetAlgorithm()=0;
  virtual bool matchPartonsToJets(int)=0;
  virtual int  matchPartonsToJetsLight()=0;
  virtual int  matchPartonsToJetsHeavy()=0;

  enum vetoStatus { NONE, LESS_JETS, MORE_JETS, HARD_JET, UNMATCHED_PARTON };
  enum partonTypes { ID_CHARM=4, ID_BOT=5, ID_TOP=6, ID_LEPMIN=11,
    ID_LEPMAX=16, ID_GLUON=21, ID_PHOTON=22 };

  // Master switch for merging
  bool   doMerge;
  // Switch for merging in the shower-kT scheme. Needed here because
  // the scheme uses different UserHooks functionality.
  bool   doShowerKt;

  // Maximum and current number of jets
  int    nJetMax, nJet;

  // Jet algorithm parameters
  int    jetAlgorithm;
  double eTjetMin, coneRadius, etaJetMax, etaJetMaxAlgo;

  // Internal jet algorithms
  Pythia8::CellJet* cellJet;
  Pythia8::SlowJet* slowJet;
  Pythia8::SlowJet* slowJetHard;

  // SlowJet specific
  int    slowJetPower;

  // Event records to store original incoming process, final-state of the
  // incoming process and what will be passed to the jet algorithm.
  // Not completely necessary to store all steps, but makes tracking the
  // steps of the algorithm a lot easier.
  Pythia8::Event eventProcessOrig, eventProcess, workEventJet;
 
  // Sort final-state of incoming process into light/heavy jets and 'other'
  vector<int> typeIdx[3];
  set<int>    typeSet[3];

  // Momenta output of jet algorithm (to provide same output regardless of
  // the selected jet algorithm)
  vector<Pythia8::Vec4> jetMomenta;

  // CellJet specific
  int    nEta, nPhi;
  double eTseed, eTthreshold;

  // Merging procedure parameters
  int    jetAllow, jetMatch, exclusiveMode;
  double coneMatchLight, coneRadiusHeavy, coneMatchHeavy;
  bool   exclusive;

  // Store the minimum eT/pT of matched light jets
  double eTpTlightMin;

};

//==========================================================================

// Declaration of main UserHooks class to perform Alpgen matching.

class JetMatchingAlpgen : virtual public JetMatching {

public:

  // Constructor and destructor
  JetMatchingAlpgen() { }
  ~JetMatchingAlpgen() { }

  // Initialisation
  bool initAfterBeams();

private:

  // Different steps of the matching algorithm.
  void sortIncomingProcess(const Pythia8::Event &);
  void jetAlgorithmInput(const Pythia8::Event &, int);
  void runJetAlgorithm();
  bool matchPartonsToJets(int);
  int  matchPartonsToJetsLight();
  int  matchPartonsToJetsHeavy();

  // Sorting utility
  void sortTypeIdx(vector < int > &vecIn);

  // Constants
  static const double GHOSTENERGY, ZEROTHRESHOLD;

};

//==========================================================================

// Declaration of main UserHooks class to perform Madgraph matching.

class JetMatchingMadgraph : virtual public JetMatching {

public:

  // Constructor and destructor
  JetMatchingMadgraph() { }
  ~JetMatchingMadgraph() { }

  // Initialisation
  bool initAfterBeams();

  // Shower step vetoes (after the first emission, for Shower-kT scheme)
  int  numberVetoStep() {return 1;}
  bool canVetoStep() { return doShowerKt; }
  bool doVetoStep(int,  int, int, const Pythia8::Event& );

  // Jet algorithm to access the jet separations in the cleaned event
  // after showering.
  Pythia8::SlowJet* slowJetDJR;
  // Function to return the jet clustering scales.
  vector<double> GetDJR() { return DJR;}
  vector<int> nMEPartons(){return nME;} 
protected:

  // Different steps of the matching algorithm.
  void sortIncomingProcess(const Pythia8::Event &);
  void jetAlgorithmInput(const Pythia8::Event &, int);
  void runJetAlgorithm();
  bool matchPartonsToJets(int);
  int  matchPartonsToJetsLight();
  int  matchPartonsToJetsHeavy();
  bool doShowerKtVeto(double pTfirst);

  // Functions to clear and set the jet clustering scales.
  void ClearDJR() { DJR.resize(0);}
  void ClearnME() { nME.resize(0);}

  void SetDJR( const Pythia8::Event& event) {

   // Clear members.
   ClearDJR();

   vector<double> result;

    // Initialize SlowJetDJR jet algorithm with event
    if (!slowJetDJR->setup(event) ) {
      infoPtr->errorMsg("Warning in JetMatchingMadgraph:iGetDJR"
        ": the SlowJet algorithm failed on setup");
      return;
    }

    // Cluster in steps to find all hadronic jets
    while ( slowJetDJR->sizeAll() - slowJetDJR->sizeJet() > 0 ) {
      // Save the next clustering scale.
      result.push_back(slowJetDJR->dNext());
      // Perform step.
      slowJetDJR->doStep();
    }

    // Save clustering scales in reserve order.
    for (int i=int(result.size())-1; i > 0; --i){
//	std::cout<<"Saving DJR "<<i<<" "<<log10(sqrt(result[i]))<<std::endl;
      DJR.push_back(log10(sqrt(result[i])));
	}
  }
  void SetnME() {
	ClearnME();
	vector<int> result;
      	nME.push_back(origTypeIdx[0].size());
	nME.push_back(typeIdx[0].size());
	//std::cout<<"Number of partons "<<origTypeIdx[0].size()<<" "<<typeIdx[0].size()<<std::endl;
  } 

  // Variables.
  vector<int> origTypeIdx[3];
  int    nQmatch;
  double qCut, qCutSq, clFact;
  bool   doFxFx;
  int    nPartonsNow;
  double qCutME, qCutMESq;

  // Vectors to store the jet clustering scales and the number of partons (old and new convention).
  vector<double> DJR;
  vector<int> nME;

};
#endif // end Pythia8_JetMatching_H
