#ifndef gen_JetMatchinhHook_h
#define gen_JetMatchingHook_h

#include <Pythia.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatching.h"

// forward declaration
class Py8toJetInput;

class JetMatchingHook : public Pythia8::UserHooks
{

public:

  JetMatchingHook( const edm::ParameterSet&, Pythia8::Info* );
  virtual ~JetMatchingHook();
  
  //
  // Julia Yarba, Jan.8, 2013
  // The "Early" option will work with Pythia8.170 or higher;
  // for lower versions, please use just VetoPartonLevel
  //
  // virtual bool canVetoPartonLevelEarly() { return true; }  
  // virtual bool doVetoPartonLevelEarly( const Pythia8::Event& event );
  virtual bool canVetoPartonLevel() { return true; }  
  virtual bool doVetoPartonLevel( const Pythia8::Event& event );
    
  void setEventNumber( int ievt ) { fEventNumber = ievt; return ; }
  
  virtual void init( lhef::LHERunInfo* runInfo );
  virtual bool initAfterBeams() { if ( fIsInitialized ) return true; fJetMatching->initAfterBeams(); fIsInitialized=true; return true; }
  void resetMatchingStatus() { fJetMatching->resetMatchingStatus(); return; }
  virtual void beforeHadronization( lhef::LHEEvent* lhee );
  
protected:

  
  JetMatchingHook() : UserHooks() {} 
  
  void setLHERunInfo( lhef::LHERunInfo* lheri ) { 
     fRunBlock=lheri;
     if ( fRunBlock == 0 ) return;
     const lhef::HEPRUP* heprup = fRunBlock->getHEPRUP();
     lhef::CommonBlocks::fillHEPRUP(heprup); 
     return;
  }
  void setLHEEvent( lhef::LHEEvent* lhee ) { 
     fEventBlock=lhee; 
     if ( fEventBlock == 0 ) return;
     const lhef::HEPEUP* hepeup = fEventBlock->getHEPEUP();
     lhef::CommonBlocks::fillHEPEUP(hepeup);
     return;
  }
    
// private:

     lhef::LHERunInfo*       fRunBlock;
     lhef::LHEEvent*         fEventBlock;
     int                     fEventNumber;
     
     Pythia8::Info*          fInfoPtr;
     
     gen::JetMatching*       fJetMatching;
     Py8toJetInput*          fJetInputFill;
          
     //void setJetAlgoInput( const Pythia8::Event& );
     //int getAncestor( int, const Pythia8::Event& );
     
     bool fIsInitialized;
 
};

#endif

#ifndef gen_JetMatchingMG5_h
#define gen_JetMatchingMG5_h

// 
//  Julia V. Yarba, Jan.8, 2013
//
//  This code takes inspirations in the original implemetation 
//  by Steve Mrenna of FNAL (see MG5hooks below), but is structured
//  somewhat differently, and is also using FastJet package instead
//  of Pythia8's native SlowJet
//
//  At this point, we inherit from JetMatchingMadgraph,
//  mainly to use parameters input machinery
//
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"

//
// FastJet package/tools
// Also gives PseudoJet & JetDefinition
//
#include "fastjet/ClusterSequence.hh"  

namespace gen
{
class JetMatchingMG5 : public JetMatchingMadgraph
{

   public:
      JetMatchingMG5(const edm::ParameterSet& params) : JetMatchingMadgraph(params), fJetFinder(0) {}
      ~JetMatchingMG5() { if (fJetFinder) delete fJetFinder; }
            
      const std::vector<int>* getPartonList() { return typeIdx; }
   
   protected:
      virtual void init( const lhef::LHERunInfo* runInfo ) { JetMatchingMadgraph::init(runInfo); initAfterBeams(); return; }
      bool initAfterBeams();
      void beforeHadronisation(const lhef::LHEEvent* );
      void beforeHadronisationExec() { return; }
      
      int match( const lhef::LHEEvent* partonLevel, const std::vector<fastjet::PseudoJet>* jetInput );
   
   private:
         
      void runJetAlgo( const std::vector<fastjet::PseudoJet>* jetInput );
      
      enum vetoStatus { NONE, LESS_JETS, MORE_JETS, HARD_JET, UNMATCHED_PARTON };
      enum partonTypes { ID_TOP=6, ID_GLUON=21, ID_PHOTON=22 };
     
      double qCut, qCutSq;
      int ktScheme;
      double clFact;
      int showerKt;
      int nQmatch;

      // Master switch for merging
      bool   doMerge;

      // Maximum and current number of jets
      int    nJetMax, nJet;

      // Jet algorithm parameters
      int    jetAlgorithm;
      double eTjetMin, coneRadius, etaJetMax, etaJetMaxAlgo;

      int nEta, nPhi;
      double eTthreshold;

      // SlowJet specific
      //
      // NOTE by JVY: we call it slowJetPower but this is actually 
      //              a flag to specify the clustering/matching scheme; 
      //              for example, slowJetPower=1 means kT scheme
      //
      int    slowJetPower;

      // Merging procedure parameters
      int    jetAllow, jetMatch, exclusiveMode;
      double coneMatchLight, coneMatchHeavy;
      bool   exclusive;  

      // Store the minimum eT/pT of matched light jets
      double eTpTlightMin;

      // Sort final-state of incoming process into light/heavy jets and 'other'
      std::vector < int > typeIdx[3];
            
      // --->
      // FastJets tool(s)
      //
      fastjet::JetDefinition* fJetFinder;
      std::vector<fastjet::PseudoJet> fInclusiveJets, fExclusiveJets, fPtSortedJets;
};

} // end namespace

#endif


//
// Julia V. Yarba, Jan.8, 2013
// IMPORTANT NOTE: This code is a slightly modified example,
//                 originally implemented by ** Steve Mrenna of FNAL **,
//                 and meant to be distributed with Pythia8 core code.
//                 Any changes that have been made to the original version
//                 are only infrastructural, in order to fit the code into
//                 CMSSW design, but all algorithms are original, as they
//                 implemented by Steve Mrenna.
//
// MG5hooks.h is a part of the PYTHIA event generator.
// Copyright (C) 2012 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file provides the MG5hooks class to perform MG5 merging.
// Example usage is provided by main32.cc.

#ifndef _MG5HOOKS_H_
#define _MG5HOOKS_H_

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

// Includes
#include "Pythia.h"

using namespace Pythia8;

//==========================================================================

// Settings

// The energy of ghost particles.
#define MG5_GHOSTENERGY   1e-10
// A zero threshold value for double comparisons.
#define MG5_ZEROTHRESHOLD 1e-10
// Debug flag to give verbose information at all stages of the merging.
//mrenna#define MG5_DEBUG         false
#define MG5_DEBUG         false

static int evtNumberSoFar = 0;

//==========================================================================
/* Is needed only in a piece of code that's commented out.
   I take it out for the moment since it's causing linking problems, don't know why...
typedef std::pair<double,int> mypair;
bool comparator ( const mypair& l, const mypair& r)
   { return l.first > r.first; }
*/

   
extern "C" {

	extern struct UPPRIV {
		int	lnhin, lnhout;
		int	mscal, ievnt;
		int	ickkw, iscale;
	} uppriv_;

	extern struct MEMAIN {
		double 	etcjet, rclmax, etaclmax, qcut, showerkt, clfact;
		int	maxjets, minjets, iexcfile, ktsche;
		int	mektsc,nexcres, excres[30];
      		int	nqmatch,excproc,iexcproc[1000],iexcval[1000];
                bool    nosingrad,jetprocs;
	} memain_;

}
   
   

// Declaration of main MG5hooks class to perform MG5 matching.

// class MG5hooks : virtual public UserHooks {
class MG5hooks : virtual public JetMatchingHook {

public:
  // Constructor and destructor
  MG5hooks() : cellJet(NULL), slowJet(NULL) {}
  MG5hooks( const edm::ParameterSet& ps, Info* info ) : JetMatchingHook( ps, info ), 
                                                        cellJet(NULL), slowJet(NULL), 
                                                        // fInfoPtr(info), 
							// fJetMatching( new gen::JetMatchingMadgraph(ps) ),
							process(-1) {}  // process will have to go away after debugging is done !!!
   
  ~MG5hooks() {
    if (cellJet) delete cellJet;
    if (slowJet) delete slowJet;
  }

  // Initialisation
  virtual bool initAfterBeams();

  // Process level vetos
  virtual bool canVetoProcessLevel();
  virtual bool doVetoProcessLevel(Event &);

  // Parton level vetos (before beam remnants and resonance decays)
  // virtual bool canVetoPartonLevelEarly();
  // virtual bool doVetoPartonLevelEarly(const Event &);
  virtual bool canVetoPartonLevel();
  virtual bool doVetoPartonLevel(const Event &);
  
  void init( lhef::LHERunInfo* runInfo ) { fJetMatching->init( runInfo ); initAfterBeams(); return; }
  // void resetMatchingStatus() { fJetMatching->resetMatchingStatus(); return; }
  void beforeHadronization( lhef::LHEEvent* lhee ) { const lhef::HEPEUP& hepeup = *(lhee->getHEPEUP());
                                                     process =  hepeup.IDPRUP; // FIXME !!!
                                                     return; }

private:

  enum vetoStatus { NONE, LESS_JETS, MORE_JETS, HARD_JET, UNMATCHED_PARTON };
      enum partonTypes { ID_TOP=6, ID_GLUON=21, ID_PHOTON=22 };
  // Different steps of the MG5 matching algorithm
  void sortIncomingProcess(const Event &);
  void jetAlgorithmInput(const Event &, int);
  void runJetAlgorithm();
  bool matchPartonsToJets(int);
  int  matchPartonsToJetsLight();
  int  matchPartonsToJetsHeavy();

  // DeltaR between two 4-vectors (eta and y variants)
  inline double Vec4eta(const Vec4 &pIn) {
    return -log(tan(pIn.theta() / 2.));
  }
  inline double Vec4y(const Vec4 &pIn) {
    return 0.5 * log((pIn.e() + pIn.pz()) / (pIn.e() - pIn.pz()));
  }
  inline double deltaReta(const Vec4 &p1, const Vec4 &p2) {
    double dEta = abs(Vec4eta(p1) - Vec4eta(p2));
    double dPhi = abs(p1.phi() - p2.phi());
    if (dPhi > M_PI) dPhi = 2. * M_PI - dPhi;
    return sqrt(dEta*dEta + dPhi*dPhi);
  }
  inline double deltaRy(const Vec4 &p1, const Vec4 &p2) {
    double dy   = abs(Vec4y(p1) - Vec4y(p2));
    double dPhi = abs(p1.phi() - p2.phi());
    if (dPhi > M_PI) dPhi = 2. * M_PI - dPhi;
    return sqrt(dy*dy + dPhi*dPhi);
  }

  // Function to sort typeIdx vectors into descending eT/pT order.
  // Uses a selection sort, as number of partons generally small
  // and so efficiency not a worry.
  void sortTypeIdx(vector < int > &vecIn) {
    for (size_t i = 0; i < vecIn.size(); i++) {
      size_t jMax = i;
      double vMax = (jetAlgorithm == 1) ?
                    eventProcess[vecIn[i]].eT() :
                    eventProcess[vecIn[i]].pT();
      for (size_t j = i + 1; j < vecIn.size(); j++) {
        double vNow = (jetAlgorithm == 1) ?
                      eventProcess[vecIn[j]].eT() :
                      eventProcess[vecIn[j]].pT();
        if (vNow > vMax) {
          vMax = vNow;
          jMax = j;
        }
      }
      if (jMax != i) swap(vecIn[i], vecIn[jMax]);
    }
  }

  double qCut, qCutSq;
  int ktScheme;
  double clFact;
  int showerKt;
  int nQmatch;

  // Master switch for merging
  bool   doMerge;

  // Maximum and current number of jets
  int    nJetMax, nJet;

  // Jet algorithm parameters
  int    jetAlgorithm;
  double eTjetMin, coneRadius, etaJetMax, etaJetMaxAlgo;

  int nEta, nPhi;
  double eTthreshold;

  // SlowJet specific
  int    slowJetPower;

  // Merging procedure parameters
  int    jetAllow, jetMatch, exclusiveMode;
  double coneMatchLight, coneMatchHeavy;
  bool   exclusive;

  // Event records to store original incoming process, final-state of the
  // incoming process and what will be passed to the jet algorithm.
  // Not completely necessary to store all steps, but makes tracking the
  // steps of the algorithm a lot easier.
  Event eventProcessOrig, eventProcess, workEventJet;

  // Internal jet algorithms
  CellJet *cellJet;
  SlowJet *slowJet;
  
  // Sort final-state of incoming process into light/heavy jets and 'other'
  std::vector < int > typeIdx[3];
  std::set    < int > typeSet[3];

  // Momenta output of jet algorithm (to provide same output regardless of
  // the selected jet algorithm)
  std::vector < Vec4 > jetMomenta;

  // Store the minimum eT/pT of matched light jets
  double eTpTlightMin;
  
  // Note: Info == Pythia8::Info but we're 'using' Pythia8 namespace
//  Info*                   fInfoPtr;
//  gen::JetMatching*       fJetMatching;
    int process;
  
};

//==========================================================================

// Main implementation of MG5hooks class. This may be split out to a
// separate C++ file if desired, but currently included here for ease
// of use.

// Initialisation routine automatically called from Pythia::init().
// Setup all parts needed for the merging.
inline bool
MG5hooks::initAfterBeams() {


   if ( fIsInitialized ) return true; // make sure it's done only once 

// Read in MG5 specific configuration variables
/*   bool setMG5    = settingsPtr->flag("MG5:setMG5"); */
   // at present, not in use 
   // -> bool setMG5 = false;

   // If ALPGEN parameters are present, then parse in MG5Par object
/*
   MG5Par par(infoPtr);
   string parStr = infoPtr->header("MGRunCard");
   if (!parStr.empty()) {
      par.parse(parStr);
      par.printParams();
   }
*/

   // Set MG5 merging parameters from the file if requested 
/*
   if (setMG5) 
   {

      doMerge = par.getParam("ickkw");

      if (par.haveParam("qcut") && par.haveParam("nqmatch") && par.haveParam("etaclmax")) {

         qCut = par.getParam("qcut");
         nQmatch = par.getParamAsInt("nqmatch");
         etaJetMax = par.getParam("etaclmax");

         // Warn if setMLM requested, but parameters not present
      } else {
         infoPtr->errorMsg("Warning in MG5Hooks:init: "
                           "no MG5 merging parameters found");
      }
   } 
   else 
   {
      doMerge        = settingsPtr->flag("MG5:merge");
      qCut           = settingsPtr->parm("MG5:qCut");
      nQmatch        = settingsPtr->mode("MG5:nQmatch");
      clFact         = settingsPtr->parm("MG5:clFact");

   }
*/

      doMerge        = uppriv_.ickkw;
      qCut           = memain_.qcut;
      nQmatch        = memain_.nqmatch;
      clFact         = 0;


  // Read in parameters

/*
  nJet           = settingsPtr->mode("MG5:nJet");
  nJetMax        = settingsPtr->mode("MG5:nJetMax");
  jetAlgorithm   = settingsPtr->mode("MG5:jetAlgorithm");
  etaJetMax      = settingsPtr->parm("MG5:etaJetMax");
  coneRadius     = settingsPtr->parm("MG5:coneRadius");
  slowJetPower   = settingsPtr->mode("MG5:slowJetPower");
  eTjetMin       = settingsPtr->parm("MG5:pTjetMin");
*/
  nJet           = memain_.minjets;
  nJetMax        = memain_.maxjets;
  //
  // this is the kT algorithm !!!
  // (in SlowJets it's defined by the slowJetPower=1)
  //
  jetAlgorithm   = 2;
  etaJetMax      = memain_.etaclmax;
  coneRadius     = 1.0;
  slowJetPower   = 1;
  eTjetMin       = 20;


  // Matching procedure
/*
  jetAllow       = settingsPtr->mode("MG5:jetAllow");
  jetMatch       = settingsPtr->mode("MG5:jetMatch");
*/
//  coneMatchLight = settingsPtr->parm("MG5:coneMatchLight");
//  coneMatchHeavy = settingsPtr->parm("MG5:coneMatchHeavy");
/*  exclusiveMode  = settingsPtr->mode("MG5:exclusive"); */
   jetMatch = 0; // hardcoded as it is in Steve's ecample/cmd
   exclusiveMode = 1;

/*  ktScheme       = settingsPtr->mode("MG5:ktScheme"); */
  ktScheme       = memain_.mektsc;


  qCutSq         = pow(qCut,2);
  etaJetMaxAlgo  = etaJetMax;


  // If not merging, then done
  if (!doMerge) return true;

  // Exclusive mode; if set to 2, then set based on nJet/nJetMax
  if (exclusiveMode == 2) {

    // No nJet or nJetMax, so default to exclusive mode
    if (nJet < 0 || nJetMax < 0) {
      infoPtr->errorMsg("Warning in MG5hooks:init: "
          "missing jet multiplicity information; running in exclusive mode");
      exclusive = true;

    // Inclusive if nJet == nJetMax, exclusive otherwise
    } else {
      exclusive = (nJet == nJetMax) ? false : true;
    }

  // Otherwise, just set as given
  } else {
    exclusive = (exclusiveMode == 0) ? false : true;
  }

  // Initialise chosen jet algorithm. CellJet.
  if (jetAlgorithm == 1) {

    // Extra options for CellJet. nSel = 1 means that all final-state
    // particles are taken and we retain control of what to select.
    // smear/resolution/upperCut are not used and are set to default values.
    int    nSel = 2, smear = 0;
    double resolution = 0.5, upperCut = 2.;
    cellJet = new CellJet(etaJetMaxAlgo, nEta, nPhi, nSel,
                          smear, resolution, upperCut, eTthreshold);

  // SlowJet
  } else if (jetAlgorithm == 2) {

    // this is basically the MadGraph one !
    //
    slowJet = new SlowJet(slowJetPower, coneRadius, eTjetMin, etaJetMaxAlgo);        
  }

  // Check the jetMatch parameter; option 2 only works with SlowJet
  if (jetAlgorithm == 1 && jetMatch == 2) {
    infoPtr->errorMsg("Warning in MG5hooks:init: "
        "jetMatch = 2 only valid with SlowJet algorithm. "
        "Reverting to jetMatch = 1.");
    jetMatch = 1;
  }

  // Setup local event records
  eventProcessOrig.init("(eventProcessOrig)", particleDataPtr);
  eventProcess.init("(eventProcess)", particleDataPtr);
  workEventJet.init("(workEventJet)", particleDataPtr);

  // Print information
  string jetStr  = (jetAlgorithm ==  1) ? "CellJet" :
                   (slowJetPower == -1) ? "anti-kT" :
                   (slowJetPower ==  0) ? "C/A"     :
                   (slowJetPower ==  1) ? "kT"      : "unknown";
  string modeStr = (exclusive)         ? "exclusive" : "inclusive";

  cout << endl
       << " *-------  MG5 matching parameters  -------*" << endl
       << " |  qCut                |  " << setw(14)
       << qCut << "  |" << endl
       << " |  nQmatch             |  " << setw(14)
       << nQmatch << "  |" << endl
       << " |  clFact              |  " << setw(14)
       << clFact << "  |" << endl
       << " |  Jet algorithm       |  " << setw(14)
       << jetStr << "  |" << endl
       << " |  eTjetMin            |  " << setw(14)
       << eTjetMin << "  |" << endl
       << " |  etaJetMax           |  " << setw(14)
       << etaJetMax << "  |" << endl
       << " |  jetAllow            |  " << setw(14)
       << jetAllow << "  |" << endl
       << " |  jetMatch            |  " << setw(14)
       << jetMatch << "  |" << endl
       << " |  Mode                |  " << setw(14)
       << modeStr << "  |" << endl
       << " *-----------------------------------------*" << endl;

  fIsInitialized=true;
  return true;

}

// Process level veto. Stores incoming event for later.
inline bool
MG5hooks::canVetoProcessLevel() { return doMerge; }

inline bool
MG5hooks::doVetoProcessLevel(Event& process) { 
  eventProcessOrig = process;
  return false;
}

// Early parton level veto (before beam remnants and resonance showers)
inline bool
MG5hooks::canVetoPartonLevel() { return doMerge; }
// MG5hooks::canVetoPartonLevelEarly() { return doMerge; }

inline bool
MG5hooks::doVetoPartonLevel(const Event& event) {
// MG5hooks::doVetoPartonLevelEarly(const Event& event) {
  // 1) Sort the original incoming process. After this step is performed,
  //    the following assignments have been made:
  //      eventProcessOrig - the original incoming process
  //      eventProcess     - the final-state of the incoming process with
  //                         resonance decays removed (and resonances
  //                         themselves now with positive status code)
  //      typeIdx[0/1/2]   - Indices into 'eventProcess' of
  //                         light jets/heavy jets/other
  //      typeSet[0/1/2]   - Indices into 'event' of light jets/heavy jets/other
  //      workEvent        - partons from the hardest subsystem + ISR + FSR only
  sortIncomingProcess(event);
  
  
  // Debug
  if (MG5_DEBUG) {
    // Begin
    cout << endl << "---------- Begin MG5 Debug ----------" << endl;

    // Original incoming process
    cout << endl << "Original incoming process:";
    eventProcessOrig.list();

    // Final-state of original incoming process
    cout << endl << "Final-state incoming process:";

    // List categories of sorted particles
    for (size_t i = 0; i < typeIdx[0].size(); i++) 
      cout << ((i == 0) ? "Light jets: " : ", ")   << setw(3) << typeIdx[0][i];
    for (size_t i = 0; i < typeIdx[1].size(); i++) 
      cout << ((i == 0) ? "\nHeavy jets: " : ", ") << setw(3) << typeIdx[1][i];
    for (size_t i = 0; i < typeIdx[2].size(); i++) 
      cout << ((i == 0) ? "\nOther:      " : ", ") << setw(3) << typeIdx[2][i];

    // Full event at this stage
    cout << endl << endl << "Event:";
    event.list();

    // Work event (partons from hardest subsystem + ISR + FSR)
    cout << endl << "Work event:";
    workEvent.list();
  }

  // 2) Light/heavy jets: iType = 0 (light jets), 1 (heavy jets)
  int iTypeEnd = (typeIdx[1].empty()) ? 1 : 2;
  for (int iType = 0; iType < iTypeEnd; iType++) {

    // 2a) Find particles which will be passed from the jet algorithm.
    //     Input from 'workEvent' and output in 'workEventJet'.
    jetAlgorithmInput(event, iType);
    
    // Debug
    if (MG5_DEBUG) {
      // Jet algorithm event
      cout << endl << "Jet algorithm event (iType = " << iType << "):";
      workEventJet.list();
    }

    // 2b) Run jet algorithm on 'workEventJet'.
    //     Output is stored in jetMomenta.
    runJetAlgorithm();

    // 2c) Match partons to jets and decide if veto is necessary
    if (matchPartonsToJets(iType) == true) {
      // Debug
      if (MG5_DEBUG) {
        cout << endl << "Event vetoed" << endl
             << "----------  End MG5 Debug  ----------" << endl;
      }
      return true;
    }
  }

  // Debug
  if (MG5_DEBUG) {
    cout << endl << "Event accepted" << endl
         << "----------  End MG5 Debug  ----------" << endl;
  }

  // If we reached here, then no veto
  return false;
}


// Step (1): sort the incoming particles
inline void
MG5hooks::sortIncomingProcess(const Event &event) {
  // Remove resonance decays from original process and keep only final
  // state. Resonances will have positive status code after this step.
  
  omitResonanceDecays(eventProcessOrig); //  --> UNDO FOR PY8.170 OR HIGHER !!! , true); 
  eventProcess = workEvent;

  // Sort original process final state into light/heavy jets and 'other'.
  // Criteria:
  //   1 <= ID <= 5 and massless, or ID == 21 --> light jet (typeIdx[0])
  //   4 <= ID <= 6 and massive               --> heavy jet (typeIdx[1])
  //   All else                               --> other     (typeIdx[2])
  // Note that 'typeIdx' stores indices into 'eventProcess' (after resonance
  // decays are omitted), while 'typeSet' stores indices into the original
  // process record, 'eventProcessOrig', but these indices are also valid
  // in 'event'.
  for (int i = 0; i < 3; i++) {
    typeIdx[i].clear();
    typeSet[i].clear();
  }
  for (int i = 0; i < eventProcess.size(); i++) {
    // Ignore nonfinal and default to 'other'
    if (!eventProcess[i].isFinal()) continue;
    int idx = 2;

    // Light jets
    if (eventProcess[i].id() == ID_GLUON || (eventProcess[i].idAbs() <= nQmatch) )
      idx = 0;

    // Heavy jets
    else if (eventProcess[i].idAbs() > nQmatch && eventProcess[i].idAbs() <= ID_TOP)
      idx = 1;

    // Store
    typeIdx[idx].push_back(i);
    typeSet[idx].insert(eventProcess[i].daughter1());
  }


/*  vector < int > typeIdxTemp[3];
  for(size_t idx=0; idx<3; ++idx) {
    // no need to sort
    if( typeIdx[idx].size() < 2 ) continue;
    vector<pair<double,int> > sortPair;
    set<int>::const_iterator sit = typeSet[idx].begin();
    for(size_t j=0; j<typeIdx[idx].size(); ++j) {
      typeIdxTemp[idx].push_back(typeIdx[idx][j]);
      mypair tempPair(eventProcess[j].pT2(),j);
      sortPair.push_back(tempPair);
      cout << *sit << endl; sit++;
    }
    sort(sortPair.begin(),sortPair.end(),comparator);
    for(size_t j=0; j<sortPair.size(); ++j) {
      size_t jTemp = sortPair[j].second;
      typeIdx[idx][j] = typeIdxTemp[idx][jTemp];
    }
    } */


  // Extract partons from hardest subsystem + ISR + FSR only into
  // workEvent. Note no resonance showers or MPIs.
  subEvent(event);
}


// Step (2a): pick which particles to pass to the jet algorithm
inline void
MG5hooks::jetAlgorithmInput(const Event &event, int iType) {
  // Take input from 'workEvent' and put output in 'workEventJet'
  workEventJet = workEvent;

  // Loop over particles and decide what to pass to the jet algorithm
  for (int i = 0; i < workEventJet.size(); ++i) {
    if (!workEventJet[i].isFinal()) continue;

    // jetAllow option to disallow certain particle types
    if (jetAllow == 1) {

      // Original AG+Py6 algorithm explicitly excludes tops,
      // leptons and photons.
      int id = workEventJet[i].idAbs();
      if ((id >= 11 && id <= 16) || id == ID_TOP || id == ID_PHOTON) {
        workEventJet[i].statusNeg();
        continue;
      }
    }

    // Get the index of this particle in original event
    int idx = workEventJet[i].daughter1();

    // Start with particle idx, and afterwards track mothers
    while (true) {

      // Light jets
      if (iType == 0) {

        // Do not include if originates from heavy jet or 'other'
        if (typeSet[1].find(idx) != typeSet[1].end() ||
            typeSet[2].find(idx) != typeSet[2].end()) {
          workEventJet[i].statusNeg();
          break;
        }

        // Made it to start of event record so done
        if (idx == 0) break;
        // Otherwise next mother and continue
        idx = event[idx].mother1();

      // Heavy jets
      } else if (iType == 1) {

        // Only include if originates from heavy jet
        if (typeSet[1].find(idx) != typeSet[1].end()) break;

        // Made it to start of event record with no heavy jet mother,
        // so DO NOT include particle
        if (idx == 0) {
          workEventJet[i].statusNeg();
          break;
        }

        // Otherwise next mother and continue
        idx = event[idx].mother1();

      } // if (iType)
    } // while (true)
  } // for (i)

  // For jetMatch = 2, insert ghost particles corresponding to
  // each hard parton in the original process
  if (jetMatch > 0) {
    for (int i = 0; i < int(typeIdx[iType].size()); i++) {
      // Get y/phi of the parton
      Vec4   pIn = eventProcess[typeIdx[iType][i]].p();
      double y   = Vec4y(pIn);
      double phi = pIn.phi();

      // Create a ghost particle and add to the workEventJet
      double e   = MG5_GHOSTENERGY;
      double e2y = exp(2. * y);
      double pz  = e * (e2y - 1.) / (e2y + 1.);
      double pt  = sqrt(e*e - pz*pz);
      double px  = pt * cos(phi);
      double py  = pt * sin(phi);
      workEventJet.append(Particle(ID_GLUON, 99, 0, 0, 0, 0, 0, 0,
                                px, py, pz, e, 0., 0, 9.));

      // Extra check on reconstructed y/phi values. If many warnings
      // of this type, MLM_GHOSTENERGY may be set too low.
#ifdef MG5_CHECK
      int lastIdx = workEventJet.size() - 1;
      if (abs(y   - workEventJet[lastIdx].y())   > MG5_ZEROTHRESHOLD ||
          abs(phi - workEventJet[lastIdx].phi()) > MG5_ZEROTHRESHOLD)
        infoPtr->errorMsg("Warning in MG5hooks:jetAlgorithmInput: "
            "ghost particle y/phi mismatch");
#endif

    } // for (i)
  } // if (jetMatch == 2)
}


// Step (2b): run jet algorithm and provide common output
inline void
MG5hooks::runJetAlgorithm() {
   return; 
}



// Step (2c): veto decision (returning true vetoes the event)
inline bool
MG5hooks::matchPartonsToJets(int iType) {
  // Use two different routines for light/heavy jets as
  // different veto conditions and for clarity
  if (iType == 0) return (matchPartonsToJetsLight() > 0);
  else            return (matchPartonsToJetsHeavy() > 0);
}

// Step(2c): light jets
// Return codes are given indicating the reason for a veto.
// Although not currently used, they are a useful debugging tool:
//   0 = no veto
//   1 = veto as number of jets less than number of partons
//   2 = veto as exclusive mode and number of jets greater than
//       number of partons
//   3 = veto as inclusive mode and there would be an extra jet
//       that is harder than any matched soft jet
//   4 = veto as there is a parton which does not match a jet
inline int
MG5hooks::matchPartonsToJetsLight() {

   evtNumberSoFar++;
  
  // Number of hard partons
  int nParton = typeIdx[0].size();

  if( !slowJet->setup(workEventJet) ) return 99;

  double localQcutSq = qCutSq;

  double dOld = -1;
  while( slowJet->sizeAll()-slowJet->sizeJet() >0 ) {

     if( slowJet->dNext() > localQcutSq ) break;
     dOld = slowJet->dNext();
     slowJet->doStep();
  }

  int nJets = slowJet->sizeJet();
  int nClus = slowJet->sizeAll();


  if(MG5_DEBUG) {

   for(int i=nJets; i<nClus; ++i) {
      printf("%d %f %f %f %f | %f \n",i,slowJet->pT(i),slowJet->y(i),slowJet->phi(i),
             slowJet->p(i).e(),slowJet->y(i));
   }
  }
  
  int nCLjets = nClus - nJets;

  if( nCLjets < nParton ) return LESS_JETS;

  
  if( exclusive ) 
  {

     if( nCLjets > nParton ) return MORE_JETS;

  }   
  else // inclusive ???
  {

    localQcutSq = dOld;
    if( !slowJet->setup(workEventJet) ) return 99;
    while( slowJet->sizeAll()-slowJet->sizeJet() >nParton ) 
    {
      slowJet->doStep();
    }


    if( clFact >= 0 ) 
    {
       vector<double> partonPt;
       for(int i=0; i<nParton; ++i) partonPt.push_back(eventProcess[typeIdx[0][i]].pT2());
       sort(partonPt.begin(),partonPt.end());
       localQcutSq = max(qCutSq, partonPt[0]);
    }
    nJets = slowJet->sizeJet();
    nClus = slowJet->sizeAll();

  }

  if( clFact != 0 ) localQcutSq *= pow(clFact,2);

/*
  if ( process > 1 )
  {     
     std::cout << " process = " << process << std::endl;
  }
*/

  Event tempEvent;
  tempEvent.init("(tempEvent)", particleDataPtr);
  int nPass=0;
  double pTminEstimate=-1;
  for(int i=nJets; i<nClus; ++i) {
     tempEvent.append( Particle(ID_GLUON, 99, 0, 0, 0, 0, 0, 0,
                                slowJet->p(i).px(), 
                                slowJet->p(i).py(), 
                                slowJet->p(i).pz(), 
                                slowJet->p(i).e(), 0., 0, 9.) );
     nPass++;
     pTminEstimate = max(pTminEstimate, slowJet->pT(i));
     if(nPass == nParton) break;
  }
  size_t tempSize = tempEvent.size();
  vector < bool > jetAssigned;
  jetAssigned.assign(tempSize, false);

  int iNow = 0;

  while( iNow < nParton ) 
  {

     Event tempEventJet;
     tempEventJet.init("(tempEventJet)", particleDataPtr);
     for(size_t i=0; i<tempSize; ++i) 
     {
        if(jetAssigned[i]) continue;
        Vec4 pIn = tempEvent[i].p();
        tempEventJet.append( Particle(ID_GLUON, 98, 0, 0, 0, 0, 0, 0,
                                   pIn.px(), 
                                   pIn.py(), 
                                   pIn.pz(), 
                                   pIn.e(), 0., 0, 9.) );
     }
     Vec4   pIn = eventProcess[typeIdx[0][iNow]].p();
     tempEventJet.append( Particle(ID_GLUON, 99, 0, 0, 0, 0, 0, 0,
                                   pIn.px(), 
                                   pIn.py(), 
                                   pIn.pz(), 
                                   pIn.e(), 0., 0, 9.) );
//    tempEventJet.list();

     if( !slowJet->setup(tempEventJet) ) return 100;

     int inxt = slowJet->iNext();
     int jnxt = slowJet->jNext();
     double dnxt = slowJet->dNext();
     
//     if( slowJet->iNext() == tempEventJet.size()-1 && slowJet->jNext() >-1 && slowJet->dNext() < localQcutSq ) 
     if( inxt == tempEventJet.size()-1 && jnxt >-1 && dnxt < localQcutSq ) 
     {
        int iKnt = -1;
        for(size_t i=0; i!=tempSize; ++i) 
	{
           if(jetAssigned[i]) continue;
           iKnt++;
           if( iKnt == slowJet->jNext() ) jetAssigned[i]=true;
        }
     } 
     else { return UNMATCHED_PARTON; }
//     iNow--;
     iNow++;
     
  }


  // Minimal eT/pT (CellJet/SlowJet) of matched light jets. Needed
  // later for heavy jet vetos in inclusive mode.

//mrenna check this now
  if (nParton > 0 && pTminEstimate>0)
     eTpTlightMin = pTminEstimate;
  else
    eTpTlightMin = -1.;

  // No veto
  return NONE;
}

// Step(2c): heavy jets
// Return codes are given indicating the reason for a veto.
// Although not currently used, they are a useful debugging tool:
//   0 = no veto as there are no extra jets present
//   1 = veto as in exclusive mode and extra jets present
//   2 = veto as in inclusive mode and extra jets were harder
//       than any matched light jet
inline int
MG5hooks::matchPartonsToJetsHeavy() {
  // Currently, heavy jets are unmatched
  // If there are no extra jets, then accept
  if (jetMomenta.empty()) return NONE;

/*  // Sort partons by eT/pT
  sortTypeIdx(typeIdx[1]);

  // Number of hard partons
  int nParton = typeIdx[1].size(); */


  // No extra jets were present so no veto
  return NONE;
}

#endif // _MG5HOOKS_H_
