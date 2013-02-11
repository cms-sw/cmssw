#ifndef gen_JetMatchingMGFastJet_h
#define gen_JetMatchingMGFastJet_h


// 
//  Julia V. Yarba, Feb.10, 2013
//
//  This code takes inspirations in the original implemetation 
//  by Steve Mrenna of FNAL (example main32), but is structured
//  somewhat differently, and is also using FastJet package 
//  instead of Pythia8's native SlowJet
//
//  At this point, we inherit from JetMatchingMadgraph,
//  mainly to use parameters input machinery
//

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

//
// FastJet package/tools
// Also gives PseudoJet & JetDefinition
//
#include "fastjet/ClusterSequence.hh"  

#include <iostream>
#include <fstream>

namespace gen
{
class JetMatchingMGFastJet : public JetMatchingMadgraph
{

   public:
      JetMatchingMGFastJet(const edm::ParameterSet& params) : JetMatchingMadgraph(params), 
                                                              fJetFinder(0),  
							      fIsInit(false) 
							      { fDJROutFlag = params.getParameter<int>("outTree_flag"); }
      ~JetMatchingMGFastJet() { if (fJetFinder) delete fJetFinder; }
            
      const std::vector<int>* getPartonList() { return typeIdx; }
   
   protected:
      virtual void init( const lhef::LHERunInfo* runInfo ) { if (fIsInit) return; JetMatchingMadgraph::init(runInfo); 
                                                             initAfterBeams(); fIsInit=true; return; }
      bool initAfterBeams();
      void beforeHadronisation(const lhef::LHEEvent* );
      void beforeHadronisationExec() { return; }
      
      int match( const lhef::LHEEvent* partonLevel, const std::vector<fastjet::PseudoJet>* jetInput );
   
   private:
               
      enum vetoStatus { NONE, LESS_JETS, MORE_JETS, HARD_JET, UNMATCHED_PARTON };
      enum partonTypes { ID_TOP=6, ID_GLUON=21, ID_PHOTON=22 };
     
      double qCut, qCutSq;
      double clFact;
      int nQmatch;
      //
      // these 2 below are legacy but keep here until furher notice
      //
      //int ktScheme;
      //int showerKt;

      // Master switch for merging
      bool   doMerge;

      // Maximum and current number of jets
      int    nJetMax, nJetMin; 

      // Jet algorithm parameters
      int    jetAlgorithm;
      double eTjetMin, coneRadius, etaJetMax, etaJetMaxAlgo;

      // SlowJet specific
      //
      // NOTE by JVY: we call it slowJetPower but this is actually 
      //              a flag to specify the clustering/matching scheme; 
      //              for example, slowJetPower=1 means kT scheme
      //
      int    slowJetPower;

      // Merging procedure parameters
      int jetAllow; 
      int exclusiveMode; 
      // bool   exclusive;  // can NOT use this name here because it's in the JetMatchingMadgraph
      bool fExcLocal;

      // Sort final-state of incoming process into light/heavy jets and 'other'
      std::vector < int > typeIdx[3];
            
      // --->
      // FastJets tool(s)
      //
      fastjet::JetDefinition* fJetFinder;
      std::vector<fastjet::PseudoJet> fClusJets, fPtSortedJets;      
      
      // output for DJR analysis
      //
      std::ofstream            fDJROutput;   
      int                      fDJROutFlag;     

      bool fIsInit;

};

} // end namespace

#endif
