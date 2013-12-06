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

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatching.h"
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
class JetMatchingMGFastJet : public JetMatching
{

   public:
   
      JetMatchingMGFastJet(const edm::ParameterSet& params);

      ~JetMatchingMGFastJet() { if (fJetFinder) delete fJetFinder; }
            
      const std::vector<int>* getPartonList() { return typeIdx; }
   
   protected:

      virtual void init( const lhef::LHERunInfo* runInfo );

      virtual bool initAfterBeams();
      virtual void beforeHadronisation( const lhef::LHEEvent* );
      virtual void beforeHadronisationExec() { return; }
      
      virtual int match( const lhef::LHEEvent* partonLevel, const std::vector<fastjet::PseudoJet>* jetInput );

      virtual double getJetEtaMax() const;

   private:
               
      //parameters staff from Madgraph

      template<typename T>
      static T parseParameter(const std::string &value);
      template<typename T>
      static T getParameter(const std::map<std::string, std::string> &params,
                            const std::string &var, const T &defValue = T());
      template<typename T>
      T getParameter(const std::string &var, const T &defValue = T()) const;

      template<typename T>
      static void updateOrDie(
                      const std::map<std::string, std::string> &params,
                      T &param, const std::string &name);

      std::map<std::string, std::string>      mgParams;

      // ----------------------------

      enum vetoStatus { NONE, LESS_JETS, MORE_JETS, HARD_JET, UNMATCHED_PARTON };
      enum partonTypes { ID_TOP=6, ID_GLUON=21, ID_PHOTON=22 };
     
      double qCut, qCutSq;
      double clFact;
      int nQmatch;
      
      // Master switch for merging
      bool   doMerge;

      // Maximum and current number of jets
      int    nJetMax, nJetMin; 

      // Jet algorithm parameters
      int    jetAlgoPower; //  similar to memain_.mektsc ?
      double coneRadius, etaJetMax ;

      // Merging procedure control flag(s)
      // (there're also inclusive, exclusive, and soup/auto in JetMatchingMGFastJet)
      //
      bool fExcLocal; // this is similar to memaev_.iexc

      // Sort final-state of incoming process into light/heavy jets and 'other'
      std::vector < int > typeIdx[3];
            
      bool runInitialized;
      bool soup;
      bool exclusive;

      // 
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
