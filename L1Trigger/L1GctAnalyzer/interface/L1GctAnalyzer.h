// -*- C++ -*-
//
// Package:    L1GctAnalyzer
// Class:      L1GctAnalyzer
// 
/**\class L1GctAnalyzer L1GctAnalyzer.cc L1Trigger/L1GctAnalyzer/src/L1GctAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gregory Heath
//         Created:  Wed Aug  9 16:02:54 CEST 2006
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1GctAnalyzer/interface/L1GctBasicHistogrammer.h"
#include "L1Trigger/L1GctAnalyzer/interface/L1GctJetCheckHistogrammer.h"
#include "L1Trigger/L1GctAnalyzer/interface/L1GctMETCheckHistogrammer.h"
//
// class declaration
//

class L1GctAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L1GctAnalyzer(const edm::ParameterSet&);
      ~L1GctAnalyzer();

   private:

      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
  std::string m_histFileName;
  TFile* m_file;
  bool doBasicHist;
  bool doJetCheckHist;
  bool doMETCheckHist;
  L1GctBasicHistogrammer* basicHist;
  std::vector<L1GctJetCheckHistogrammer*> jetCheckHist;
  std::vector<std::string> jetCheckOptions;
  std::vector<L1GctMETCheckHistogrammer*> mETCheckHist;
  std::vector<std::string> mETCheckOptions;
};

