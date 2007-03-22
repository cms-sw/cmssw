// -*- C++ -*-
//
// Package:    L1GctTestAnalyzer
// Class:      L1GctTestAnalyzer
// 
/**\class L1GctTestAnalyzer L1GctTestAnalyzer.cc L1Trigger/GlobalCaloTrigger/test/L1GctTestAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu May 18 16:45:23 CEST 2006
// $Id: L1GctTestAnalyzer.h,v 1.7 2007/03/02 15:49:18 jbrooke Exp $
//
//


// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <iostream>
#include <fstream>

//
// class decleration
//

class L1GctTestAnalyzer : public edm::EDAnalyzer {
public:
  explicit L1GctTestAnalyzer(const edm::ParameterSet&);
  ~L1GctTestAnalyzer();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
  void doRctEM(const edm::Event&, edm::InputTag label);
  void doInternEM(const edm::Event&, edm::InputTag label);
  void doEM(const edm::Event&, edm::InputTag label);
  void doJets(const edm::Event&, edm::InputTag label);
  
private:
  // ----------member data ---------------------------
  
  edm::InputTag rawLabel_;
  edm::InputTag emuLabel_;
  std::string outFilename_;
  std::ofstream outFile_;

  bool doHW_;
  bool doEmu_;
  bool doRctEM_;
  bool doInternEM_;
  bool doEM_;
  bool doJets_;

  unsigned rctEmMinRank_;
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
