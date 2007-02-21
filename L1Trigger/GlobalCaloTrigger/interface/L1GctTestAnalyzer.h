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
// $Id: L1GctTestAnalyzer.h,v 1.3 2007/02/20 16:49:58 jbrooke Exp $
//
//


// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
  
  void doRctEM(const edm::Event&, std::string label);
  void doInternEM(const edm::Event&, std::string label);
  void doEM(const edm::Event&, std::string label);
  void doJets(const edm::Event&, std::string label);
  
private:
  // ----------member data ---------------------------
  
  std::string rawLabel_;
  std::string emuLabel_;
  std::string outFilename_;
  std::ofstream outFile_;

  unsigned doRctEM_;
  unsigned doInternEM_;
  unsigned doEM_;
  unsigned doJets_;
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
