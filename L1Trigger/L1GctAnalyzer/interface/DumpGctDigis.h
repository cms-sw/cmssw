// -*- C++ -*-
//
// Package:    DumpGctDigis
// Class:      DumpGctDigis
// 
/**\class DumpGctDigis DumpGctDigis.cc L1Trigger/GlobalCaloTrigger/test/DumpGctDigis.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu May 18 16:45:23 CEST 2006
// $Id: DumpGctDigis.h,v 1.1 2007/07/18 13:14:06 tapper Exp $
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

class DumpGctDigis : public edm::EDAnalyzer {
public:
  explicit DumpGctDigis(const edm::ParameterSet&);
  ~DumpGctDigis();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
  void doRctEM(const edm::Event&, edm::InputTag label);
  void doEM(const edm::Event&, edm::InputTag label);
  void doRegions(const edm::Event&, edm::InputTag label);
  void doJets(const edm::Event&, edm::InputTag label);
  void doInternEM(const edm::Event&, edm::InputTag label);
  void doFibres(const edm::Event&, edm::InputTag label);
  
private:
  // ----------member data ---------------------------
  
  edm::InputTag rawLabel_;
  edm::InputTag emuRctLabel_;
  edm::InputTag emuGctLabel_;
  std::string outFilename_;
  std::ofstream outFile_;

  bool doHW_;
  bool doEmu_;
  bool doRctEM_;
  bool doEM_;
  bool doRegions_;
  bool doJets_;
  bool doInternEM_;
  bool doFibres_;

  unsigned rctEmMinRank_;
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
