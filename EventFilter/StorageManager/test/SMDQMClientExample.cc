// -*- C++ -*-
//
// Package:    EventFilter/StorageManager
// Class:      SMDQMClientExample
// 
/**\class SMDQMClientExample

  Description: Example DQM Client 

  $Id: SMDQMClientExample.cc,v 1.10 2010/08/06 20:24:32 wmtan Exp $

*/


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


using std::cout; using std::endl;

//
// class declaration
//

class SMDQMClientExample : public edm::EDAnalyzer {
public:
   explicit SMDQMClientExample( const edm::ParameterSet& );
   ~SMDQMClientExample();
   
  void analyze( const edm::Event&, const edm::EventSetup& );

  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
                          edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  virtual void endJob(void);

private:
      // ----------member data ---------------------------

  MonitorElement * h1;
  MonitorElement * h2;
  MonitorElement * h3;
  MonitorElement * h4;
  MonitorElement * h5;
  MonitorElement * h6;
  MonitorElement * h7;
  MonitorElement * h8;
  MonitorElement * h9;
  MonitorElement * i1;
  MonitorElement * f1;
  MonitorElement * s1;
  float XMIN; float XMAX;
  // event counter
  int counter;
  // back-end interface
  DQMStore * dbe;

};

SMDQMClientExample::SMDQMClientExample( const edm::ParameterSet& iConfig )
  : counter(0)
{
  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  dbe->showDirStructure();
}


SMDQMClientExample::~SMDQMClientExample()
{
}

void SMDQMClientExample::endJob(void)
{
  dbe->save("outtest.root");  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void SMDQMClientExample::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup )
{   
  std::vector<std::string> topLevelFolderList;
  dbe->cd();
  topLevelFolderList = dbe->getSubdirs();

  std::vector<std::string>::const_iterator dirIter;
  for (dirIter = topLevelFolderList.begin();
       dirIter != topLevelFolderList.end();
       dirIter++) {
    std::string dirName = *dirIter;
    std::cout << "### directory = " << dirName << std::endl;
  }
  dbe->showDirStructure();

  // determine the "global" status of the system
  int status = dbe->getStatus();
  switch(status)
  {
    case dqm::qstatus::ERROR:
      std::cout << " Error(s)";
      break;
    case dqm::qstatus::WARNING:
      std::cout << " Warning(s)";
      break;
    case dqm::qstatus::OTHER:
      std::cout << " Some tests did not run;";
      break; 
    default:
      std::cout << " No problems";
  }
  std::cout << " after getSystemStatus" << std::endl;

  h7 = dbe->get("C1/C3/histo7");
  if(h7) {
    // fit h7 to gaussian
    TH1F * root_ob = h7->getTH1F();
     double int_ent = root_ob->Integral();
     double num_ent = root_ob->GetEntries();
     std::cout << "num entries in h7 = " << num_ent
               << " integral of bin contents = " << int_ent << std::endl;
    root_ob->Fit("gaus");
     root_ob->Reset();
  } else {
    std::cout << "did not find histo7" << std::endl;
  }

  h8 = dbe->get("D1/histo8");
  if(h8) {
    // fit h7 to gaussian
    TH1F * root_ob = h8->getTH1F();
     double int_ent = root_ob->Integral();
     double num_ent = root_ob->GetEntries();
     std::cout << "num entries in h8 = " << num_ent
               << " integral of bin contents = " << int_ent << std::endl;
    root_ob->Fit("gaus");
    //root_ob->Fit("gaus","Q");
     //root_ob->Reset();
  } else {
    std::cout << "did not find histo8" << std::endl;
  }

  i1 = dbe->get("C1/C3/int1");
  if(i1) {
    std::cout << "int1 value std::string: \"" << i1->valueString() << "\"" << std::endl;
  } else {
    std::cout << "did not find int1" << std::endl;
  }

  f1 = dbe->get("C1/C3/float1");
  if(f1) {
    std::cout << "float1 value std::string: \"" << f1->valueString() << "\"" << std::endl;
  } else {
    std::cout << "did not find float1" << std::endl;
  }

  s1 = dbe->get("C1/C3/s1");
  if(s1) {
    std::cout << "s1 value std::string: \"" << s1->valueString() << "\"" << std::endl;
  } else {
    std::cout << "did not find s1" << std::endl;
  }

  ++counter;
}

void SMDQMClientExample::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& eSetup)
{
  std::cout << "Doing end of lumi processing for lumi number "
            << lumiSeg.luminosityBlock() << " of run " 
            << lumiSeg.run() << std::endl;
}

void SMDQMClientExample::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  std::cout << "Doing end of run processing for run number "
            <<  run.run() << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SMDQMClientExample);
