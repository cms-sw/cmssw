// -*- C++ -*-
//
// Package:    DQMServices/Daemon
// Class:      SMDQMSourceExample
// 
/**\class SMDQMSourceExample

  Description: Example DQM Source with multiple top level folders for
  testing in the Storage manager. This started from the DQMSourceExample.cc
  file in DQMServices/Daemon/test, but modified to include another top level
  folder, to remove the 1 sec wait, and to do the fitting without printout.

  $Id: SMDQMSourceExample.cc,v 1.16 2010/08/06 20:24:32 wmtan Exp $

*/


// system include files
#include <memory>
#include <iostream>
#include <math.h>
#include <cstdio>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <TRandom.h> // this is just the random number generator

using std::cout; using std::endl;

//
// class declaration
//

class SMDQMSourceExample : public edm::EDAnalyzer {
public:
   explicit SMDQMSourceExample( const edm::ParameterSet& );
   ~SMDQMSourceExample();
   
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

  virtual void beginJob();

  virtual void endJob(void);

  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
                          edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

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

SMDQMSourceExample::SMDQMSourceExample( const edm::ParameterSet& iConfig )
  : counter(0)
{
  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();
  
  const int NBINS = 500; XMIN = 0; XMAX = 50;

  // book some histograms here

  // create and cd into new folder
  dbe->setCurrentFolder("C1");
  h1 = dbe->book1D("histo", "Example 1D histogram.", NBINS, XMIN, XMAX);
  h2 = dbe->book2D("histo2", "Example 2 2D histogram.", NBINS, XMIN, XMAX, 
		   NBINS, XMIN, XMAX);
  // create and cd into new folder
  dbe->setCurrentFolder("C1/C2");
  h3 = dbe->book1D("histo3", "Example 3 1D histogram.", NBINS, XMIN, XMAX);
  h4 = dbe->book1D("histo4", "Example 4 1D histogram.", NBINS, XMIN, XMAX);
  h5 = dbe->book1D("histo5", "Example 5 1D histogram.", NBINS, XMIN, XMAX);
  h6 = dbe->book1D("histo6", "Example 6 1D histogram.", NBINS, XMIN, XMAX);
  // create and cd into new folder
  dbe->setCurrentFolder("C1/C3");
  const int NBINS2 = 10;
  h7 = dbe->book1D("histo7", "Example 7 1D histogram.", NBINS2, XMIN, XMAX);
  char temp[1024];
  for(int i = 1; i <= NBINS2; ++i)
    {
      sprintf(temp, " bin no. %d", i);
      h7->setBinLabel(i, temp);
    }
  i1 = dbe->bookInt("int1");
  f1 = dbe->bookFloat("float1");
  s1 = dbe->bookString("s1", "my string");

  // create and cd into a new top level folder
  dbe->setCurrentFolder("D1");
  h8 = dbe->book1D("histo8", "Example 8 1D histogram.", NBINS, XMIN, XMAX);
  // create and cd into new sublevel folder
  dbe->setCurrentFolder("D1/D2");
  h9 = dbe->book2D("histo9", "Example 9 2D histogram.", NBINS, XMIN, XMAX, 
		   NBINS, XMIN, XMAX);

  h2->setAxisTitle("Customized x-axis", 1);
  h2->setAxisTitle("Customized y-axis", 2);

  // assign tag to MEs h1, h2 and h7
  const unsigned int detector_id = 17;
  dbe->tag(h1, detector_id);
  dbe->tag(h2, detector_id);
  dbe->tag(h7, detector_id);
  // tag full directory
  dbe->tagContents("C1/C3", detector_id);

  // assign tag to MEs h4 and h6
  const unsigned int detector_id2 = 25;
  const unsigned int detector_id3 = 50;
  dbe->tag(h4, detector_id2);
  dbe->tag(h6, detector_id3);

  // contents of h5 & h6 will be reset at end of monitoring cycle
  h5->setResetMe(true);
  h6->setResetMe(true);
  dbe->showDirStructure();
}


SMDQMSourceExample::~SMDQMSourceExample()
{
  std::cout << "SMDQMSourceExample: "
            << "Destructor called." << std::endl;
   
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  
}

void SMDQMSourceExample::beginJob()
{
  std::cout << "SMDQMSourceExample: "
            << "Doing beginning of job processing." << std::endl;
}

void SMDQMSourceExample::endJob(void)
{
  std::cout << "SMDQMSourceExample: "
            << "Doing end of job processing." << std::endl;
  dbe->save("test.root");  
  dbe->rmdir("C1");
  dbe->rmdir("D1");
}

void SMDQMSourceExample::beginRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  std::cout << "SMDQMSourceExample: "
            << "Doing beginning of run processing for run number "
            <<  run.run() << std::endl;
}

void SMDQMSourceExample::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
                                            edm::EventSetup const& eSetup)
{
  std::cout << "SMDQMSourceExample: "
            << "Doing end of lumi processing for lumi number "
            << lumiSeg.luminosityBlock() << " of run "
            << lumiSeg.run() << std::endl;
}

void SMDQMSourceExample::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  std::cout << "SMDQMSourceExample: "
            << "Doing end of run processing for run number "
            <<  run.run() << std::endl;
  h1->Reset();
  h2->Reset();
  h3->Reset();
  h4->Reset();
  h5->Reset();
  h6->Reset();
  h7->Reset();
  h8->Reset();
  h9->Reset();
  //dbe->rmdir("C1");
  //dbe->rmdir("D1");
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void SMDQMSourceExample::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup )
{   
  i1->Fill(gRandom->Integer(static_cast<int>(XMAX)));
  f1->Fill(-3.14);
 
   // Filling the histogram with random data
  srand( 0 );

  if(counter%1000 == 0)
    std::cout << " # of cycles = " << counter << std::endl;

  for(int i = 0; i != 20; ++i ) 
    {
      float x = gRandom->Uniform(XMAX);
      h1->Fill(x,1./log(x+1));
      h3->Fill(x, 1);
      h4->Fill(gRandom->Gaus(30, 3), 1.0);
      h5->Fill(gRandom->Poisson(15), 0.5);
      h6->Fill(gRandom->Gaus(25, 15), 1.0);
      h7->Fill(gRandom->Gaus(25, 8), 1.0);
      h8->Fill(gRandom->Gaus(25, 7), 1.0);
    }

  // fit h4 to gaussian
  h4->getTH1F()->Fit("gaus","Q");
  
  for ( int i = 0; i != 10; ++i ) 
    {
      float x = gRandom->Gaus(15, 7);
      float y = gRandom->Gaus(20, 5);
      h2->Fill(x,y);
      h9->Fill(x,y);
    }

  ++counter;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SMDQMSourceExample);
