// -*- C++ -*-
//
// Package:    DQMServices/Examples
// Class:      DQMSourceExample
// 
/**\class DQMSourceExample

Description: Simple example showing how to create a DQM Source creating & shipping
monitoring elements

Implementation:
<Notes on implementation>
*/
//
//
//
#include "DQMServices/Examples/interface/DQMSourceExample.h"

#include <TRandom.h> // this is just the random number generator

using namespace std;
using namespace edm;

//
// constructors and destructor
//
DQMSourceExample::DQMSourceExample( const edm::ParameterSet& ps )
  : DQMAnalyzer(ps)
{

/// use this to read in reference histograms from file
//  dbe->readReferenceME("ref_test.root");
//  dbe->open("ref_test.root",false,"","prep");

// use this to collate histograms from files
//  dbe->open("test1.root",true,"","Collate");
//  dbe->open("test2.root",true,"","Collate");

/// use this to retrieve CMSSW version of file
//  cout << dbe->getFileReleaseVersion("ref_test.root") << endl;


/// book some histograms here
  const int NBINS = 50; XMIN = 0; XMAX = 50;

  // create and cd into new folder
  dbe->setCurrentFolder(PSrootFolder+"C1");
  h1 = dbe->book1D("histo", "Example 1D histogram.", NBINS, XMIN, XMAX);
  h2 = dbe->book2D("histo2", "Example 2 2D histogram.", NBINS, XMIN, XMAX, 
		   NBINS, XMIN, XMAX);
  p1 = dbe->bookProfile("prof1","my profile",NBINS,XMIN,XMAX,NBINS,XMIN,XMAX,"");
  // create and cd into new folder
  dbe->setCurrentFolder(PSrootFolder+"C1/C2");
  h3 = dbe->book1D("histo3", "Example 3 1D histogram.", NBINS, XMIN, XMAX);
  h4 = dbe->book1D("histo4", "Example 4 1D histogram.", NBINS, XMIN, XMAX);
  h5 = dbe->book1D("histo5", "Example 5 1D histogram.", NBINS, XMIN, XMAX);
  h6 = dbe->book1D("histo6", "Example 6 1D histogram.", NBINS, XMIN, XMAX);
  // create and cd into new folder
  dbe->setCurrentFolder(PSrootFolder+"C1/C3");
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

  h2->setAxisTitle("Customized x-axis", 1);
  h2->setAxisTitle("Customized y-axis", 2);

  // assign tag to MEs h1, h2 and h7
  const unsigned int detector_id = 17;
  dbe->tag(h1, detector_id);
  dbe->tag(h2, detector_id);
  dbe->tag(h7, detector_id);
  // tag full directory
  dbe->tagContents(PSrootFolder+"C1/C3", detector_id);

  // assign tag to MEs h4 and h6
  const unsigned int detector_id2 = 25;
  const unsigned int detector_id3 = 50;
  dbe->tag(h4, detector_id2);
  dbe->tag(h6, detector_id3);

  // contents of h5 & h6 will be reset at end of monitoring cycle
  h5->setResetMe(true);
  h6->setResetMe(true);
  dbe->showDirStructure();
  
  
/// test referenceME methods
//  cout << (dbe->makeReferenceME(h1)?1:0) << endl;
//  MonitorElement* rh1 = dbe->getReferenceME(h1);
//  cout << rh1->getPathname() << endl;
//  dbe->deleteME(rh1);
//  cout << h1->getPathname() << endl;
//  cout << (dbe->isReferenceME(h2)?1:0) << endl;
//  cout << (dbe->makeReferenceME(h2)?1:0) << endl;
//  MonitorElement* rh2 = dbe->getReferenceME(h2);
//  cout << (dbe->isReferenceME(rh2)?1:0) << endl;
//  cout << (dbe->isReferenceME(h2)?1:0) << endl;
//  dbe->deleteME(h2);
//  dbe->deleteME(f1);
//  dbe->showDirStructure();
}


DQMSourceExample::~DQMSourceExample()
{
   
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void DQMSourceExample::beginJob(const EventSetup& context){
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginJob(context);
  // then do your thing

}

//--------------------------------------------------------
void DQMSourceExample::beginRun(const EventSetup& context) {
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginRun(context);
  // then do your thing

}

//--------------------------------------------------------
void DQMSourceExample::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
     const EventSetup& context) {
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginLuminosityBlock(lumiSeg,context);
  // then do your thing
  
}

// ----------------------------------------------------------
void DQMSourceExample::analyze(const Event& iEvent, 
			       const EventSetup& iSetup )
{  
  // call DQMAnalyzer some place
  DQMAnalyzer::analyze(iEvent,iSetup); 
  
  i1->Fill(4);
  f1->Fill(-3.14);
 
   // Filling the histogram with random data
  srand( 0 );

  if(getNumEvents()%1000 == 0)
    cout << " # of cycles = " << getNumEvents() << endl;

  for(int i = 0; i != 20; ++i ) 
    {
      float x = gRandom->Uniform(XMAX);
      h1->Fill(x,1./log(x+1));
      h3->Fill(x, 1);
      h4->Fill(gRandom->Gaus(30, 3), 1.0);
      h5->Fill(gRandom->Poisson(15), 0.5);
      h6->Fill(gRandom->Gaus(25, 15), 1.0);
      h7->Fill(gRandom->Gaus(25, 8), 1.0);
    }

  // fit h4 to gaussian, this should not be done in the source
  /*MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (h4);
  if(ob)
    {
      TH1F * root_ob = dynamic_cast<TH1F *> (ob->operator->());
      if(root_ob)root_ob->Fit("gaus");
    }*/
  
  for ( int i = 0; i != 10; ++i ) 
    {
      float x = gRandom->Gaus(15, 7);
      float y = gRandom->Gaus(20, 5);
      h2->Fill(x,y);
      p1->Fill(x,y);
    }
  //      (*(*i1))++;
  usleep(100);

}




//--------------------------------------------------------
void DQMSourceExample::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                          const EventSetup& context) {
  // do your thing here

  // call DQMAnalyzer at the end 
  DQMAnalyzer::endLuminosityBlock(lumiSeg,context);
}
//--------------------------------------------------------
void DQMSourceExample::endRun(const Run& r, const EventSetup& context){
  // do your thing here
  
  // call DQMAnalyzer at the end
  DQMAnalyzer::endRun(r,context); 
}
//--------------------------------------------------------
void DQMSourceExample::endJob(){
  // do your thing ... call DQMAnalyzer in the end
  
  DQMAnalyzer::endJob();
}


