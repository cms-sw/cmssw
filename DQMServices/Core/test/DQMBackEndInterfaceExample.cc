// -*- C++ -*-
//
// Package:    DQMServices/CoreROOT
// Class:      DQMStoreExample
// 
/**\class DQMStoreExample

Description: Simple example showing how to book, fill and delete monitoring elements

Implementation:
<Notes on implementation>
*/
//
//
//


// system include files
#include <memory>

#include <sstream>
#include <string>
#include <vector>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//
const int NBINS = 50;

class DQMStoreExample : public edm::EDAnalyzer {
public:
  explicit DQMStoreExample( const edm::ParameterSet& );
  ~DQMStoreExample();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
  virtual void endJob(void);

private:
  // ----------member data ---------------------------
  
  MonitorElement * h0;
  MonitorElement * h1;
  MonitorElement * h2;
  MonitorElement * h7;
  MonitorElement * h8;
  MonitorElement * s1;
  MonitorElement * h3;
  MonitorElement * h4;
  MonitorElement * h5;
  MonitorElement * h6;
  MonitorElement * i1;
  MonitorElement * f1;
  MonitorElement * h54;
  MonitorElement * h64;
  MonitorElement * i2;
  // event counter
  int counter;
  // back-end interface
  DQMStore * dbe;
  // test back-end interface functionality
  void integrityChecks();
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DQMStoreExample::DQMStoreExample(const edm::ParameterSet&
						       iConfig ) 
  : counter(0)
{
  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  const float XMIN = 0; const float XMAX = 50;
  h0 = dbe->book1D("histo0", "Example 1D histogram.", NBINS, XMIN, XMAX );
  
  // create and cd into new folder
  dbe->setCurrentFolder("C1/C2/C3");
  h1 = dbe->book1D("histo", "Example 1D histogram.", NBINS, XMIN, XMAX );
  h2 = dbe->book2D("histo2", "Example 2D histogram.", NBINS, XMIN, XMAX,
		   NBINS, XMIN, XMAX);
  assert(NBINS == h1->getNbinsX());

  h7 = dbe->bookProfile("histo7", "Example Profile histogram.", 
			NBINS, XMIN, XMAX, NBINS, XMIN, XMAX);
  
  s1 = dbe->bookString("string1","This is a test string");
  
  dbe->setCurrentFolder("B1/B2/");
  h3 = dbe->book1D("histo3", "Example 1D histogram.", NBINS, XMIN, XMAX );
  h4 = dbe->book2D("histo4", "Example 2D histogram.", NBINS, XMIN, XMAX, 
		   NBINS, XMIN, XMAX);
  assert(NBINS == h4->getNbinsX());
  assert(NBINS == h4->getNbinsY());
  
  // cd into folder
  dbe->setCurrentFolder("C1/C2");
  h5 = dbe->book1D("histo5", "Example 1D histogram.", NBINS, XMIN, XMAX );
  h6 = dbe->book2D("histo6", "Example 2D histogram.", NBINS, XMIN, XMAX, 
		   NBINS, XMIN, XMAX);
  i1 = dbe->bookInt("integer1");
  f1 = dbe->bookFloat("float1");
  //   s1 = dbe->bookString("s1", "my string");

  // create and cd into new folder
  dbe->setCurrentFolder("C1/C2/C4");
  h54 = dbe->book1D("histo5", "Example 1D histogram.", NBINS, XMIN, XMAX );
  h64 = dbe->book2D("histo6", "Example 2D histogram.", NBINS, XMIN, XMAX, 
		    NBINS, XMIN, XMAX);
  i2 = dbe->bookInt("integer2");
  // remove directory (and contents)
  dbe->rmdir("C1/C2/C4");
  
  // recreate and cd into directory
  dbe->setCurrentFolder("C1/C2/C4");
  h54 = dbe->book1D("histo5", "Example 1D histogram.", NBINS, XMIN, XMAX );
  dbe->rmdir("C1/C2/C4");
  
  dbe->setCurrentFolder("C1/C2/C4");
  h64 = dbe->book2D("histo6", "Example 2D histogram.", NBINS, XMIN, XMAX, 
		    NBINS, XMIN, XMAX);

  h8 = dbe->book1D("histo8", "Example 1D histogram.", NBINS, XMIN, XMAX );

  // now do some cross-checks...
  integrityChecks();
}

// test back-end interface functionality
void DQMStoreExample::integrityChecks()
{
 
  std::vector<MonitorElement * > contents, dbe_ret;
  
  // First set of checks: make sure we can get MonitorElements in each folder:

  // contents of top (root) folder
  contents.push_back(h0);
  dbe_ret = dbe->getContents(""); // this is the top folder
  assert(dbe_ret == contents);
  // contents of B1/B2
  contents.clear();
  contents.push_back(h3);  contents.push_back(h4);
  dbe_ret = dbe->getContents("B1/B2");
  assert(dbe_ret == contents);
  // contents of C1/C2
  contents.clear();
  contents.push_back(f1); contents.push_back(h5);
  contents.push_back(h6); contents.push_back(i1);
  dbe_ret = dbe->getContents("C1/C2");
  assert(dbe_ret == contents);  
  // contents of C1/C2/C3
  contents.clear();
  contents.push_back(h1); contents.push_back(h2);
  contents.push_back(h7); contents.push_back(s1);
  dbe_ret = dbe->getContents("C1/C2/C3");
  assert(dbe_ret == contents);  
  // contents of C1/C2/C4
  contents.clear();
  contents.push_back(h64); contents.push_back(h8);
  dbe_ret = dbe->getContents("C1/C2/C4");
  assert(dbe_ret == contents);  

  // Second set of checks: make sure we can use simple pathnames including subdirs

  // contents of "C1/C2" (including subdirectories)
  contents.clear();
  contents.push_back(f1); contents.push_back(h5);
  contents.push_back(h6); contents.push_back(i1);
  contents.push_back(h1); contents.push_back(h2);
  contents.push_back(h7); contents.push_back(s1);
  contents.push_back(h64); contents.push_back(h8);
  dbe_ret = dbe->getAllContents("C1/C2");
  assert(dbe_ret == contents); 
  dbe_ret = dbe->getAllContents("C1/C2/"); // w/ or w/o a "slash" at the end
  assert(dbe_ret == contents);

  // Third set of checks: make sure we can use simple pathnames with wildcards

  // contents of "C1/C2" (including subdirectories)
  dbe_ret = dbe->getMatchingContents("C1/C2/*");
  assert(dbe_ret == contents);

  // contents of "C1/C2": only histograms (including subdirectories)
  contents.clear();
  contents.push_back(h5); contents.push_back(h6);
  contents.push_back(h1); contents.push_back(h2);
  contents.push_back(h7);
  contents.push_back(h64); contents.push_back(h8);
  dbe_ret = dbe->getMatchingContents("C1/C2/*histo*");
  assert(dbe_ret == contents);

  contents.erase(contents.begin()+2); // h1
  dbe_ret = dbe->getMatchingContents("C1/C2/*sto?");
  assert(dbe_ret == contents);
}

DQMStoreExample::~DQMStoreExample()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  // go to top directory
  dbe->cd();
  // remove MEs at top directory
  dbe->removeContents(); 
  // remove directory (including subdirectories recursively)
  dbe->rmdir("C1");
  dbe->rmdir("B1");
}

void DQMStoreExample::endJob(void)
{
  // rounding error
  float epsilon = 0.0001;

  // loop over all bins
  for(int bin = 1; bin <= NBINS; ++bin)
    {
      float content = 50. * (float)std::rand() / RAND_MAX;
      float error   =  5. * (float)std::rand() / RAND_MAX;
      std::ostringstream label; label << "bin # " << bin;
      // set bin content & uncertainty
      h8->setBinContent(bin, content);
      h8->setBinError(  bin, error);
      // set bin label (of X-axis)
      h8->setBinLabel(bin, label.str());

      // make sure contents & uncertainties are correctly set...
      assert(fabs(content - h8->getBinContent(bin)) < epsilon);
      assert(fabs(error   - h8->getBinError(bin)  ) < epsilon);
    }

  dbe->showDirStructure();
  dbe->save("test.root");  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void DQMStoreExample::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  for ( int i = 0; i < 10; ++i ) {
    float x = 50. * (float)std::rand()/ RAND_MAX;
    
    h0->Fill(x/2);
    h1->Fill(x);
    h5->Fill(x-1.);
  }
  for ( int i = 0; i < 10; ++i ) {
    for ( int j = 0; j < 10; ++j ) {
      float x = 50. * (float)std::rand() / RAND_MAX;
      float y = 50. * (float)std::rand() / RAND_MAX;

      h2->Fill(x,y);
      h7->Fill(x,y);
      h3->Fill(y);
      h4->Fill(y,x);
      h6->Fill(x*y,x/y);
    }
  }

  i1->Fill(547467);
  f1->Fill(3.14159274101257324);
  counter++;
}

// define this as a plug-in
DEFINE_FWK_MODULE(DQMStoreExample);

