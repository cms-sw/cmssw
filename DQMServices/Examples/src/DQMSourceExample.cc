/*
 * \file DQMSourceExample.cc
 * \author C.Leonidopoulos
 * Last Update:
 * $Date: 2008/02/22 23:52:29 $
 * $Revision: 1.14 $
 * $Author: lat $
 *
 * Description: Simple example showing how to create a DQM Source creating and filling
 * monitoring elements
*/

#include "DQMServices/Examples/interface/DQMSourceExample.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TRandom.h" // this is just the random number generator
#include <math.h>

using namespace std;
using namespace edm;

//
// constructors and destructor
//
DQMSourceExample::DQMSourceExample( const edm::ParameterSet& ps ) :
counterEvt_(0)
{
     dbe_ = Service<DQMStore>().operator->();
     parameters_ = ps;
     monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
     cout << "Monitor name = " << monitorName_ << endl;
     if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
     prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
     cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;
 
//// use this to read in reference histograms from file
//   dbe_->readReferenceME("DQM_referenceME_R000000001.root");
//   dbe_->open("DQM_referenceME_R000000001.root",false,"","prep");

//// use this to collate histograms from files
//  dbe_->open("../test/DQM_YourSubsystemName_R000000002.root");
//  dbe_->open("../test/DQM_YourSubsystemName_R000000003.root");

//// use this to merge histograms from different runs into the same file
///  dbe_->open("DQM_EcalBarrel_R000020994.root",true,"","");
///  dbe_->open("DQM_L1T_R000020994.root",true,"","");

//  dbe_->open("Ecal_000017224.root",true,"","Run017224");

//// use this to retrieve CMSSW version of file
//  cout << dbe_->getFileReleaseVersion("ref_test.root") << endl;

/// book some histograms here
  const int NBINS = 50; XMIN = 0; XMAX = 50;
  
  
  rooth1 = new TH1F("rooth1","rooth1",NBINS,XMIN,XMAX);
  rooth1->GetXaxis()->SetTitle("X axis title");
  rooth1->GetYaxis()->SetTitle("Y axis title");

  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"C1");
  h1 = dbe_->book1D("histo", "Example 1D histogram.", NBINS, XMIN, XMAX);
  h1->setAxisTitle("x-axis title", 1);
  h1->setAxisTitle("y-axis title", 2);
  h2 = dbe_->book2D("histo2", "Example 2 2D histogram.", NBINS, XMIN, XMAX, 
		   NBINS, XMIN, XMAX);
  p1 = dbe_->bookProfile("prof1","my profile",NBINS,XMIN,XMAX,NBINS,XMIN,XMAX,"");
  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"C1/C2");
  h3 = dbe_->book1D("histo3", "Example 3 1D histogram.", NBINS, XMIN, XMAX);
  h4 = dbe_->book1D("histo4", "Example 4 1D histogram.", NBINS, XMIN, XMAX);
  h5 = dbe_->book1D("histo5", "Example 5 1D histogram.", NBINS, XMIN, XMAX);
  h6 = dbe_->book1D("histo6", "Example 6 1D histogram.", NBINS, XMIN, XMAX);
  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"C1/C3");
  const int NBINS2 = 10;
  h7 = dbe_->book1D("histo7", "Example 7 1D histogram.", NBINS2, XMIN, XMAX);
  char temp[1024];
  for(int i = 1; i <= NBINS2; ++i)
    {
      sprintf(temp, " bin no. %d", i);
      h7->setBinLabel(i, temp);
    }
  i1 = dbe_->bookInt("int1");
  f1 = dbe_->bookFloat("float1");
  s1 = dbe_->bookString("s1", "my string");

  h2->setAxisTitle("Customized x-axis", 1);
  h2->setAxisTitle("Customized y-axis", 2);

  // assign tag to MEs h1, h2 and h7
  const unsigned int detector_id = 17;
  dbe_->tag(h1, detector_id);
  dbe_->tag(h2, detector_id);
  dbe_->tag(h7, detector_id);
  // tag full directory
  dbe_->tagContents(monitorName_+"C1/C3", detector_id);

  // assign tag to MEs h4 and h6
  const unsigned int detector_id2 = 25;
  const unsigned int detector_id3 = 50;
  dbe_->tag(h4, detector_id2);
  dbe_->tag(h6, detector_id3);

  // contents of h5 & h6 will be reset at end of monitoring cycle
  h5->setResetMe(true);
  h6->setResetMe(true);
  dbe_->showDirStructure();
  std::vector<std::string> tags;
  dbe_->getAllTags(tags);
  for (size_t i = 0, e = tags.size(); i < e; ++i)
    std::cout << "TAGS [" << i << "] = " << tags[i] << std::endl;

/// test referenceME methods
//  cout << (dbe_->makeReferenceME(h1)?1:0) << endl;
//  MonitorElement* rh1 = dbe_->getReferenceME(h1);
//  cout << rh1->getPathname() << endl;
//  dbe_->deleteME(rh1);
//  cout << h1->getPathname() << endl;
//  cout << (dbe_->isReferenceME(h2)?1:0) << endl;
//  cout << (dbe_->makeReferenceME(h2)?1:0) << endl;
//  MonitorElement* rh2 = dbe_->getReferenceME(h2);
//  cout << (dbe_->isReferenceME(rh2)?1:0) << endl;
//  cout << (dbe_->isReferenceME(h2)?1:0) << endl;
//  dbe_->deleteME(h2);
//  dbe_->deleteME(f1);
//  dbe_->showDirStructure();
}


DQMSourceExample::~DQMSourceExample()
{
   
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void DQMSourceExample::beginJob(const EventSetup& context){

}

//--------------------------------------------------------
void DQMSourceExample::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void DQMSourceExample::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
     const EventSetup& context) {
  
}

// ----------------------------------------------------------
void DQMSourceExample::analyze(const Event& iEvent, 
			       const EventSetup& iSetup )
{  
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  // cout << " processing conterEvt_: " << counterEvt_ <<endl;
  
  i1->Fill(4);
  f1->Fill(-3.14);
 
   // Filling the histogram with random data
  srand( 0 );

  if(counterEvt_%1000 == 0)
    cout << " # of events = " << counterEvt_ << endl;

  for(int i = 0; i != 20; ++i ) 
    {
      float x = gRandom->Uniform(XMAX);
      h1->Fill(x,1./log(x+1));
      rooth1->Fill(x,1./log(x+1));
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
}
//--------------------------------------------------------
void DQMSourceExample::endRun(const Run& r, const EventSetup& context){

  
  dbe_->setCurrentFolder(monitorName_+"C1");
  // dbe_->clone1D("cloneh1",rooth1);

}
//--------------------------------------------------------
void DQMSourceExample::endJob(){
}


