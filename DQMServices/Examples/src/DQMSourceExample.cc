/*
 * \file DQMSourceExample.cc
 * \author C.Leonidopoulos
 * Last Update:
 * $Date: 2009/12/14 22:22:23 $
 * $Revision: 1.26 $
 * $Author: wmtan $
 *
 * Description: Simple example showing how to create a DQM source creating and filling
 * monitoring elements
*/

#include "DQMServices/Examples/interface/DQMSourceExample.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TRandom.h"
#include <math.h>

using namespace std;
using namespace edm;

//==================================================================//
//================= Constructor and Destructor =====================//
//==================================================================//
DQMSourceExample::DQMSourceExample( const edm::ParameterSet& ps ){
  parameters_ = ps;
  initialize();
}

DQMSourceExample::~DQMSourceExample(){
}

//==================================================================//
//======================= Initialise ===============================//
//==================================================================//
void DQMSourceExample::initialize() {

  ////---- initialise Event and LS counters
  counterEvt_ = 0; counterLS_  = 0;

  ////---- get DQM store interface
  dbe_ = Service<DQMStore>().operator->();

  ////---- define base folder for the contents of this job 
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  cout << "DQMSourceExample: Monitor name = " << monitorName_ << endl;
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  
  ////--- get steerable parameters
  prescaleLS_  = parameters_.getUntrackedParameter<int>("prescaleLS",  -1);
  cout << "DQMSourceExample: DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  cout << "DQMSourceExample: DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;

// read in files (use DQMStore.collateHistograms = True for summing
//  dbe_->load("ref.root");
//  dbe_->load("ref.root");
}

//==================================================================//
//========================= beginJob ===============================//
//==================================================================//
void DQMSourceExample::beginJob() {
  ////---- get DQM store interface
  dbe_ = Service<DQMStore>().operator->();

  ////----  create summ and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"DQMsource/Summary");
  summ = dbe_->book1D("summary", "Run Summary", 100, 0, 100); 

  //-------------------------------------
  // testing of Quality Tests 
  //-------------------------------------

   ////---- create and cd into new folder
   dbe_->setCurrentFolder(monitorName_+"DQMsource/QTests");

   ////---- define histogram binning
   NBINS = 40 ; XMIN  =  0.; XMAX  = 40.;

  ////---- book histograms for testsing of quality tests
  ////     a quality test applied to each of these histograms
  ////     as defined in DQMServices/Examples/test/QualityTests.xml
  xTrue     = dbe_->book1D("XTrue",       "X Range QTest",                  NBINS, XMIN, XMAX);
  xFalse    = dbe_->book1D("XFalse",      "X Range QTest",                  NBINS, XMIN, XMAX);
  yTrue     = dbe_->book1D("YTrue",       "Y Range QTest",                  NBINS, XMIN, XMAX);
  yFalse    = dbe_->book1D("YFalse",      "Y Range QTest",                  NBINS, XMIN, XMAX);
  wExpTrue  = dbe_->book2D("WExpTrue",    "Contents Within Expected QTest", NBINS, XMIN, XMAX, NBINS, XMIN, XMAX);
  wExpFalse = dbe_->book2D("WExpFalse",   "Contents Within Expected QTest", NBINS, XMIN, XMAX, NBINS, XMIN, XMAX);
  meanTrue  = dbe_->book1D("MeanTrue",    "Mean Within Expected QTest",     NBINS, XMIN, XMAX);
  meanFalse = dbe_->book1D("MeanFalse",   "Mean Within Expected QTest",     NBINS, XMIN, XMAX);
  deadTrue  = dbe_->book1D("DeadTrue",    "Dead Channel QTest",             NBINS, XMIN, XMAX);
  deadFalse = dbe_->book1D("DeadFalse",   "Dead Channel QTest",             NBINS, XMIN, XMAX);
  noisyTrue  = dbe_->book1D("NoisyTrue",  "Noisy Channel QTest",            NBINS, XMIN, XMAX);
  noisyFalse = dbe_->book1D("NoisyFalse", "Noisy Channel QTest",            NBINS, XMIN, XMAX);


  //-------------------------------------
  // book several ME more  
  //-------------------------------------

  ////----  create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"DQMsource/C1");
  const int NBINS2 = 10;
 
  i1        = dbe_->bookInt("int1");
  f1        = dbe_->bookFloat("float1");
  s1        = dbe_->bookString("s1", "My string");
  h1        = dbe_->book1D("h1f", "Example TH1F 1D histogram.", NBINS2, XMIN, XMAX);
  h2        = dbe_->book1S("h1s", "Example TH1S histogram.", NBINS, XMIN, XMAX);
//  h3        = dbe_->book1DD("h1d", "Example TH1D histogram.", NBINS, XMIN, XMAX);
//  h4        = dbe_->book2DD("h2d", "Example TH2D histogram.", NBINS, XMIN, XMAX,NBINS, XMIN, XMAX);
  p1        = dbe_->bookProfile(  "prof1", "My profile 1D", NBINS,XMIN,XMAX,NBINS,XMIN,XMAX,"");
  p2        = dbe_->bookProfile2D("prof2", "My profile 2D", NBINS,XMIN,XMAX,NBINS,XMIN,XMAX,NBINS,XMIN,XMAX,"");
  h1hist    = dbe_->book1D("history 1D","Example 1 1D history plot", 30, 0.,30.);
 
  // set labels for h1
  char temp[1024];
  for(int i = 1; i <= NBINS2; ++i) {
    sprintf(temp, " bin no. %d", i);
    h1->setBinLabel(i, temp);
  }

  // assign tag to MEs h1
  const unsigned int detector_id = 17;
  dbe_->tag(h1, detector_id);

  // tag full directory
  dbe_->tagContents(monitorName_+"DQMsource/C1", detector_id);

  /*
  // contents of h5 & h6 will be reset at end of monitoring cycle
  h5->setResetMe(true);
  h6->setResetMe(true);
  dbe_->showDirStructure();
  std::vector<std::string> tags;
  dbe_->getAllTags(tags);
  for (size_t i = 0, e = tags.size(); i < e; ++i)
    std::cout << "TAGS [" << i << "] = " << tags[i] << std::endl;
  */

    dbe_->showDirStructure ();
}

//==================================================================//
//========================= beginRun ===============================//
//==================================================================//
void DQMSourceExample::beginRun(const edm::Run& r, const EventSetup& context) {
}


//==================================================================//
//==================== beginLuminosityBlock ========================//
//==================================================================//
void DQMSourceExample::beginLuminosityBlock(const LuminosityBlock& lumiSeg,
					    const EventSetup& context) {
}


//==================================================================//
//==================== analyse (takes each event) ==================//
//==================================================================//
void DQMSourceExample::analyze(const Event& iEvent, const EventSetup& iSetup) {
  counterEvt_++;
  if (prescaleEvt_<1)  return;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0)  return;
  //  cout << " processing conterEvt_: " << counterEvt_ <<endl;

  // fill integer and float
// number exceeding 32 bits
  i1->Fill(400000000000000LL); // FIXME use double
  f1->Fill(-3.14);
 
  //----------------------------------------
  // Filling the histograms with random data
  //----------------------------------------

  srand( 0 );
  // fill summ histo
  if(counterEvt_%1000 == 0) {
    cout << " # of events = " << counterEvt_ << endl;
    summ->Fill(counterEvt_/1000., counterEvt_);
  }
  // fill summ histo
  if(counterEvt_%100 == 0) {
    h1hist->ShiftFillLast(gRandom->Gaus(12,1.),1.,5);
  }

  float z  = gRandom->Uniform(XMAX);
  xTrue->Fill(  z, 1./log(z+1.) );
  xFalse->Fill( z+(XMAX/2.),  z );
  yTrue->Fill(  z, 1./log(z+1.) );
  yFalse->Fill( z, z );
  meanTrue->Fill(  gRandom->Gaus(10,  2), 1.);
  meanFalse->Fill( gRandom->Gaus(12,  3), 1.);
  wExpTrue->Fill(  gRandom->Gaus(12,  1), gRandom->Gaus(12, 1), 1.);
  wExpFalse->Fill( gRandom->Gaus(20,  2), gRandom->Gaus(20, 2), 1.);
  deadTrue->Fill(  gRandom->Gaus(20, 10), 2.);
  deadFalse->Fill( gRandom->Gaus(20,  4), 1.);
  h2->Fill(  gRandom->Gaus(20,  4), 1.);
//  h3->Fill(  XMIN, 0xffff00000000LL);
//  h4->Fill(  XMIN, XMIN, 0xffff00000000LL); 
  
  //h1hist->Print();
  //h1hist->Print();

  for ( int i = 0; i != 10; ++i ) {
    float w = gRandom->Uniform(XMAX);
    noisyTrue->Fill(  w, 1.);
    noisyFalse->Fill( z, 1.);
    float x = gRandom->Gaus(12, 1);
    float y = gRandom->Gaus(20, 2);
    p1->Fill(x, y);
    p2->Fill(x, y, (x+y)/2.);
    h1->Fill(y, 1.);
  }

  // usleep(100);

}

//==================================================================//
//========================= endLuminosityBlock =====================//
//==================================================================//
void DQMSourceExample::endLuminosityBlock(const LuminosityBlock& lumiSeg,
					  const EventSetup& context) {

}

//==================================================================//
//============================= endRun =============================//
//==================================================================//
void DQMSourceExample::endRun(const Run& r, const EventSetup& context) {

}

//==================================================================//
//============================= endJob =============================//
//==================================================================//
void DQMSourceExample::endJob() {
   std::cout << "DQMSourceExample::endJob()" << std::endl;
}
