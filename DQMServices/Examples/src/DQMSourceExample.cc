/*
 * \file DQMSourceExample.cc
 * \author C.Leonidopoulos
 * Last Update:
 * $Date: 2008/03/28 15:53:38 $
 * $Revision: 1.15 $
 * $Author: lat $
 *
 * Description: Simple example showing how to create a DQM Source creating and filling
 * monitoring elements
*/

#include "DQMServices/Examples/interface/DQMSourceExample.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TRandom.h"
#include <math.h>

using namespace std;
using namespace edm;


// -----------------------------
//  constructors and destructor
// -----------------------------

DQMSourceExample::DQMSourceExample( const edm::ParameterSet& ps ) {

  parameters_ = ps;
  initialize();

}


DQMSourceExample::~DQMSourceExample() {

}


// ----------------------------------

void DQMSourceExample::initialize() {

  counterLS_  = 0;
  counterEvt_ = 0;

  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();

  // base folder for the contents of this job
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  cout << "Monitor name = " << monitorName_ << endl;
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;

  prescaleLS_  = parameters_.getUntrackedParameter<int>("prescaleLS",  -1);
  cout << "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;

  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  cout << "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;

}


// ---------------------------------------------------------

void DQMSourceExample::beginJob(const EventSetup& context) {

  NBINS = 40 ;
  XMIN  =  0.;
  XMAX  = 40.;

  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();

  //  create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"Summary");

  summ = dbe_->book1D("summary", "Run Summary", 100, 0, 100);

  //  create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"QTests");

  fitResult = dbe_->book1D("fitResults", "Guassian fit results", 2, 0, 1);

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

  //  create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"C1");
  const int NBINS2 = 10;

  p1 = dbe_->bookProfile(  "prof1", "My profile 1D", NBINS,XMIN,XMAX,NBINS,XMIN,XMAX,"");
  p2 = dbe_->bookProfile2D("prof2", "My profile 2D", NBINS,XMIN,XMAX,NBINS,XMIN,XMAX,NBINS,XMIN,XMAX,"");

  h7 = dbe_->book1D("histo7", "Example 7 1D histogram.", NBINS2, XMIN, XMAX);

  char temp[1024];
  for(int i = 1; i <= NBINS2; ++i) {

    sprintf(temp, " bin no. %d", i);
    h7->setBinLabel(i, temp);

  }

  i1 = dbe_->bookInt("int1");
  f1 = dbe_->bookFloat("float1");
  s1 = dbe_->bookString("s1", "My string");


  // assign tag to MEs h1, h2 and h7
  const unsigned int detector_id = 17;
  //  dbe_->tag(h1, detector_id);
  //  dbe_->tag(h2, detector_id);
  dbe_->tag(h7, detector_id);

  // tag full directory
  dbe_->tagContents(monitorName_+"C1", detector_id);

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

}


// ----------------------------------------------------------------------------

void DQMSourceExample::beginRun(const edm::Run& r, const EventSetup& context) {

}


// ------------------------------------------------------------------------

void DQMSourceExample::beginLuminosityBlock(const LuminosityBlock& lumiSeg,
					    const EventSetup& context) {

}


// ----------------------------------------------------------------------------

void DQMSourceExample::analyze(const Event& iEvent, const EventSetup& iSetup) {

  counterEvt_++;
  if (prescaleEvt_<1)  return;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0)  return;
  //  cout << " processing conterEvt_: " << counterEvt_ <<endl;

  i1->Fill(4);
  f1->Fill(-3.14);
 
  // Filling the histogram with random data
  srand( 0 );

  if(counterEvt_%1000 == 0) {
    cout << " # of events = " << counterEvt_ << endl;
    summ->Fill(counterEvt_/1000., counterEvt_);
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

  for ( int i = 0; i != 10; ++i ) {

    float w = gRandom->Uniform(XMAX);

    noisyTrue->Fill(  w, 1.);
    noisyFalse->Fill( z, 1.);

    float x = gRandom->Gaus(12, 1);
    float y = gRandom->Gaus(20, 2);

    p1->Fill(x, y);
    p2->Fill(x, y, (x+y)/2.);
    h7->Fill(y, 1.);

  }

  usleep(100);

  float mean = 0., rms = 0.;

  if (TH1F *rootHisto = meanTrue->getTH1F()) {

    TF1 *f1 = new TF1("f1","gaus",1,3);
    rootHisto->Fit("f1");
    mean = f1->GetParameter(1);
    rms  = f1->GetParameter(2);
  }

  fitResult->setBinContent(1, mean);
  fitResult->setBinContent(2,  rms);

}


// ----------------------------------------------------------------------

void DQMSourceExample::endLuminosityBlock(const LuminosityBlock& lumiSeg,
					  const EventSetup& context) {

}


// ---------------------------------------------------------------------

void DQMSourceExample::endRun(const Run& r, const EventSetup& context) {

}


// ------------------------------

void DQMSourceExample::endJob() {

}
