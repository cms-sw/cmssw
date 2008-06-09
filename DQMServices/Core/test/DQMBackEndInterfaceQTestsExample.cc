// -*- C++ -*-
//
// Package:    DQMServices/CoreROOT
// Class:      DQMStoreQTestsExample
// 
/**\class DQMStoreQTestsExample

Description: Simple example that fills monitoring elements and 
             compares them to reference

Implementation:
<Notes on implementation>
*/
//
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/QTest.h"

#include <TRandom.h>

#include <iostream>
#include <string>

using std::cout; using std::endl;
using std::string; using std::vector;

const int sample_int_value = 5;
//
// class declaration
//

class DQMStoreQTestsExample : public edm::EDAnalyzer {
public:
  explicit DQMStoreQTestsExample( const edm::ParameterSet& );
  ~DQMStoreQTestsExample();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
  virtual void endJob(void);

private:
  // ----------member data ---------------------------
  
  // the test objects
  MonitorElement * h1;
  MonitorElement * int1;
  MonitorElement *poMPLandauH1_;
  // the reference objects
  MonitorElement * href;
  MonitorElement * intRef1;
  // more test objects
  MonitorElement * testh2f;
  MonitorElement * testprof;
  MonitorElement * testprof2d;
  // event counter
  int counter;
  // back-end interface
  DQMStore * dbe;
  // quality tests
  Comp2RefChi2 * chi2_test; // chi2 test
  Comp2RefKolmogorov * ks_test; // Kolmogorov test
  ContentsXRange * xrange_test; // contents within x-range test
  ContentsYRange * yrange_test;  // contents within y-range test
  DeadChannel * deadChan_test;  // check against dead channels
  NoisyChannel * noisyChan_test;  // check against noisy channels
  Comp2RefEqualH * equalH_test; // equality test for histograms
  MeanWithinExpected * meanNear_test; // mean-within-expected test
  // MostProbableLandau *poMPLandau_test_;
  // contents within z-range tests
  // ContentsTH2FWithinRange * zrangeh2f_test; 
  // ContentsProfWithinRange * zrangeprof_test; 
  // ContentsProf2DWithinRange * zrangeprof2d_test;

  // use <ref> as the reference for the quality tests
  void setReference(MonitorElement * ref);
  // run quality tests; expected_status: test status that is expected
  // (see Core/interface/QTestStatus.h)
  // test_type: info message on what kind of tests are run
  void runTests(int expected_status, string test_type);
  // called by runTests; return status
  int checkTest(QCriterion *qc);
  // show channels that failed test
  void showBadChannels(QCriterion *qc);

  // gaussian parameters for generated distribution
  float mean_; float sigma_;
  float dLandauMP_;
  float dLandauSigma_;
};

// constructors and destructor
DQMStoreQTestsExample::DQMStoreQTestsExample(const edm::ParameterSet& iConfig ) : counter(0)
{
  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  // set # of bins, range for histogram(s)
  const int NBINS = 50; const float XMIN = -3; const float XMAX = 3;
  h1 = dbe->book1D("histo_1", "Example 1D histogram.", NBINS, XMIN, XMAX );
  href = dbe->book1D("href", "Reference histogram", NBINS, XMIN, XMAX );
  poMPLandauH1_ = dbe->book1D( "landau_hist", "Example Landau Histogram",
                               NBINS, XMIN, XMAX);

  int1 = dbe->bookInt("int1");
  intRef1 = dbe->bookInt("int1Ref");


  testh2f = dbe->book2D("testh2f", "testh2f histo", NBINS/10, XMIN, XMAX, 
			NBINS/10, XMIN, XMAX);
  // profile histogram
  testprof = dbe->bookProfile("testprof", "testprof histo", 
			      NBINS/10, XMIN, XMAX, NBINS/10, XMIN, XMAX);
  // 2D profile histogram
  testprof2d = dbe->bookProfile2D("testprof2d", "testprof2d histo", 
				  NBINS/10, XMIN, XMAX, NBINS/10, XMIN, XMAX,
				  NBINS/10, XMIN, XMAX);
  
  mean_ = (XMIN + XMAX)/2.0;
  sigma_ = (XMAX - XMIN)/6.0;

  dLandauMP_    = ( XMIN + XMAX) / 4.0;
  dLandauSigma_ = ( XMAX - XMIN) / 6.0;

  // fill in reference histogram with random data
  for(unsigned i = 0; i != 10000; ++i)
    href->Fill(gRandom->Gaus(mean_, sigma_));
  
  // instantiate the quality tests
  chi2_test = new Comp2RefChi2("my_chi2");
  ks_test = new Comp2RefKolmogorov("my_kolm");
  xrange_test = new ContentsXRange("my_xrange");
  yrange_test = new ContentsYRange("my_yrange");
  deadChan_test = new DeadChannel("deadChan");
  noisyChan_test = new NoisyChannel("noisyChan");
  equalH_test = new Comp2RefEqualH("my_histo_equal");
  meanNear_test = new MeanWithinExpected("meanNear");
  //zrangeh2f_test = new ContentsTH2FWithinRange("zrangeh2f");
  //zrangeprof_test = new ContentsProfWithinRange("zrangeprof");
  //zrangeprof2d_test = new ContentsProf2DWithinRange("zrangeprof2d");
  //poMPLandau_test_ = new MostProbableLandau( "mplandau");
  
  // set reference for chi2, ks tests
  setReference(href);
  // set allowed range to [10, 90]% of nominal
  xrange_test->setAllowedXRange(0.1*XMIN, 0.9*XMAX);
  // set allowed range to [0, 15] entries
  yrange_test->setAllowedYRange(0, 40);
  // set threshold for "dead channel" definition (default: 0)
  deadChan_test->setThreshold(0);
  // set tolerance for noisy channel
  noisyChan_test->setTolerance(0.30);
  // set # of neighboring channels for calculating average (default: 1)
  noisyChan_test->setNumNeighbors(2);
  // use RMS of distribution to judge if mean near expected value
  meanNear_test->useRMS();
  // Setup MostProbableLandau
  //poMPLandau_test_->setXMin( 0.1 * XMIN);
  //poMPLandau_test_->setXMax( 0.9 * XMAX);
  //poMPLandau_test_->setMostProbable( dLandauMP_);
  //poMPLandau_test_->setSigma( dLandauSigma_);

  // fill in test integer
  int1->Fill(sample_int_value);
}

// use <ref> as the reference for the quality tests
void DQMStoreQTestsExample::setReference(MonitorElement * ref)
{
// FIXME, need to use reference in proper location /Reference here
//  if(chi2_test)chi2_test->setReference(ref);
//  if(ks_test)ks_test->setReference(ref);  
//  if(equalH_test)equalH_test->setReference(ref);
}


DQMStoreQTestsExample::~DQMStoreQTestsExample()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  if(chi2_test)delete chi2_test;
  if(ks_test)delete ks_test;
  if(xrange_test)delete xrange_test;
  if(yrange_test)delete yrange_test;
  if(deadChan_test)delete deadChan_test;
  if(noisyChan_test)delete noisyChan_test;
  if(equalH_test) delete equalH_test;
  if(meanNear_test)delete meanNear_test;
//  if(zrangeh2f_test) delete zrangeh2f_test;
//  if(zrangeprof_test) delete zrangeprof_test;
//  if(zrangeprof2d_test) delete zrangeprof2d_test;
//  if( poMPLandau_test_) delete poMPLandau_test_;

}

void DQMStoreQTestsExample::endJob(void)
{
  setReference(0);
  
  // attempt to run tests w/o a reference histogram
  runTests(dqm::qstatus::INVALID, "tests w/o reference");

  // test #3: set reference, but assume low statistics
  setReference(href);
  // set expected mean value
  meanNear_test->setExpectedMean(mean_);
  // set allowed range for mean & RMS
  //zrangeh2f_test->setMeanRange(0., 3.);
  //zrangeh2f_test->setRMSRange(0., 2.);
  //zrangeh2f_test->setMeanTolerance(2.);
  //zrangeprof_test->setMeanRange(0., 1.);
  //zrangeprof_test->setRMSRange(0., 1.);
  //zrangeprof_test->setMeanTolerance(2.);
  //zrangeprof2d_test->setMeanRange(0., 3.);
  //zrangeprof2d_test->setRMSRange(0., 1.);
  //zrangeprof2d_test->setMeanTolerance(2.);

  chi2_test->setMinimumEntries(10000);
  ks_test->setMinimumEntries(10000);
  xrange_test->setMinimumEntries(10000);
  yrange_test->setMinimumEntries(10000);
  deadChan_test->setMinimumEntries(10000);
  noisyChan_test->setMinimumEntries(10000);
  equalH_test->setMinimumEntries(10000);
  meanNear_test->setMinimumEntries(10000);
  //zrangeh2f_test->setMinimumEntries(10000);
  //zrangeprof_test->setMinimumEntries(10000);
  //zrangeprof2d_test->setMinimumEntries(10000);
  //poMPLandau_test_->setMinimumEntries( 10000);

  // attempt to run tests after specifying a minimum # of entries that is too high
  runTests(dqm::qstatus::INSUF_STAT, "tests w/ low statistics");

  // test #4: this should be the normal test
  chi2_test->setMinimumEntries(0);
  ks_test->setMinimumEntries(0);
  xrange_test->setMinimumEntries(0);
  yrange_test->setMinimumEntries(0);
  deadChan_test->setMinimumEntries(0);
  noisyChan_test->setMinimumEntries(0);
  equalH_test->setMinimumEntries(0);
  meanNear_test->setMinimumEntries(0);
  //zrangeh2f_test->setMinimumEntries(0);
  //zrangeprof_test->setMinimumEntries(0);
  //zrangeprof2d_test->setMinimumEntries(0);
  //poMPLandau_test_->setMinimumEntries( 0);
  // run tests normally
  runTests(0, "regular tests");
}

// run quality tests; expected_status: test status that is expected
// (see Core/interface/QTestStatus.h)
// test_type: info message on what kind of tests are run
void DQMStoreQTestsExample::runTests(int expected_status, 
					    string test_type)
{
  cout << " ========================================================== " << endl;
  cout << " Results of attempt to run " << test_type << ", expected status " << expected_status << endl;
  
  chi2_test->runTest(h1);
  checkTest(chi2_test);

  ks_test->runTest(h1);
  checkTest(ks_test);

  xrange_test->runTest(h1);
  checkTest(xrange_test);

  yrange_test->runTest(h1);
  checkTest(yrange_test);
  showBadChannels(yrange_test);

  deadChan_test->runTest(h1);
  checkTest(deadChan_test);
  showBadChannels(deadChan_test);

  noisyChan_test->runTest(h1);
  checkTest(noisyChan_test);
  showBadChannels(noisyChan_test);

  meanNear_test->runTest(h1);
  checkTest(meanNear_test);

  //poMPLandau_test_->runTest( poMPLandauH1_);
  //checkTest( poMPLandau_test_);

  equalH_test->runTest(h1);
  checkTest(equalH_test);
  showBadChannels(equalH_test);

  //zrangeh2f_test->runTest(testh2f);
  //checkTest(zrangeh2f_test);
  //showBadChannels(zrangeh2f_test);

  //zrangeprof_test->runTest(testprof);
  //checkTest(zrangeprof_test);
  //showBadChannels(zrangeprof_test);

  //zrangeprof2d_test->runTest(testprof2d);
  //checkTest(zrangeprof2d_test);
  //showBadChannels(zrangeprof2d_test);

  int status = 0;
  status = chi2_test->getStatus();
  if (expected_status && status != expected_status)
    cout << "ERROR: Comp2RefChi2 test expected status " << expected_status
	 << ", got " << status << endl;
  status = ks_test->getStatus();
  if (expected_status && status != expected_status)
    cout << "ERROR: Comp2RefKolmogorov test expected status " << expected_status
	 << ", got " << status << endl;

  status = xrange_test->getStatus();
  // there is no "INVALID" result when running "contents within x-range" test
  if (expected_status
      && expected_status != dqm::qstatus::INVALID
      && expected_status != status)
    cout << "ERROR: ContentsXRange test expected status " << expected_status
	 << ", got " << status << endl;

  status = yrange_test->getStatus();
  // there is no "INVALID" result when running "contents within y-range" test
  if (expected_status
      && expected_status != dqm::qstatus::INVALID
      && expected_status != status)
    cout << "ERROR: ContentsYRange test expected status " << expected_status
	 << ", got " << status << endl;

  status = deadChan_test->getStatus();
  // there is no "INVALID" result when running "dead channel" test
  if (expected_status
      && expected_status != dqm::qstatus::INVALID
      && expected_status != status)
    cout << "ERROR: DeadChannel test expected status " << expected_status
	 << ", got " << status << endl;

  status = noisyChan_test->getStatus();
  // there is no "INVALID" result when running "noisy channel" test
  if (expected_status
      && expected_status != dqm::qstatus::INVALID
      && expected_status != status)
    cout << "ERROR: NoisyChannel test expected status " << expected_status
	 << ", got " << status << endl;

  status = meanNear_test->getStatus();
  if (expected_status && expected_status != status)
    cout << "ERROR: MeanWithinExpected test expected status " << expected_status
	 << ", got " << status << endl;

  //status = poMPLandau_test_->getStatus();
  //if (expected_status && expected_status != status)
  //  cout << "ERROR: MostProbableLandau test expected status " << expected_status
  //	 << ", got " << status << endl;

  status = equalH_test->getStatus();
  if (expected_status && expected_status != status)
    cout << "ERROR: Comp2RefEqualH test expected status " << expected_status
	 << ", got " << status << endl;

  //status = zrangeh2f_test->getStatus();
  // if (expected_status && expected_status != status)
  //  cout << "ERROR: Comp2RefEqualInt test expected status " <<  expected_status
  //	 << ", got " << status << endl;

  //status = zrangeprof_test->getStatus();
  // if (expected_status && expected_status != status)
  //  cout << "ERROR: ContentsProfWithinRange test expected status " << expected_status
  //	   << ", got " << status << endl;

  //status = zrangeprof2d_test->getStatus();
  // if (expected_status && expected_status != status)
  //  cout << "ERROR: ContentsProf2DWithinRange test expected status " << expected_status
  //	   << ", got " << status << endl;
}

// called by runTests; return status
int DQMStoreQTestsExample::checkTest(QCriterion *qc)
{
  if(!qc)return -1;
  
  int status = qc->getStatus();
  cout << " Test name: " << qc->getName() << " (Algorithm: " 
	    << qc->algoName() << "), Result:"; 
  
  switch(status)
    {
    case dqm::qstatus::STATUS_OK: 
      cout << " Status ok " << endl;
      break;
    case dqm::qstatus::WARNING: 
      cout << " Warning " << endl;
      break;
    case dqm::qstatus::ERROR : 
      cout << " Error " << endl;
      break;
    case dqm::qstatus::DISABLED : 
      cout << " Disabled " << endl;
      break;
    case dqm::qstatus::INVALID: 
      cout << " Invalid " << endl;
      break;
    case dqm::qstatus::INSUF_STAT: 
      cout << " Not enough statistics " << endl;
      break;

    default:
      cout << " Unknown (status = " << status << ") " << endl;
    }
  
  string message = qc->getMessage();
  cout << " Message:" << message << endl;
  
  return status;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void DQMStoreQTestsExample::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  // fill in test histogram with random data
  h1->Fill(gRandom->Gaus(mean_, sigma_)); 

  testh2f->Fill(gRandom->Gaus(mean_, sigma_), 
		gRandom->Gaus(mean_, sigma_));

  testprof->Fill(gRandom->Gaus(mean_, sigma_), 
		 TMath::Abs(gRandom->Gaus(mean_, sigma_)));

  testprof2d->Fill(gRandom->Gaus(mean_, sigma_), 
		   gRandom->Gaus(mean_, sigma_), 
		   TMath::Abs(gRandom->Gaus(mean_, sigma_)));

  poMPLandauH1_->Fill( gRandom->Landau( dLandauMP_, dLandauSigma_));

  ++counter;
}

// show channels that failed test
void DQMStoreQTestsExample::showBadChannels(QCriterion *qc)
{
  vector<dqm::me_util::Channel> badChannels = qc->getBadChannels();
  if(!badChannels.empty())
    cout << " Channels that failed test " << qc->algoName() << ":\n";
  
  vector<dqm::me_util::Channel>::iterator it = badChannels.begin();
  while(it != badChannels.end())
    {
      cout << " Channel ("
           << it->getBinX() << ","
           << it->getBinY() << ","
           << it->getBinZ()
	   << ") Contents: " << it->getContents() << " +- " 
	   << it->getRMS() << endl;

      ++it;
    }
}


// define this as a plug-in
DEFINE_FWK_MODULE(DQMStoreQTestsExample);

