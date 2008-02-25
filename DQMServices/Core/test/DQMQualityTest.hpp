#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/Core/interface/QTest.h"

#include <TH1F.h>
#include <TRandom.h>

#include <iostream>

const int sample_int_value = 5;
float mean, sigma; // gaussian parameters for generated distribution

class DQMQualityTest
{
 public: 
  // arguments: # of bins and range for histogram to be tested
  DQMQualityTest(int NBINS, float XMIN, float XMAX)
  { 
    xmin_ = XMIN; xmax_ = XMAX;
    // distribution: gaussian w/ parameters: mean, sigma
    mean = (xmin_ + xmax_)/2.0;
    sigma = (xmax_ - xmin_)/6.0;

    dLandauMP_    = ( xmin_ + xmax_) / 4.0;
    dLandauSigma_ = ( xmax_ - xmin_) / 6.0;

    // reference histogram
    my_ref = new TH1F("my_ref", "reference histo", NBINS, XMIN, XMAX);
    // test histogram
    my_test = new TH1F("my_test", "test histo", NBINS, XMIN, XMAX);

    // h2f histogram
    my_testh2f = new TH2F("my_testh2f", "testh2f histo", NBINS/10, XMIN, XMAX, 
			  NBINS/10, XMIN, XMAX);
    // profile histogram
    my_testprof = new TProfile("my_testprof", "testprof histo", 
			       NBINS/10, XMIN, XMAX, XMIN, XMAX);
    // profile histogram
    my_testprof2d = new TProfile2D("my_testprof2d", "testprof2d histo", 
				   NBINS/10, XMIN, XMAX, NBINS/10, XMIN, XMAX,
				   XMIN, XMAX);
    // test histogram
    poTH1Landau_test_ = new TH1F("poTH1Landau_test_", "landau test histo", NBINS, XMIN, XMAX);

    // root stuff
    my_ref->SetDirectory(0); my_test->SetDirectory(0);
    my_testh2f->SetDirectory(0); my_testprof->SetDirectory(0); 
    my_testprof2d->SetDirectory(0); 

    // reference integer
    my_int_ref = new int();
    // test integer
    my_int = new int();

    // Chi2-based test
    chi2_test_ = new Comp2RefChi2("my_chi2");
    // Kolmogorov-based test
    ks_test_ = new Comp2RefKolmogorov("my_kolm");
    // contents within x-range test
    xrange_test_ = new ContentsXRange("my_xrange");
    // set allowed range to [10, 90]% of nominal
    xrange_test_->setAllowedXRange(0.1*xmin_, 0.9*xmax_);
     // contents within y-range test
    yrange_test_ = new ContentsYRange("my_yrange");
    // set allowed range to [0, 40] entries
    yrange_test_->setAllowedYRange(0, 40);
   // check for dead channels
    deadChan_test_ = new DeadChannel("deadChan");
    // set threshold for dead channel (default: 0)
    deadChan_test_->setThreshold(0);
   // check for noisy channels
    noisyChan_test_ = new NoisyChannel("noisyChan");
    // set tolerance for noisy channel
    noisyChan_test_->setTolerance(0.30);
    // set # of neighboring channels for calculating average (default: 1)
    noisyChan_test_->setNumNeighbors(2);

    // Mean-within-expected-value test
    meanNear_test_ = new MeanWithinExpected("meanNear");
    // set expected mean value
    meanNear_test_->setExpectedMean(mean);
    // use RMS of distribution to judge if mean near expected value
    meanNear_test_->useRMS();
    //
    emu_test_ = new AllContentWithinFixedRange("Ricks_test");

    // contents within z-range test
    zrangeh2f_test_ = new ContentsTH2FWithinRange("zrangeh2f");
    zrangeh2f_test_->setMeanRange(0., 3.);
    zrangeh2f_test_->setRMSRange(0., 2.);
    zrangeh2f_test_->setMeanTolerance(2.);
    zrangeprof_test_ = new ContentsProfWithinRange("zrangeprof");
    zrangeprof_test_->setMeanRange(0., 1.);
    zrangeprof_test_->setRMSRange(0., 1.);
    zrangeprof_test_->setMeanTolerance(2.);
    zrangeprof2d_test_ = new ContentsProf2DWithinRange("zrangeprof2d");
    zrangeprof2d_test_->setMeanRange(0., 3.);
    zrangeprof2d_test_->setRMSRange(0., 1.);
    zrangeprof2d_test_->setMeanTolerance(2.);

    // MostProbableLandau
    poMPLandau_test_ = new MostProbableLandau( "mplandau");
    poMPLandau_test_->setXMin( xmin_);
    poMPLandau_test_->setXMax( xmax_);
    poMPLandau_test_->setMostProbable( dLandauMP_);
    poMPLandau_test_->setSigma( dLandauSigma_);

    // equality test for histograms
    equalH1_test_ = new Comp2RefEqualH1("my_histo_equal");
    // equality test for integers
    equalInt_test_ = new Comp2RefEqualInt("my_int_equal");
    // init
    setReference();
  }

  ~DQMQualityTest()
  {
    delete my_ref;
    delete my_test;
    delete my_int_ref;
    delete my_int;
    delete chi2_test_;
    delete ks_test_;
    delete xrange_test_;
    delete yrange_test_;
    delete deadChan_test_;
    delete noisyChan_test_;
    delete equalH1_test_;
    delete equalInt_test_;
    delete meanNear_test_;
    delete emu_test_;
    delete my_testh2f;
    delete my_testprof;
    delete my_testprof2d;
    delete zrangeh2f_test_;
    delete zrangeprof_test_;
    delete zrangeprof2d_test_;
    delete poMPLandau_test_;
  }
  // N_ref: statistics for reference histogram
  // N_test: statistics for test histogram
  void generateData(unsigned N_ref = 10000, unsigned N_test = 1500)
  {
    // fill in reference histogram
    for(unsigned i = 0; i != N_ref; ++i)
      my_ref->Fill(gRandom->Gaus(mean, sigma));

    // fill in test histogram
    for(unsigned i = 0; i != N_test; ++i)
      my_test->Fill(gRandom->Gaus(mean, sigma));

    for(unsigned i = 0; i != N_test; ++i)
      my_testh2f->Fill(gRandom->Gaus(mean, sigma), 
		       gRandom->Gaus(mean, sigma));

    for(unsigned i = 0; i != N_test; ++i)
      my_testprof->Fill(gRandom->Gaus(mean, sigma), 
			TMath::Abs(gRandom->Gaus(mean, sigma)));

    for(unsigned i = 0; i != N_test; ++i)
      my_testprof2d->Fill(gRandom->Gaus(mean, sigma), 
			  gRandom->Gaus(mean, sigma), 
			  TMath::Abs(gRandom->Gaus(mean, sigma)));

    *my_int = sample_int_value;

    for( unsigned i = 0; i < N_test; ++i) {
      poTH1Landau_test_->Fill( gRandom->Landau( dLandauMP_, dLandauSigma_));
    }
  }

  // run tests, get probability, printout results
  void runTests(float * prob_chi2, float * prob_ks, float * prob_xrange, 
		float * prob_yrange, float * prob_deadChan, 
		float * prob_noisyChan, 
		float * probH1_equal, float* probInt_equal, float * prob_mean,
		float * probZH2, float * probZProf, float * probZProf2D)
  {
    std::cout << " Running test " << chi2_test_->getName() 
	      << " (Algorithm: " << chi2_test_->getAlgoName() << ") "
	      << std::endl;
    *prob_chi2 = chi2_test_->runTest(my_test);
    std::cout << " Chi2 Probability = " << *prob_chi2 << std::endl;
    //
    std::cout << " Running test " << ks_test_->getName() 
	      << " (Algorithm: " << ks_test_->getAlgoName() << ") " 
	      << std::endl;
    *prob_ks = ks_test_->runTest(my_test);
    std::cout << " Kolmogorov Probability = " << *prob_ks << std::endl;
    //
    std::cout << " Running test " << xrange_test_->getName() 
	      << " (Algorithm: " << xrange_test_->getAlgoName() << ") " 
	      << std::endl;
    *prob_xrange = xrange_test_->runTest(my_test);
    std::cout << " Entry fraction within allowed x-range = " 
	      << *prob_xrange << std::endl;
    //
    std::cout << " Running test " << yrange_test_->getName() 
	      << " (Algorithm: " << yrange_test_->getAlgoName() << ") " 
	      << std::endl;
    *prob_yrange = yrange_test_->runTest(my_test);
    std::cout << " Fraction of bin within allowed y-range = " 
	      << *prob_yrange << std::endl;
    showBadChannels(yrange_test_);
    //
    std::cout << " Running test " << deadChan_test_->getName() 
	      << " (Algorithm: " << deadChan_test_->getAlgoName() << ") " 
	      << std::endl;
    *prob_deadChan = deadChan_test_->runTest(my_test);
    std::cout << " Fraction of alive channels = " 
	      << *prob_deadChan << std::endl;
    showBadChannels(deadChan_test_);
    //
    std::cout << " Running test " << noisyChan_test_->getName() 
	      << " (Algorithm: " << noisyChan_test_->getAlgoName() << ") " 
	      << std::endl;
    *prob_noisyChan = noisyChan_test_->runTest(my_test);
    std::cout << " Fraction of quiet channels = " 
	      << *prob_noisyChan << std::endl;
    showBadChannels(noisyChan_test_);
    //
    std::cout << " Running test " << meanNear_test_->getName() 
	      << " (Algorithm: " << meanNear_test_->getAlgoName() << ") " 
	      << std::endl;
    *prob_mean = meanNear_test_->runTest(my_test);
    std::cout << " Probability that mean deviation is statistical fluctuation = " 
	      << *prob_mean << std::endl;
    //meanNear_test
    std::cout << " Running test " << emu_test_->getName() 
	      << " (Algorithm: " << emu_test_->getAlgoName() << ") " 
	      << std::endl;
    float emu_result = emu_test_->runTest(my_test);
    std::cout << " Result = " 
	      << emu_result << std::endl;
    // 
    std::cout << " Running test " << equalH1_test_->getName() 
	      << " (Algorithm: " << equalH1_test_->getAlgoName() << ") " 
	      << std::endl;
    *probH1_equal = equalH1_test_->runTest(my_test);
    std::cout << " Identical contents?"; 
    if(*probH1_equal == 1)
      std::cout << " Yes";
    else
      std::cout << " No";
    std::cout << std::endl;
    //
    showBadChannels(equalH1_test_);
    //
    std::cout << " Running test " << equalInt_test_->getName() 
	      << " (Algorithm: " << equalInt_test_->getAlgoName() << ") " 
	      << std::endl;
    *probInt_equal = equalInt_test_->runTest(my_int);
    std::cout << " Identical contents?"; 
    if(*probInt_equal == 1)
      std::cout << " Yes";
    else
      std::cout << " No";
    std::cout << std::endl;
    //
    std::cout << " Running test " << zrangeh2f_test_->getName()
              << " (Algorithm: " << zrangeh2f_test_->getAlgoName() << ") "
              << std::endl;
    *probZH2 = zrangeh2f_test_->runTest(my_testh2f);
    std::cout << " Result = " <<  *probZH2 << std::endl;
    showBadChannels(zrangeh2f_test_);
    // 
    std::cout << " Running test " << zrangeprof_test_->getName()
              << " (Algorithm: " << zrangeprof_test_->getAlgoName() << ") "
              << std::endl;
    *probZProf = zrangeprof_test_->runTest(my_testprof);
    std::cout << " Result = " << *probZProf << std::endl;
     showBadChannels(zrangeprof_test_);
    //
    std::cout << " Running test " << zrangeprof2d_test_->getName()
              << " (Algorithm: " << zrangeprof2d_test_->getAlgoName() 
	      << ") " << std::endl;
    *probZProf2D = zrangeprof2d_test_->runTest(my_testprof2d);
    std::cout << " Result = " << *probZProf2D << std::endl;
    showBadChannels(zrangeprof2d_test_);
    //
    std::cout << " Running test " << poMPLandau_test_->getName()
              << " (Algorithm: " << poMPLandau_test_->getAlgoName() << ") "
              << std::endl;
    *probZProf = poMPLandau_test_->runTest( poTH1Landau_test_);
    std::cout << " Result = " << *probZProf << std::endl;
    showBadChannels( poMPLandau_test_);
  }

 protected:

  float xmin_; float xmax_; // range for histograms
  
  TH1F * my_ref; // reference histogram
  TH1F * my_test; // test histogram
  int * my_int_ref; // reference integer
  int * my_int; // test integer

  TH1F  *poTH1Landau_test_; // test histogram for MostProbableLandau
  float dLandauMP_; 
  float dLandauSigma_;
  
  TH2F* my_testh2f;
  TProfile* my_testprof;
  TProfile2D* my_testprof2d;

  Comp2RefChi2 * chi2_test_; // chi2 test
  Comp2RefKolmogorov * ks_test_; // Kolmogorov test
  ContentsXRange * xrange_test_; // contents within x-range test
  ContentsYRange * yrange_test_;  // contents within y-range test
  DeadChannel * deadChan_test_;  // check for dead channels
  NoisyChannel * noisyChan_test_;  // check for noisy channels
  Comp2RefEqualH1 * equalH1_test_; // equality test for histograms
  Comp2RefEqualInt * equalInt_test_; // equality test for integers
  MeanWithinExpected * meanNear_test_; // mean-within-expected test
  AllContentWithinFixedRange *emu_test_; //EMU Test Function

  // contents within z-range tests
  ContentsTH2FWithinRange * zrangeh2f_test_; 
  ContentsProfWithinRange * zrangeprof_test_; 
  ContentsProf2DWithinRange * zrangeprof2d_test_;

  // MostProbables
  MostProbableLandau *poMPLandau_test_;

  // 
  void setReference()
  {
    if(chi2_test_)chi2_test_->setReference(my_ref);
    if(ks_test_)ks_test_->setReference(my_ref);
    if(equalH1_test_)equalH1_test_->setReference(my_ref);

    // set reference equal to test integer
    *my_int_ref = sample_int_value;
    if(equalInt_test_)equalInt_test_->setReference(my_int_ref);
  }

  // show channels that failed test
  void showBadChannels(QCriterion *qc)
  {
    std::vector<dqm::me_util::Channel> badChannels = qc->getBadChannels();
    if(!badChannels.empty())
      std::cout << " Channels that failed test " << qc->algoName() 
		<< ":\n";
    
    std::vector<dqm::me_util::Channel>::iterator it = badChannels.begin();
    while(it != badChannels.end())
      {
	std::cout << " Channel ("
                  << it->getBinX() << ","
                  << it->getBinY() << ","
                  << it->getBinZ()
		  << ") Contents: " << it->getContents()
		  << " +- " << it->getRMS() << std::endl;	
	++it;
      }
  }

};
