
#include "DQMServices/Core/interface/Standalone.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/QTest.h"

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
    // Process each file given as argument.
    edm::ParameterSet emptyps;
    std::vector<edm::ParameterSet> emptyset;
    edm::ServiceToken services(edm::ServiceRegistry::createSet(emptyset));

    edm::ServiceRegistry::Operate operate(services);

    //dbe_ = edm::Service<DQMStore>().operator->();
    DQMStore* dbe_ = new DQMStore(emptyps);	

    xmin_ = XMIN; xmax_ = XMAX;
    // distribution: gaussian w/ parameters: mean, sigma
    mean = (xmin_ + xmax_)/2.0;
    sigma = (xmax_ - xmin_)/6.0;

    // reference histogram
    dbe_->setCurrentFolder("/Reference");
    my_ref = dbe_->book1D("my_ref", "reference histo", NBINS, XMIN, XMAX);
    // test histogram
    dbe_->setCurrentFolder("/");
    my_test = dbe_->book1D("my_test", "test histo", NBINS, XMIN, XMAX);
    dbe_->setCurrentFolder("/");

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
//    emu_test_ = new AllContentWithinFixedRange("Ricks_test");

    // MostProbableLandau
    //poMPLandau_test_ = new MostProbableLandau( "mplandau");
    //poMPLandau_test_->setXMin( xmin_);
    //poMPLandau_test_->setXMax( xmax_);
    //poMPLandau_test_->setMostProbable( dLandauMP_);
    //poMPLandau_test_->setSigma( dLandauSigma_);

    // equality test for histograms
    equalH_test_ = new Comp2RefEqualH("my_histo_equal");
    // equality test for integers
    // equalInt_test_ = new Comp2RefEqualInt("my_int_equal");
    // init
  }

  ~DQMQualityTest()
  {
    delete chi2_test_;
    delete ks_test_;
    delete xrange_test_;
    delete yrange_test_;
    delete deadChan_test_;
    delete noisyChan_test_;
    delete equalH_test_;
    delete meanNear_test_;
//    delete emu_test_;
    delete dbe_;
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
  }

  // run tests, get probability, printout results
  void runTests(float * prob_chi2, float * prob_ks, float * prob_xrange, 
		float * prob_yrange, float * prob_deadChan, 
		float * prob_noisyChan, 
		float * probH_equal, float * prob_mean)
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
    //

//   std::cout << " Running test " << emu_test_->getName() 
//	      << " (Algorithm: " << emu_test_->getAlgoName() << ") " 
//	      << std::endl;
//    float emu_result = emu_test_->runTest(my_test);
//    std::cout << " Result = " 
//	      << emu_result << std::endl;
    // 
    std::cout << " Running test " << equalH_test_->getName() 
	      << " (Algorithm: " << equalH_test_->getAlgoName() << ") " 
	      << std::endl;
    *probH_equal = equalH_test_->runTest(my_test);
    std::cout << " Identical contents?"; 
    if(*probH_equal == 1)
      std::cout << " Yes";
    else
      std::cout << " No";
    std::cout << std::endl;
    //
    showBadChannels(equalH_test_);
  }

 protected:

  DQMStore* dbe_ ; 
  float xmin_; float xmax_; // range for histograms
  
  MonitorElement * my_ref; // reference histogram
  MonitorElement * my_test; // test histogram

  Comp2RefChi2 * chi2_test_; // chi2 test
  Comp2RefKolmogorov * ks_test_; // Kolmogorov test
  ContentsXRange * xrange_test_; // contents within x-range test
  ContentsYRange * yrange_test_;  // contents within y-range test
  DeadChannel * deadChan_test_;  // check for dead channels
  NoisyChannel * noisyChan_test_;  // check for noisy channels
  Comp2RefEqualH * equalH_test_; // equality test for histograms
  //Comp2RefEqualInt * equalInt_test_; // equality test for integers
  MeanWithinExpected * meanNear_test_; // mean-within-expected test
//  AllContentWithinFixedRange *emu_test_; //EMU Test Function

  // MostProbables
  // MostProbableLandau *poMPLandau_test_;

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
