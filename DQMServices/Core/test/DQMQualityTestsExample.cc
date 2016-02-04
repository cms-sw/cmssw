#include "DQMServices/Core/test/DQMQualityTest.hpp"

int main(int argc, char** argv)
{
  // # of bins and range for histogram (see header file for details)
  const unsigned NBINS = 100; 
  const float XMIN = -3.0; const float XMAX = 3.0;

  // create instance of test-suite
  DQMQualityTest * dqm_test = new DQMQualityTest(NBINS, XMIN, XMAX);
  // generate random data
  dqm_test->generateData();
  
  float prob_chi2, prob_ks, prob_xrange, prob_yrange, prob_deadChan, 
    prob_noisyChan, probH_equal, prob_mean = 0;
  // run tests, obtain probabilities
  dqm_test->runTests(&prob_chi2, &prob_ks, &prob_xrange, &prob_yrange,
		     &prob_deadChan, &prob_noisyChan, &probH_equal, 
		     &prob_mean);

  delete dqm_test;
  return 0;
}
