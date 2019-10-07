#include "DataFormats/HGCDigi/interface/HGCSample.h"
#include <iostream>
#include <assert.h>
#include <string>

// run for instance with:
//
//                         time HGCSampleTest  10000000000
//
// for a measureble amount of time taken


int main(int argc, char** argv) {

  std::cout << "Basic performance tests for HGCSample\n" << std::endl;
  std::cout << "num parameters entered: " << argc << std::endl;

  // first command line argument is the number of trials
  unsigned long int repetitions = 100;
  if (argc>1) repetitions  =  std::stoul (argv[1],nullptr,0);
  // second command line argument (whatever it is) will activate
  //                  the random choice of values for all inputs
  bool generateRandomValues = (argc > 2 ? true: false);


  uint32_t adc    = 124;
  uint32_t gain   = 1;
  uint32_t thrADC = 995;

  bool     thr    = adc > thrADC;
  uint32_t toa    = 0;
  bool     mode   = false;


  // do the trials
  unsigned long int u =0;
  for( ; u<repetitions; u++)
    {
      HGCSample aSample;

      // randomise all inputs, if chosen at the command line
      if(generateRandomValues){
	adc    = rand() %4046;
	gain   = rand() %4;
	thrADC = rand() %2;
	mode   = rand() %2;
	toa    = rand() %1024;
	std::cout << adc << "\t" << mode << std::endl;
      }


      // writing on an empty container first
      aSample.set(thr, mode, gain, toa, adc);
      assert( thr  == aSample.threshold() );
      assert( mode == aSample.mode() );
      assert( toa  == aSample.toa() );
      assert( adc  == aSample.data() );      
      
      HGCSample bSample;
      bSample.setThreshold(thr);
      bSample.setMode(mode);
      bSample.setGain(gain+100);
      bSample.setToA(toa+100);
      bSample.setData(adc+100);

      // cover the case where we write on a container with numbers already set
      bSample.setThreshold(thr);
      bSample.setMode(mode);
      bSample.setGain(gain);
      bSample.setToA(toa);
      bSample.setData(adc);
      assert( thr  == aSample.threshold() &&  thr  == bSample.threshold() );
      assert( mode == aSample.mode()      &&  mode == bSample.mode() );
      assert( gain == aSample.gain()      &&  gain == bSample.gain() );
      assert( toa  == aSample.toa()       &&  toa  == bSample.toa() );
      assert( adc  == aSample.data()      &&  adc  == bSample.data() );

    }  

  std::cout << "\nDone " << repetitions << "\t" << u<< ", ciao" << std::endl;

  return 0;
}
