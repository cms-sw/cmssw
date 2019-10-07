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
  unsigned long int repetitions = 100;
  if (argc==2) repetitions =  std::stoul (argv[1],nullptr,0);//   ((int) argv[1]);

  uint32_t adc    = 124;
  uint32_t thrADC = 995;

  bool     thr    = adc > thrADC;
  uint32_t toa    = 0;
  bool     mode   = false;

  unsigned long int u =0;
  for(; u<repetitions; u++)
    {
      HGCSample aSample;
      // writing on an empty container first
      aSample.set(thr, mode, toa, adc);
      assert( thr  == aSample.threshold() );
      assert( mode == aSample.mode() );
      assert( toa  == aSample.toa() );
      assert( adc  == aSample.data() );      
      
      HGCSample bSample;
      bSample.setThreshold(thr);
      bSample.setMode(mode);
      bSample.setToA(toa+100);
      bSample.setData(adc+100);

      // cover the case where we write on a container with numbers already set
      bSample.setThreshold(thr);
      bSample.setMode(mode);
      bSample.setToA(toa);
      bSample.setData(adc);
      assert( thr  == aSample.threshold() &&  thr  == bSample.threshold() );
      assert( mode == aSample.mode()      &&  mode == bSample.mode() );
      assert( toa  == aSample.toa()       &&  toa  == bSample.toa() );
      assert( adc  == aSample.data()      &&  adc  == bSample.data() );

    }  

  std::cout << "\nDone " << repetitions << "\t" << u<< ", ciao" << std::endl;

  return 0;
}
