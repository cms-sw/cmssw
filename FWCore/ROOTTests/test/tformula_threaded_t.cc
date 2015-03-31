#include "TFormula.h"
#include "TThread.h"
#include "TObject.h"
#include "TVirtualStreamerInfo.h"
#include <thread>
#include <memory>
#include <atomic>
#include <cassert>
#include <iostream>


void printHelp(const char* iName, int iDefaultNThreads)
{
  std::cout << iName <<" [number of threads] \n\n"
            <<"If no arguments are given "<<iDefaultNThreads<<" threads will be used"<<std::endl;
}

int parseOptionsForNumberOfThreads(int argc, char** argv)
{
  constexpr int kDefaultNThreads = 4;
  int returnValue = kDefaultNThreads;
  if( argc == 2 ) {
    if(strcmp("-h",argv[1]) ==0) {
      printHelp(argv[0],kDefaultNThreads);
      exit( 0 );
    }
    
    returnValue = atoi(argv[1]);
 }
  
  if( argc > 2) {
    printHelp(argv[0],kDefaultNThreads);
    exit(1);
  }
  return returnValue ;
}

int main(int argc, char** argv)
{
  const int kNThreads = parseOptionsForNumberOfThreads(argc, argv);

  std::atomic<int> canStart{kNThreads};
  std::vector<std::thread> threads;
  
  TThread::Initialize();
  //When threading, also have to keep ROOT from logging all TObjects into a list
  TObject::SetObjectStat(false);
  
  //Have to avoid having Streamers modify themselves after they have been used
  TVirtualStreamerInfo::Optimize(false);
  
  
  for(int i=0; i<kNThreads; ++i) {
    threads.emplace_back([i,&canStart]() {
        static thread_local TThread guard;
        --canStart;
        while( canStart > 0 ) {}
        
        TFormula f("testFormula","1./(1.+(4.61587e+06*(((1./(0.5*TMath::Max(1.e-6,x+1.)))-1.)/1.16042e+07)))");
        
        for(int i=0; i<100;++i) {
          double x = double(i)/100.;
          f.Eval(x);
        }
      });
  }
  canStart = true;
  
  for(auto& thread: threads) {
    thread.join();
  }
  
  return 0;
}
