#include "TProfile.h"
#include "TThread.h"
#include "TObject.h"
#include "TVirtualStreamerInfo.h"
#include <thread>
#include <memory>
#include <vector>
#include <atomic>
#include <sstream>
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

  std::atomic<bool> canStart{false};
  std::vector<std::unique_ptr<TProfile>> profiles;
  std::vector<std::thread> threads;
  
  TH1::AddDirectory(kFALSE);
  
  TThread::Initialize();
  //When threading, also have to keep ROOT from logging all TObjects into a list
  TObject::SetObjectStat(false);
  
  //Have to avoid having Streamers modify themselves after they have been used
  TVirtualStreamerInfo::Optimize(false);

  
  for(int i=0; i<kNThreads; ++i) {
    std::ostringstream s;
    profiles.push_back(std::unique_ptr<TProfile>(new TProfile(s.str().c_str(),s.str().c_str(), 100,10,11,0,10)));
    profiles.back()->SetBit(TH1::kCanRebin);
    auto profile = profiles.back().get();
    threads.emplace_back([i,profile,&canStart]() {
        static thread_local TThread guard;
        while(not canStart) {}
        for(int x=10; x>0; --x) {
          for(int y=0; y<20; ++y) {
            profile->Fill(double(x), double(y),1.);
          }
        }
      });
  }
  canStart = true;

  for(auto& thread: threads) {
    thread.join();
  }
  
  return 0;
}
