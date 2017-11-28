#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <vector>
#include <thread>
#include <iostream>

XERCES_CPP_NAMESPACE_USE

void
testInit()
{
  cms::concurrency::xercesInitialize();
  {
      std::cerr << std::this_thread::get_id() << std::endl;
      XercesDOMParser parser;
  }
  cms::concurrency::xercesTerminate();
}

int
main()
{
  std::vector<std::thread> threads;
  threads.emplace_back(testInit);
  threads.emplace_back(testInit);
  threads.emplace_back(testInit);
  threads.emplace_back(testInit);

  for (auto &thread : threads) 
    thread.join();
}
