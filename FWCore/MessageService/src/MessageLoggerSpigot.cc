#include "FWCore/MessageService/interface/MessageLoggerSpigot.h"
#include "FWCore/MessageService/interface/MessageLoggerScribe.h"

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"

#include <iostream>


using namespace edm::service;


namespace  {
void
  runMessageLoggerScribe()
{
  MessageLoggerScribe  m;
  m.run();
}
}  // namespace

namespace edm {
namespace service {


MessageLoggerSpigot::MessageLoggerSpigot()
  : scribe( ( (void) MessageLoggerQ::instance() // ensure Q's static data init'd
            , runMessageLoggerScribe  // start a new thread
          ) )
{
  //std::cout << "MessageLoggerSpigot ctor\n";
}


MessageLoggerSpigot::~MessageLoggerSpigot()
{
  MessageLoggerQ::END();
  scribe.join();
}

} // end of namespace service  
} // end of namespace edm  
