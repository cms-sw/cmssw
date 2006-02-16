#include "FWCore/MessageService/interface/MessageServicePresence.h"
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


MessageServicePresence::MessageServicePresence()
  : Presence()
            , scribe( ( (void) MessageLoggerQ::instance() // ensure Q's static data init'd
            , runMessageLoggerScribe  // start a new thread
          ) )
{
  //std::cout << "MessageLoggerSpigot ctor\n";
}


MessageServicePresence::~MessageServicePresence()
{
  MessageLoggerQ::END();
  scribe.join();
}

} // end of namespace service  
} // end of namespace edm  
