#include "FWCore/MessageLogger/interface/MessageLoggerSpigot.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageLoggerScribe.h"


#include <iostream>


namespace edm {


namespace  {
void
  runMessageLoggerScribe()
{
  MessageLoggerScribe  m;
  m.run();
}
}  // namespace


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


} // namespace edm
