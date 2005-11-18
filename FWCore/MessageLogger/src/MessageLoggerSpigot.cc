#include "FWCore/MessageLogger/interface/MessageLoggerSpigot.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageLoggerScribe.h"


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
  :scribe( runMessageLoggerScribe )  // starts a new thread
{ }


MessageLoggerSpigot::~MessageLoggerSpigot()
{
  MessageLoggerQ::END();
  scribe.join();
}


} // namespace edm
