#include "FWCore/MessageLogger/interface/MessageLoggerSpigot.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "boost/thread/thread.hpp"
#include <iostream>

namespace edm {

// TEMPORARY function to thread out:
void
  messageLoggerScribe() {std::cout << "=== messageLoggerScribe()\n";}


MessageLoggerSpigot::MessageLoggerSpigot()
  :scribe( messageLoggerScribe )
{
  std::cout << "=== MessageLoggerSpigot ctor body\n";
}


MessageLoggerSpigot::~MessageLoggerSpigot()
{
  std::cout << "=== MessageLoggerSpigot dtor entered\n";
  MessageLoggerQ::END();
  scribe.join();
  std::cout << "=== MessageLoggerSpigot dtorafter join()\n";
}

} // namespace edm
