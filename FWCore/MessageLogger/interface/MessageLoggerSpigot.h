#ifndef FWCore_MessageLogger_MessageLoggerSpigot_h
#define FWCore_MessageLogger_MessageLoggerSpigot_h

#include "boost/thread/thread.hpp"

namespace edm {

class MessageLoggerSpigot {
public:

  // ---  birth/death:
  MessageLoggerSpigot();
  ~MessageLoggerSpigot();

private:
  // --- no copying:
  MessageLoggerSpigot( MessageLoggerSpigot const & );
  void  operator= ( MessageLoggerSpigot const & );

  // --- data members:
  boost::thread  scribe;

}; // MessageLoggerSpigot


} // namespace edm



#endif // FWCore_MessageLogger_MessageLoggerSpigot_h
