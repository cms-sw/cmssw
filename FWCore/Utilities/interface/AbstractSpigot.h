#ifndef FWCore_Utilities_AbstractSpigot_h
#define FWCore_Utilities_AbstractSpigot_h

// -*- C++ -*-

/*
  An interface class defining a spigot.  A spigot is an object that an 
  executable can instantiate at an early time in order to initialize 
  various things and to set in motion a run() method.  The destructor
  takes action to terminate the artifacts of the run() method.
  
  The prototypical use of this is to establish the MessageLoggerSpigot.
  That class appears in MessageService, which is a plugin.  Bu using this
  abstract class, we can arrange that cmsRun, in Framework, has no link
  dependency on MessageService.  Instead, the MessageLoggerSpigot is 
  dynamically (run-time) loaded.  
*/

namespace edm {

class AbstractSpigot
{
public:
  AbstractSpigot() { }
  virtual ~AbstractSpigot() = 0;
  virtual void run() = 0;
};

}
#endif // FWCore_Utilities_AbstractSpigot_h
