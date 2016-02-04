#ifndef FWCore_Utilities_Presence_h
#define FWCore_Utilities_Presence_h

// -*- C++ -*-

/*
  An interface class defining a presence.  A presence is an object that an 
  executable can instantiate at an early time in order to initialize 
  various things.  The destructor
  takes action to terminate the artifacts of the run() method.
  
  The prototypical use of this is to establish the MessageServicePresence.
  That class appears in MessageService, which is a plugin.  By using this
  abstract class, we can arrange that cmsRun, in Framework, has no link
  dependency on MessageService.  Instead, the MessageServicePresence is 
  dynamically (run-time) loaded.  
*/

namespace edm {

  class Presence {
  public:
    Presence() {}
    virtual ~Presence() = 0;
  };
}
#endif // FWCore_Utilities_Presence_h
