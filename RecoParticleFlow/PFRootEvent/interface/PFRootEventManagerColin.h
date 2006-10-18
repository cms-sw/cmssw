#ifndef Demo_PFRootEvent_PFRootEventManagerColin_h
#define Demo_PFRootEvent_PFRootEventManagerColin_h

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/EventColin.h"


class PFRootEventManagerColin : public PFRootEventManager {
  
 public:
  PFRootEventManagerColin(const char* file);

  ~PFRootEventManagerColin();

  bool processEntry(int entry);
  void write();


 private:
  TTree       *outTree_;
  EventColin  *event_;
  
};

#endif
