#ifndef EDM_EVENTAUX_H
#define EDM_EVENTAUX_H

#include "FWCore/Framework/interface/ProcessNameList.h"
#include "FWCore/EDProduct/interface/CollisionID.h"

// Auxiliary event data that is persistent

namespace edm
{
  struct EventAux {
    EventAux() : process_history_(), id_() {}
    explicit EventAux(CollisionID id) : process_history_(), id_(id) {}
    ~EventAux() {}
    // most recently process that processed this event
    // is the last on the list, this defines what "latest" is
    ProcessNameList process_history_;
    // Collision ID
    CollisionID id_;
  };
}

#endif 
