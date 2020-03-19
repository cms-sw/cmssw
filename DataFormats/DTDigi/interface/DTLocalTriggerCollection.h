#ifndef DTLocalTrigger_DTLocalTriggerCollection_h
#define DTLocalTrigger_DTLocalTriggerCollection_h

/** \class DTLocalTriggerCollection
 *  The collection containing DT Local Triggers in the event.
 *
 *  \author FR?
 */

#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/DTDigi/interface/DTLocalTrigger.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<DTChamberId, DTLocalTrigger> DTLocalTriggerCollection;

#endif
