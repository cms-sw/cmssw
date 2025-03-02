#ifndef RPCDigi_RPCDigiPhase2Collection_h
#define RPCDigi_RPCDigiPhase2Collection_h
/** \class RPCDigiCollection
 *  
 *  \author Borislav Pavlov 
 *  \date 14 June 2024
 */

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/RPCDigi/interface/RPCDigiPhase2.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<RPCDetId, RPCDigiPhase2> RPCDigiPhase2Collection;

#endif
