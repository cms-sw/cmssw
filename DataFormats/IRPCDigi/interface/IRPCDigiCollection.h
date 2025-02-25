#ifndef IRPCDigi_IRPCDigiCollection_h
#define IRPCDigi_IRPCDigiCollection_h
/** \class RPCDigiCollection
 *  
 *  \author Borislav Pavlov 
 *  \date 14 July 2021
 */

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/IRPCDigi/interface/IRPCDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<RPCDetId, IRPCDigi> IRPCDigiCollection;

#endif
