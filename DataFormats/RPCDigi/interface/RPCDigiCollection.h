#ifndef RPCDigi_RPCDigiCollection_h
#define RPCDigi_RPCDigiCollection_h
/** \class RPCDigiCollection
 *
 *  \author Ilaria Segoni
 *  \date 21 Apr 2005
 */

#include <DataFormats/MuonData/interface/MuonDigiCollection.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/RPCDigi/interface/RPCDigi.h>

typedef MuonDigiCollection<RPCDetId, RPCDigi> RPCDigiCollection;

#endif
