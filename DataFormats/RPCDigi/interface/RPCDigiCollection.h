#ifndef RPCDigi_RPCDigiCollection_h
#define RPCDigi_RPCDigiCollection_h
/** \class RPCDigiCollection
 *  
 *  \author Ilaria Segoni 
 *  \version $Id: RPCDigiCollection.h,v 1.2 2006/05/16 14:36:29 mmaggi Exp $
 *  \date 21 Apr 2005
 */

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<RPCDetId, RPCDigi> RPCDigiCollection;

#endif

