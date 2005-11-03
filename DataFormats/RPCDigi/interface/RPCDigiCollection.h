#ifndef RPCDigi_RPCDigiCollection_h
#define RPCDigi_RPCDigiCollection_h

/** \class RPCDigiCollection
 *  ED
 *
 *  $Date: 2005/10/07 17:40:53 $
 *  $Revision: 1.1 $
 *  \author Stefano ARGIRO
 */

/** \class RPCDigiCollection
   
   \author Ilaria Segoni 
   \version $Id: RPCDigiCollection.h,v 1.1 2005/10/07 17:40:53 segoni Exp $
   \date 21 Apr 2005
*/

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<RPCDetId, RPCDigi> RPCDigiCollection;

#endif

