// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemCSCSegmentAccessor
//
// Implementation:
//     An example of how to write a plugin based FWItemAccessorBase derived class.
//
// Original Author:  Giulio Eulisse
//         Created:  Thu Feb 18 15:19:44 EDT 2008
// $Id: FWItemMuonAccessors.cc,v 1.7 2010/06/18 12:44:05 yana Exp $
//

// system include files
#include <assert.h>

#include "TClass.h"

// user include files
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"

#include "Fireworks/Core/interface/FWItemRandomAccessor.h"

REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<CSCRecHit2DCollection>,CSCRecHit2DCollection,"CSCRecHit2DCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<CSCSegmentCollection>,CSCSegmentCollection,"CSCSegmentCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<DTRecSegment2DCollection>,DTRecSegment2DCollection,"DTSegment2DCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<DTRecSegment4DCollection>,DTRecSegment4DCollection,"DTSegment4DCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<DTRecHitCollection>,DTRecHitCollection,"DTRecHitCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<RPCRecHitCollection>,RPCRecHitCollection,"RPCRecHitCollectionAccessor");

typedef FWItemMuonDigiAccessor<DTDigiCollection,DTDigi> DTDigiAccessor;
REGISTER_TEMPLATE_FWITEMACCESSOR(DTDigiAccessor, DTDigiCollection, "DTDigiCollectionAccessor");

typedef FWItemMuonDigiAccessor<CSCWireDigiCollection, CSCWireDigi> CSCWireDigiAccessor;
REGISTER_TEMPLATE_FWITEMACCESSOR(CSCWireDigiAccessor, CSCWireDigiCollection, "CSCWireDigiCollectionAccessor");

typedef FWItemMuonDigiAccessor<CSCStripDigiCollection, CSCStripDigi> CSCStripDigiAccessor;
REGISTER_TEMPLATE_FWITEMACCESSOR(CSCStripDigiAccessor, CSCStripDigiCollection, "CSCStripDigiCollectionAccessor");

typedef FWItemMuonDigiAccessor<RPCDigiCollection, RPCDigi> RPCDigiAccessor;
REGISTER_TEMPLATE_FWITEMACCESSOR(RPCDigiAccessor, RPCDigiCollection, "RPCDigiCollectionAccessor");

typedef FWItemMuonDigiAccessor<CSCRPCDigiCollection, CSCRPCDigi> CSCRPCDigiAccessor;
REGISTER_TEMPLATE_FWITEMACCESSOR(CSCRPCDigiAccessor, CSCRPCDigiCollection, "CSCRPCDigiCollectionAccessor");
