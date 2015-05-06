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
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
//#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"
//#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"

#include "Fireworks/Core/interface/FWItemRandomAccessor.h"

REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<CSCRecHit2DCollection>,CSCRecHit2DCollection,"CSCRecHit2DCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<CSCSegmentCollection>,CSCSegmentCollection,"CSCSegmentCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<DTRecSegment2DCollection>,DTRecSegment2DCollection,"DTSegment2DCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<DTRecSegment4DCollection>,DTRecSegment4DCollection,"DTSegment4DCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<DTRecHitCollection>,DTRecHitCollection,"DTRecHitCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<RPCRecHitCollection>,RPCRecHitCollection,"RPCRecHitCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<GEMRecHitCollection>,GEMRecHitCollection,"GEMRecHitCollectionAccessor");
//REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<ME0RecHitCollection>,ME0RecHitCollection,"ME0RecHitCollectionAccessor");
//REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemRandomAccessor<ME0SegmentCollection>,ME0SegmentCollection,"ME0SegmentCollectionAccessor");

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

typedef FWItemMuonDigiAccessor<GEMDigiCollection, GEMDigi> GEMDigiAccessor;
REGISTER_TEMPLATE_FWITEMACCESSOR(GEMDigiAccessor, GEMDigiCollection, "GEMDigiCollectionAccessor");

typedef FWItemMuonDigiAccessor<GEMPadDigiCollection, GEMPadDigi> GEMPadDigiAccessor;
REGISTER_TEMPLATE_FWITEMACCESSOR(GEMPadDigiAccessor, GEMPadDigiCollection, "GEMPadDigiCollectionAccessor");

typedef FWItemMuonDigiAccessor<GEMCoPadDigiCollection, GEMCoPadDigi> GEMCoPadDigiAccessor;
REGISTER_TEMPLATE_FWITEMACCESSOR(GEMCoPadDigiAccessor, GEMCoPadDigiCollection, "GEMCoPadDigiCollectionAccessor");

typedef FWItemMuonDigiAccessor<ME0DigiPreRecoCollection, ME0DigiPreReco> ME0DigiPreRecoAccessor;
REGISTER_TEMPLATE_FWITEMACCESSOR(ME0DigiPreRecoAccessor, ME0DigiPreRecoCollection, "ME0DigiPreRecoCollectionAccessor");
