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
#include <cassert>

#include "TClass.h"

// user include files
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

#include "Fireworks/Core/interface/FWItemRandomAccessor.h"

REGISTER_TEMPLATE_FWITEMACCESSOR(BXVectorAccessor<BXVector<l1t::HGCalTriggerCell>>,
                                 BXVector<l1t::HGCalTriggerCell>,
                                 "HGCalTriggerCellCollectionAccessor");

REGISTER_TEMPLATE_FWITEMACCESSOR(BXVectorAccessor<BXVector<l1t::HGCalMulticluster>>,
                                 BXVector<l1t::HGCalMulticluster>,
                                 "HGCalTriggerClusterCollectionAccessor");
