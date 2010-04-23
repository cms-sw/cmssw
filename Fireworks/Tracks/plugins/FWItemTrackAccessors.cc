// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWItemTrackAccessor
//
// Implementation:
//
// Original Author:  Tom McCauley
//         Created:  Thu Feb 18 15:19:44 EDT 2008
// $Id: FWItemTrackAccessors.cc,v 1.2 2010/03/11 14:21:47 mccauley Exp $
//

#include <assert.h>
#include "Reflex/Object.h"
#include "TClass.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "Fireworks/Core/interface/FWItemRandomAccessor.h"

REGISTER_TEMPLATE_FWITEMACCESSOR( FWItemDetSetAccessor<edm::DetSetVector<SiStripDigi> >,edm::DetSetVector<SiStripDigi>, "SiStripDigiCollectionAccessor" );
REGISTER_TEMPLATE_FWITEMACCESSOR( FWItemDetSetAccessor<edm::DetSetVector<PixelDigi> >, edm::DetSetVector<PixelDigi>, "SiPixelDigiCollectionAccessor" );
REGISTER_TEMPLATE_FWITEMACCESSOR( FWItemDetSetAccessor<edmNew::DetSetVector<SiStripCluster> >, edmNew::DetSetVector<SiStripCluster>, "SiStripClusterCollectionAccessor" );

