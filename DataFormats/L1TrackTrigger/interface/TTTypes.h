/*! \brief   Definition of all the relevant data types
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_TYPES_H
#define L1_TRACK_TRIGGER_TYPES_H

/// Standard CMS Formats
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"

/// The reference types
typedef edm::Ref< edm::DetSetVector< PixelDigi >, PixelDigi > Ref_PixelDigi_;

#endif

