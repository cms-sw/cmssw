#ifndef TrackIPData_h
#define TrackIPData_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"

namespace reco {


struct TrackIPData
{
  Measurement1D impactParameter3D;
  Measurement1D impactParameter2D;
  //float decayLen;
  //float decayLenError;
};

}
#endif

