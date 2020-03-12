#include <string>

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/BTauReco/interface/IPTagInfo.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"

using namespace reco;

reco::btag::SortCriteria TrackSorting::getCriterium(const std::string &name) {
  using namespace reco::btag;
  if (name == "sip3dSig")
    return IP3DSig;
  if (name == "prob3d")
    return Prob3D;
  if (name == "sip2dSig")
    return IP2DSig;
  if (name == "prob2d")
    return Prob2D;
  if (name == "sip2dVal")
    return IP2DValue;

  throw cms::Exception("InvalidArgument") << "Identifier \"" << name << "\" does not represent a valid "
                                          << "track sorting criterium." << std::endl;
}
