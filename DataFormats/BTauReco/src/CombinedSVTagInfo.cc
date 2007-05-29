#include "DataFormats/BTauReco/interface/CombinedSVTagInfo.h"

reco::CombinedSVTagInfo::CombinedSVTagInfo ( const reco::TaggingVariableList & l,
              double discriminator, const JetTracksAssociationRef & jtaRef ) :
  JTATagInfo(jtaRef), vars_(l), discriminator_(discriminator)
{}

reco::CombinedSVTagInfo::CombinedSVTagInfo ()
{}

float reco::CombinedSVTagInfo::discriminator() const
{
  return discriminator_;
}

const reco::TaggingVariableList & reco::CombinedSVTagInfo::variables() const
{
  return vars_;
}

reco::CombinedSVTagInfo * reco::CombinedSVTagInfo::clone() const
{
  return new reco::CombinedSVTagInfo (*this);
}

reco::CombinedSVTagInfo::~CombinedSVTagInfo()
{}
