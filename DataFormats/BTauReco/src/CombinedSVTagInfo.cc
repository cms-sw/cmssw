#include "DataFormats/BTauReco/interface/CombinedSVTagInfo.h"

reco::CombinedSVTagInfo::CombinedSVTagInfo ( const reco::TaggingVariableList & l,
                                   double discriminator ) :
  vars_(l), discriminator_(discriminator)
{}

reco::CombinedSVTagInfo::CombinedSVTagInfo ()
{}

double reco::CombinedSVTagInfo::discriminator() const
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

void reco::CombinedSVTagInfo::setJetTag ( const reco::JetTagRef ref )
{
  basetag_=ref;
}
