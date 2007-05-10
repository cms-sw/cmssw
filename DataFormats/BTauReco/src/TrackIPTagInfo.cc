#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

using namespace reco;

TaggingVariableList TrackIPTagInfo::taggingVariables(void) const {
 TaggingVariableList vars;
 
   std::vector<Measurement1D>::const_iterator it=m_ip3d.begin();
   std::vector<Measurement1D>::const_iterator itEnd=m_ip3d.end();
   for(;it!=itEnd;++it)
    {
      vars.insert(TaggingVariable(reco::btau::trackSip3d,it->significance()));
    } 
   for(it=m_ip2d.begin(),itEnd=m_ip2d.end();it!=itEnd;++it)
    {
      vars.insert(TaggingVariable(reco::btau::trackSip2d,it->significance()));
    } 
 return vars;
}
