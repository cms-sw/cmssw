#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include <map>

using namespace reco;
using namespace std;

TaggingVariableList TrackIPTagInfo::taggingVariables(void) const {
 TaggingVariableList vars;
 
   std::vector<TrackIPData>::const_iterator it=m_data.begin();
   std::vector<TrackIPData>::const_iterator itEnd=m_data.end();
   for(;it!=itEnd;++it)
    {
      vars.insert(TaggingVariable(reco::btau::trackSip3d,it->ip3d.significance()));
      vars.insert(TaggingVariable(reco::btau::trackSip2d,it->ip2d.significance()));
    } 
 return vars;
}

TrackRefVector TrackIPTagInfo::sortedTracks(std::vector<size_t> indexes) const
{
 TrackRefVector tr;
 for(size_t i =0 ; i < indexes.size(); i++) tr.push_back(m_selectedTracks[indexes[i]]);
 return tr;
}

std::vector<size_t> TrackIPTagInfo::sortedIndexes(SortCriteria mode) const
{
 float cut=-1e99;
 if((mode == Prob3D || mode == Prob2D)) cut=1e99;
 return sortedIndexesWithCut(cut,mode);
}

std::vector<size_t> TrackIPTagInfo::sortedIndexesWithCut(float cut, SortCriteria mode) const
{
 multimap<float,size_t> sortedIdx;
 size_t nSelectedTracks = m_selectedTracks.size();
 std::vector<size_t> result;
 
//check if probabilities are available
 if((mode == Prob3D || mode == Prob2D) && ! hasProbabilities()) 
  {
   return result;
  }

 for(size_t i=0;i<nSelectedTracks;i++) 
  {
     float sortingKey;
     switch(mode)
     {
      case IP3DSig:
           sortingKey=m_data[i].ip3d.significance();
           break;
      case IP2DSig:
           sortingKey=m_data[i].ip2d.significance();
           break;
      case IP3DValue:
           sortingKey=m_data[i].ip3d.value();
           break;
      case IP2DValue:
           sortingKey=m_data[i].ip2d.value();
           break;
      case Prob3D:
           sortingKey=m_prob3d[i];
           break;
      case Prob2D:
           sortingKey=m_prob2d[i];
           break;

      default:
       sortingKey=i;
     }   
     sortedIdx.insert(pair<float,size_t>(sortingKey,i));
  }

//Descending: 
if(mode == IP3DSig || mode == IP2DSig ||mode ==  IP3DValue || mode == IP2DValue)
 { 
   for(multimap<float,size_t>::reverse_iterator it = sortedIdx.rbegin(); it!=sortedIdx.rend(); it++)
    if(it->first >= cut) result.push_back(it->second);
 } else
//Ascending:
 {
  for(multimap<float,size_t>::iterator it = sortedIdx.begin(); it!=sortedIdx.end(); it++)
    if(it->first <= cut) result.push_back(it->second);
 }
 return result;
}
