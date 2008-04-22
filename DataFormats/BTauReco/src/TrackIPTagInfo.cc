#include <cmath>
#include <map>

#include <Math/VectorUtil.h>

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

using namespace reco;
using namespace std;

static double etaRel(const math::XYZVector &dir, const math::XYZVector &track)
{
  double momPar = dir.Dot(track);
  double energy = sqrt(track.Mag2() + ROOT::Math::Square(0.13957));
  return 0.5 * log((energy + momPar) / (energy - momPar));
}

TaggingVariableList TrackIPTagInfo::taggingVariables(void) const {
  TaggingVariableList vars;

  math::XYZVector jetDir = jet()->momentum().Unit();
  bool havePv = primaryVertex().isNonnull();
  GlobalPoint pv;
  if (havePv)
    pv = GlobalPoint(primaryVertex()->x(),
                     primaryVertex()->y(),
                     primaryVertex()->z());

  std::vector<size_t> indexes = sortedIndexes(); // use default criterium
  for(std::vector<size_t>::const_iterator it = indexes.begin();
      it != indexes.end(); ++it)
   {
     using namespace ROOT::Math;
     TrackRef track = m_selectedTracks[*it];
     const TrackIPData *data = &m_data[*it];
     math::XYZVector trackMom = track->momentum();
     double trackMag = std::sqrt(trackMom.Mag2());

     vars.insert(btau::trackMomentum, trackMag, true);
     vars.insert(btau::trackEta, trackMom.Eta(), true);
     vars.insert(btau::trackEtaRel, etaRel(jetDir, trackMom), true);
     vars.insert(btau::trackPtRel, VectorUtil::Perp(trackMom, jetDir), true);
     vars.insert(btau::trackPPar, jetDir.Dot(trackMom), true);
     vars.insert(btau::trackDeltaR, VectorUtil::DeltaR(trackMom, jetDir), true);
     vars.insert(btau::trackPtRatio, VectorUtil::Perp(trackMom, jetDir) / trackMag, true);
     vars.insert(btau::trackPParRatio, jetDir.Dot(trackMom) / trackMag, true);
     vars.insert(btau::trackSip3dVal, data->ip3d.value(), true);
     vars.insert(btau::trackSip3dSig, data->ip3d.significance(), true);
     vars.insert(btau::trackSip2dVal, data->ip2d.value(), true);
     vars.insert(btau::trackSip2dSig, data->ip2d.significance(), true);
     vars.insert(btau::trackDecayLenVal, havePv ? (data->closestToJetAxis - pv).mag() : -1.0, true);
     vars.insert(btau::trackJetDist, data->distanceToJetAxis, true);
     vars.insert(btau::trackFirstTrackDist, data->distanceToFirstTrack, true);
     vars.insert(btau::trackChi2, track->normalizedChi2(), true);
     vars.insert(btau::trackNTotalHits, track->hitPattern().numberOfValidHits(), true);
     vars.insert(btau::trackNPixelHits, track->hitPattern().numberOfValidPixelHits(), true);
   } 
  vars.finalize();
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
