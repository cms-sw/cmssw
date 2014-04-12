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
     vars.insert(btau::trackJetDistVal, data->distanceToJetAxis.value(), true);
     vars.insert(btau::trackJetDistSig, data->distanceToJetAxis.significance(), true);
     vars.insert(btau::trackGhostTrackDistVal, data->distanceToGhostTrack.value(), true);
     vars.insert(btau::trackGhostTrackDistSig, data->distanceToGhostTrack.significance(), true);
     vars.insert(btau::trackGhostTrackWeight, data->ghostTrackWeight, true);
     vars.insert(btau::trackChi2, track->normalizedChi2(), true);
     vars.insert(btau::trackNTotalHits, track->hitPattern().numberOfValidHits(), true);
     vars.insert(btau::trackNPixelHits, track->hitPattern().numberOfValidPixelHits(), true);
   } 
  vars.finalize();
  return vars;
}

TrackRefVector TrackIPTagInfo::sortedTracks(const std::vector<size_t>& indexes) const
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
     sortedIdx.insert(std::pair<float,size_t>(sortingKey,i));
  }

//Descending: 
if(mode == IP3DSig || mode == IP2DSig ||mode ==  IP3DValue || mode == IP2DValue)
 { 
   for(std::multimap<float,size_t>::reverse_iterator it = sortedIdx.rbegin(); it!=sortedIdx.rend(); it++)
    if(it->first >= cut) result.push_back(it->second);
 } else
//Ascending:
 {
  for(std::multimap<float,size_t>::iterator it = sortedIdx.begin(); it!=sortedIdx.end(); it++)
    if(it->first <= cut) result.push_back(it->second);
 }
 return result;
}

bool TrackIPTagInfo::passVariableJTA(const variableJTAParameters &params, double jetpT, double trackpT, double jettrackdR) {

  bool pass = false;

  // intermediate pt range (between min_pT and max_pT), apply variable JTA !
  if ( jetpT > params.min_pT && jetpT < params.max_pT ) {
    double deltaRfunction_highpt = -jetpT * params.a_dR + params.b_dR;
    double ptfunction_highpt = jetpT * params.a_pT + params.b_pT;
    
    if (jettrackdR < deltaRfunction_highpt
	&&
	trackpT > ptfunction_highpt) 
      pass = true;
    
    //  cout << "TrackIPTagInfo: passVariableJTA: dR and TrackpT " << jettrackdR << " " << trackpT << endl;
    
    //high pt range, apply fixed default cuts
  }else if (jetpT > params.max_pT ) {
    if (jettrackdR < params.max_pT_dRcut
	&&
	trackpT > params.max_pT_trackPTcut)
      pass = true;

    // low pt range, apply fixed default cuts
  }else {
    if (jettrackdR < params.min_pT_dRcut)
      pass = true;
  }
  
  return pass;
}

std::vector<bool> TrackIPTagInfo::variableJTA(const variableJTAParameters &params) const{
  
  std::vector<bool> result;

  //Jet parameters
  double jetpT = jet()->pt();
  math::XYZVector jetDir = jet()->momentum().Unit();

  for(size_t  i = 0 ; i<  m_selectedTracks.size();  i++) {
    
    //Track parameters
    TrackRef track = m_selectedTracks[i];    
    double trackpT = track->pt();
    math::XYZVector trackMom = track->momentum();

    // do the math in passVariableJTA
    result.push_back(passVariableJTA( params, jetpT, trackpT, ROOT::Math::VectorUtil::DeltaR(trackMom, jetDir)));

  }  
  
  return result;
}
