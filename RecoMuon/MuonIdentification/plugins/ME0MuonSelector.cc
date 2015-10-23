/** 
 *Filter to select me0Muons based on pulls and differences w.r.t. me0Segments
 *
 *
 */

#include "RecoMuon/MuonIdentification/interface/ME0MuonSelector.h"

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "DataFormats/Math/interface/deltaPhi.h"


namespace muon {
SelectionType selectionTypeFromString( const std::string &label )
{
  const static SelectionTypeStringToEnum selectionTypeStringToEnumMap[] = {
      { "All", All },
      { "VeryLoose", VeryLoose },
      { "Loose", Loose },
      { "Tight", Tight },
      { 0, (SelectionType)-1 }
   };

   SelectionType value = (SelectionType)-1;
   bool found = false;
   for(int i = 0; selectionTypeStringToEnumMap[i].label && (! found); ++i)
      if (! strcmp(label.c_str(), selectionTypeStringToEnumMap[i].label)) {
         found = true;
         value = selectionTypeStringToEnumMap[i].value;
      }

   // in case of unrecognized selection type
   if (! found) throw cms::Exception("MuonSelectorError") << label << " is not a recognized SelectionType";
   return value;
}
}


bool muon::isGoodMuon(const reco::ME0Muon& me0muon, SelectionType type)
{
  switch (type)
    {
    case muon::All:
      return true;
      break;
    case muon::VeryLoose:
      return isGoodMuon(me0muon,3,4,20,20,3.14); 
      break;
    case muon::Loose:
      return isGoodMuon(me0muon,3,2,3,2,0.5);
      break;
    case muon::Tight:
      return isGoodMuon(me0muon,3,2,3,2,0.15);
      break;
    default:
      return false;
    }
}




bool muon::isGoodMuon(const reco::ME0Muon& me0muon, double MaxPullX, double MaxDiffX, double MaxPullY, double MaxDiffY, double MaxDiffPhiDir )
{
  using namespace reco;

  ME0Segment thisSegment = me0muon.me0segment();


  LocalPoint r3FinalReco = me0muon.localTrackPosAtSurface();

  AlgebraicSymMatrix55 C = me0muon.localTrackCov();
  LocalPoint thisPosition(thisSegment.localPosition());

  
  double sigmax = sqrt(C[3][3]+thisSegment.localPositionError().xx() );      
  double sigmay = sqrt(C[4][4]+thisSegment.localPositionError().yy() );

  bool X_MatchFound = false, Y_MatchFound = false, Dir_MatchFound = false;
  
  
  if ( ( (std::abs(thisPosition.x()-r3FinalReco.x())/sigmax ) < MaxPullX ) || (std::abs(thisPosition.x()-r3FinalReco.x()) < MaxDiffX ) ) X_MatchFound = true;
  if ( ( (std::abs(thisPosition.y()-r3FinalReco.y())/sigmay ) < MaxPullY ) || (std::abs(thisPosition.y()-r3FinalReco.y()) < MaxDiffY ) ) Y_MatchFound = true;
  if ( std::abs(reco::deltaPhi(me0muon.localTrackMomAtSurface().phi(),thisSegment.localDirection().phi())) < MaxDiffPhiDir) Dir_MatchFound = true;

  return (X_MatchFound && Y_MatchFound && Dir_MatchFound);

}

