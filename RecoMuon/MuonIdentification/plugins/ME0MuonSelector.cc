/** 
 *Filter to select me0Muons based on pulls and differences w.r.t. me0Segments
 *
 *
 */

#include "RecoMuon/MuonIdentification/interface/ME0MuonSelector.h"

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"


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


bool muon::isGoodMuon(edm::ESHandle<ME0Geometry> me0Geom, const reco::ME0Muon& me0muon, SelectionType type)
{
  switch (type)
    {
    case muon::All:
      return true;
      break;
    case muon::VeryLoose:
      return isGoodMuon(me0Geom, me0muon,3,4,20,20,3.14); 
      break;
    case muon::Loose:
      return isGoodMuon(me0Geom, me0muon,3,2,3,2,0.5);
      break;
    case muon::Tight:
      return isGoodMuon(me0Geom, me0muon,3,2,3,2,0.15);
      break;
    default:
      return false;
    }
}




bool muon::isGoodMuon(edm::ESHandle<ME0Geometry> me0Geom, const reco::ME0Muon& me0muon, double MaxPullX, double MaxDiffX, double MaxPullY, double MaxDiffY, double MaxDiffPhiDir )
{
  ME0Segment thisSegment = me0muon.me0segment();
  
  ME0DetId id = thisSegment.me0DetId();

  auto roll = me0Geom->etaPartition(id); 
  
  float zSign  = me0muon.globalTrackMomAtSurface().z()/fabs(me0muon.globalTrackMomAtSurface().z());
  if ( zSign * roll->toGlobal(thisSegment.localPosition()).z() < 0 ) return false;
	
  //GlobalPoint r3FinalReco_glob(r3FinalReco_globv.x(),r3FinalReco_globv.y(),r3FinalReco_globv.z());

  LocalPoint r3FinalReco = roll->toLocal(me0muon.globalTrackPosAtSurface());
  LocalVector p3FinalReco=roll->toLocal(me0muon.globalTrackMomAtSurface());

  LocalPoint thisPosition(thisSegment.localPosition());
  LocalVector thisDirection(thisSegment.localDirection().x(),thisSegment.localDirection().y(),thisSegment.localDirection().z());  //FIXME

  //The same goes for the error
  AlgebraicMatrix thisCov(4,4,0);   
  for (int i = 1; i <=4; i++){
    for (int j = 1; j <=4; j++){
      thisCov(i,j) = thisSegment.parametersError()(i,j);
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////


  LocalTrajectoryParameters ltp(r3FinalReco,p3FinalReco,me0muon.trackCharge());
  JacobianCartesianToLocal jctl(roll->surface(),ltp);
  AlgebraicMatrix56 jacobGlbToLoc = jctl.jacobian(); 

  AlgebraicMatrix55 Ctmp =  (jacobGlbToLoc * me0muon.trackCov()) * ROOT::Math::Transpose(jacobGlbToLoc); 
  AlgebraicSymMatrix55 C;  // I couldn't find any other way, so I resort to the brute force
  for(int i=0; i<5; ++i) {
    for(int j=0; j<5; ++j) {
      C[i][j] = Ctmp[i][j]; 
      
    }
  }  

  Double_t sigmax = sqrt(C[3][3]+thisSegment.localPositionError().xx() );      
  Double_t sigmay = sqrt(C[4][4]+thisSegment.localPositionError().yy() );

  bool X_MatchFound = false, Y_MatchFound = false, Dir_MatchFound = false;
  
  
  if ( ( (fabs(thisPosition.x()-r3FinalReco.x())/sigmax ) < MaxPullX ) || (fabs(thisPosition.x()-r3FinalReco.x()) < MaxDiffX ) ) X_MatchFound = true;
  if ( ( (fabs(thisPosition.y()-r3FinalReco.y())/sigmay ) < MaxPullY ) || (fabs(thisPosition.y()-r3FinalReco.y()) < MaxDiffY ) ) Y_MatchFound = true;
  
  if ( fabs(me0muon.globalTrackMomAtSurface().phi()-roll->toGlobal(thisSegment.localDirection()).phi()) < MaxDiffPhiDir) Dir_MatchFound = true;

  return (X_MatchFound && Y_MatchFound && Dir_MatchFound);

}

