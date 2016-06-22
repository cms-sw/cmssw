#include "CACell.h"


bool CACell::have_similar_curvature(const CACell* otherCell , const TrackingRegion& region) const
{
  float r1 = otherCell->get_inner_r();
  float r2 = get_inner_r();
  float r3 = get_outer_r();

  float deltaPhi1 = reco::deltaPhi(otherCell->get_inner_phi(), get_inner_phi());
  float deltaPhi2 = reco::deltaPhi(get_inner_phi(), get_outer_phi());

  float dphi_dr1 = deltaPhi1 / (r2 - r1);
  float dphi_dr2 = deltaPhi2 / (r3 - r2);

  return fabs((dphi_dr1 - dphi_dr2) / dphi_dr2) < 0.2f;
  //  bool haveSameSign = deltaPhi1*deltaPhi2 >= 0.0f;
  //  
  //  return deltaPhi1*deltaPhi2 >= 0.0f;
}

void CACell::find_ntuplets ( std::vector<CAntuplet>& foundNtuplets, CAntuplet& tmpNtuplet, const unsigned int minHitsPerNtuplet) const
{

  // the building process for a track ends if:
  // it has no right neighbor
  // it has no compatible neighbor
  // the ntuplets is then saved if the number of hits it contains is greater than a threshold
  
  if (theOuterNeighbors.size() == 0 )
  {
    if ( tmpNtuplet.size() >= minHitsPerNtuplet - 1)
      foundNtuplets.push_back(tmpNtuplet);
    else
      return;
  } else
  {
    unsigned int numberOfOuterNeighbors = theOuterNeighbors.size();
    for ( unsigned int i=0 ; i < numberOfOuterNeighbors; ++i)
    {
      tmpNtuplet.push_back((theOuterNeighbors[i]));
      theOuterNeighbors[i]->find_ntuplets(foundNtuplets, tmpNtuplet, minHitsPerNtuplet );
      tmpNtuplet.pop_back();
    }
  }
}

