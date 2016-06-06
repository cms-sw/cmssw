#include "CACell.h"


void CACell::evolve()
{

  hasSameStateNeighbors = 0;
  unsigned int numberOfNeighbors = theOuterNeighbors.size();
  
  for (unsigned int i =0 ; i< numberOfNeighbors; ++i)
  {

    if (theOuterNeighbors[i]->get_CA_state() == theCAState)
    {



      hasSameStateNeighbors = 1;

      break;
    }
  }

}



void CACell::check_alignment_and_tag(CACell* innerCell, const float pt_min)
{

  if (are_aligned_RZ(innerCell, pt_min) && have_similar_curvature(innerCell))
  {
    tag_as_inner_neighbor(innerCell);
    innerCell->tag_as_outer_neighbor(this);
  }
}

bool CACell::are_aligned_RZ(const CACell* otherCell, const float pt_min) const
{


  float r1 = otherCell->get_inner_r();   
  float r2 = get_inner_r();
  float r3 = get_outer_r();
    

  float z1 = otherCell->get_inner_z();
  float z2 = get_inner_z();
  float z3 = get_outer_z();

  float distance_13_squared = (r1-r3)*(r1-r3) + (z1-z3)*(z1-z3);  
  float tan_12_13 = 2*fabs(z1 * (r2 - r3) + z2 * (r3 - r1) +z3 * (r1 - r2))/distance_13_squared;
    
  return tan_12_13*pt_min <= 0.001f;

}

bool CACell::have_similar_curvature(const CACell* otherCell) const
{
  float r1 = otherCell->get_inner_r();   
  float r2 = get_inner_r();
  float r3 = get_outer_r();
    
  float deltaPhi1 = reco::deltaPhi(otherCell->get_inner_phi(), get_inner_phi());
  float deltaPhi2 = reco::deltaPhi(get_inner_phi(), get_outer_phi());
  
  float dphi_dr1 = deltaPhi1/(r2-r1);
  float dphi_dr2 = deltaPhi2/(r3-r2);

  return fabs(dphi_dr1 - dphi_dr2) < 0.25;

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
  //  bool hasOneCompatibleNeighbor = false;
    unsigned int numberOfOuterNeighbors = theOuterNeighbors.size();
    for ( unsigned int i=0 ; i < numberOfOuterNeighbors; ++i)
    {
//      if (tmpNtuplet.size() <= 2 )
  //    {
  //      hasOneCompatibleNeighbor = true;
        tmpNtuplet.push_back((theOuterNeighbors[i]));
        theOuterNeighbors[i]->find_ntuplets(foundNtuplets, tmpNtuplet, minHitsPerNtuplet );
        tmpNtuplet.pop_back();
//      }
    }

 //   if (!hasOneCompatibleNeighbor && tmpNtuplet.size() >= minHitsPerNtuplet - 1)
 //   {
   //   foundNtuplets.push_back(tmpNtuplet);
   // }
  }

}

