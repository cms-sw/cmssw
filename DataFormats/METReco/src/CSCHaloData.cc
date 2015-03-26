#include "DataFormats/METReco/interface/CSCHaloData.h"

/*
  [class]:  CSCHaloData
  [authors]: R. Remington, The University of Florida
  [description]: See CSCHaloData.h 
  [date]: October 15, 2009
*/

using namespace reco;
CSCHaloData::CSCHaloData()
{
  nTriggers_PlusZ = 0;
  nTriggers_MinusZ = 0 ;
  nTracks_PlusZ = 0 ;
  nTracks_MinusZ = 0;
  HLTAccept=false;

  nOutOfTimeTriggers_PlusZ=0;
  nOutOfTimeTriggers_MinusZ=0;
  nOutOfTimeHits = 0 ;

  nTracks_Small_dT = 0;
  nTracks_Small_beta =0;
  nTracks_Small_dT_Small_beta = 0;

  // MLR
  nFlatHaloSegments = 0;
  // End MLR
}

int CSCHaloData::NumberOfHaloTriggers(HaloData::Endcap z) const
{
  if( z == HaloData::plus )
    return nTriggers_PlusZ;
  else if( z == HaloData::minus )
    return nTriggers_MinusZ;
  else 
    return nTriggers_MinusZ + nTriggers_PlusZ;
}

short int CSCHaloData::NumberOfOutOfTimeTriggers(HaloData::Endcap z ) const
{
  if( z == HaloData::plus  ) 
    return nOutOfTimeTriggers_PlusZ;
  else if( z == HaloData::minus ) 
    return nOutOfTimeTriggers_MinusZ;
  else
    return nOutOfTimeTriggers_PlusZ+nOutOfTimeTriggers_MinusZ;
}

int CSCHaloData::NumberOfHaloTracks(HaloData::Endcap z) const 
{
  int n = 0 ;
  for(unsigned int i = 0 ; i < TheTrackRefs.size() ; i++ )
    {
      edm::Ref<reco::TrackCollection> iTrack( TheTrackRefs[i] ) ;
      // Does the track go through both endcaps ? 
      bool Traversing =  (iTrack->outerPosition().z() > 0 &&  iTrack->innerPosition().z() < 0) ||  (iTrack->outerPosition().z() < 0 &&  iTrack->innerPosition().z() > 0);
      // Does the track go through only +Z endcap ?
      bool PlusZ =  (iTrack->outerPosition().z() > 0 && iTrack->innerPosition().z() > 0 ) ;
      // Does the track go through only -Z endcap ? 
      bool MinusZ = (iTrack->outerPosition().z()< 0 && iTrack->innerPosition().z() < 0) ;

      if( (z == HaloData::plus) && ( PlusZ || Traversing) ) 
	n++;
      else if( (z == HaloData::minus) && ( MinusZ || Traversing ) )
	n++;
      else if( (z == HaloData::both) && (PlusZ || MinusZ || Traversing) ) 
	n++ ;
    }
  return n;
}
