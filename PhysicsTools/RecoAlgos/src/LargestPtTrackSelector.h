#ifndef RecoAlgos_LargestPtTrackSelector_h
#define RecoAlgos_LargestPtTrackSelector_h
/** \class LargestPtTrackSelector
 *
 * selects the fist N tracks with largest Pt
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SortTrackSelector.h,v 1.1 2006/07/21 10:27:05 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/SortTrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/PtComparator.h"
#include "DataFormats/TrackReco/interface/Track.h"

typedef SortTrackSelector<PtInverseComparator<reco::Track> > LargestPtTrackSelector;

#endif
