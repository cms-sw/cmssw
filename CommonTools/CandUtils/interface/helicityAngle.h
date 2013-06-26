#ifndef CandUtils_helicityAngle_h
#define CandUtils_helicityAngle_h
/**  helicityAngle
 *
 * Utility function that returns the helicity angle
 * It is defined as the angle between the candidate 
 * momentum and one of the daughters momentum in the 
 * mother's center-of-mass reference frame. 
 * This angle has a two-fold ambiguity (h, pi - h ), and
 * by convention the angle smaller than pi/2 is chosen.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: helicityAngle.h,v 1.1 2009/02/26 09:17:34 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"

/// return the helicity angle for two particles. 
double helicityAngle( const reco::Candidate & mother, const reco::Candidate & daughter);

/// return the helicity angle of a two body decay with daughter automatically retreived
/// Note: asserts if the candidate does not have two daughters
double helicityAngle( const reco::Candidate & c );

#endif
