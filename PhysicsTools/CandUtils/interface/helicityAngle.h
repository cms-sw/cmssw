#ifndef CandUtils_helicityAngle_h
#define CandUtils_helicityAngle_h
/**  helicityAngle
 *
 * Utility function that returns the helicity angle
 * of two body decay. It is defined as the angle 
 * between the candidate momentum and one of the 
 * daughters momentum in the mother's center-of-mass
 * reference frame. 
 * This angle has a two-fold ambiguity (h, pi - h ), and
 * by convention the angle smaller than pi/2 is chosen.
 *
 * Asserts if the candidate does not have two daughters.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.5 $
 *
 * $Id: helicityAngle.h,v 1.5 2006/03/03 10:09:18 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"

/// return the helicity angle
double helicityAngle( const reco::Candidate & c );

#endif
