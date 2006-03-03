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
 * \version $Revision$
 *
 * $Id: Track.h,v 1.12 2006/03/01 12:23:40 llista Exp $
 *
 */
namespace reco {
  class Candidate;
}

/// return the helicity angle
double helicityAngle( const reco::Candidate & c );

#endif
