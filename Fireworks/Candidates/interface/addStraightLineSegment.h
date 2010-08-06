#ifndef Fireworks_Candidates_addStraightLineSegment_h
#define Fireworks_Candidates_addStraightLineSegment_h
// -*- C++ -*-
//
// Package:     Candidates
// Class  :     addStraightLineSegment
// 
/**\class addStraightLineSegment addStraightLineSegment.h Fireworks/Candidates/interface/addStraightLineSegment.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jan  6 16:41:33 EST 2009
// $Id$
//

// system include files

// user include files

// forward declarations
class TEveStraightLineSet;

namespace reco {
   class Candidate;
}

namespace fw {
   void addStraightLineSegment( TEveStraightLineSet * marker,
                               reco::Candidate const * cand,
                               double scale_factor = 2);
}
#endif
