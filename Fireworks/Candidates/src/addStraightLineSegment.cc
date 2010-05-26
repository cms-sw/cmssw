// -*- C++ -*-
//
// Package:     Candidates
// Class  :     addStraightLineSegment
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jan  6 16:41:37 EST 2009
// $Id$
//

// system include files
#include "TEveStraightLineSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

// user include files
#include "Fireworks/Candidates/interface/addStraightLineSegment.h"


//
// constants, enums and typedefs
//

void fw::addStraightLineSegment( TEveStraightLineSet * marker,
                                reco::Candidate const * cand,
                                double scale_factor)
{
   double phi = cand->phi();
   double theta = cand->theta();
   double size = cand->pt() * scale_factor;
   marker->AddLine( 0, 0, 0, size * cos(phi)*sin(theta), size *sin(phi)*sin(theta), size*cos(theta));
}
