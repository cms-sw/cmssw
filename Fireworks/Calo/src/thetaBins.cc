// -*- C++ -*-
//
// Package:     Calo
// Class  :     thetaBins
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Dec 11 22:59:38 EST 2008
// $Id: thetaBins.cc,v 1.4 2010/06/07 18:58:16 matevz Exp $
//

// system include files
#include <math.h>

// user include files
#include "Fireworks/Calo/interface/thetaBins.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"


// NOTE:
//       Here we assume 72 bins in phi. At high eta we have only 36 and at the
//       very end 18 bins. These large bins are splited among smaller bins
//       decreasing energy in each entry by factor of 2 and 4 for 36 and 18 bin
//       cases. Other options will be implemented later
//
// http://ecal-od-software.web.cern.ch/ecal-od-software/documents/documents/cal_newedm_roadmap_v1_0.pdf
// Eta mapping:
//   ieta - [-41,-1]+[1,41] - total 82 bins
//   calo tower gives eta of the ceneter of each bin
//   size:
//      0.087 - [-20,-1]+[1,20]
//      the rest have variable size from 0.09-0.30
// Phi mapping:
//   iphi - [1-72]
//   calo tower gives phi of the center of each bin
//   for |ieta|<=20 phi bins are all of the same size
//      iphi 36-37 transition corresponds to 3.1 -> -3.1 transition
//   for 20 < |ieta| < 40
//      there are only 36 active bins corresponding to odd numbers
//      iphi 35->37, corresponds to 3.05 -> -3.05 transition
//   for |ieta| >= 40
//      there are only 18 active bins 3,7,11,15 etc
//      iphi 31 -> 35, corresponds to 2.79253 -> -3.14159 transition

namespace fireworks {
   std::vector<std::pair<double,double> >
   thetaBins()
   {
      const int n_bins = fw3dlego::xbins_n - 1;
      std::vector<std::pair<double,double> > thetaBins(n_bins);
      for (int i = 0; i < n_bins; ++i )
      {
         thetaBins[i].first  = 2*atan( exp(-fw3dlego::xbins[i]) );
         thetaBins[i].second = 2*atan( exp(-fw3dlego::xbins[i+1]) );
      }
      return thetaBins;
   }
}
