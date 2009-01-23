#ifndef Fireworks_Calo_thetaBins_h
#define Fireworks_Calo_thetaBins_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     thetaBins
//
/**\class thetaBins thetaBins.h Fireworks/Calo/interface/thetaBins.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Dec 11 22:59:29 EST 2008
// $Id: thetaBins.h,v 1.1 2008/12/12 04:16:10 chrjones Exp $
//

// system include files
#include <vector>
#include <algorithm>

// user include files

// forward declarations

namespace fireworks {
   std::vector<std::pair<double,double> > thetaBins();
}


#endif
