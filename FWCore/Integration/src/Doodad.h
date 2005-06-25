#ifndef FWCOREINTEGRATION_DOODAD_H
#define FWCOREINTEGRATION_DOODAD_H
// -*- C++ -*-
//
// Package:     FWCoreIntegration
// Class  :     Doodad
// 
/**\class Doodad Doodad.h FWCore/FWCoreIntegration/interface/Doodad.h

 Description: Dummy class used to test the EventSetup in the integration test

 Usage:
    Isn't actually usable :)

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 13:23:03 EDT 2005
// $Id$
//

// system include files

// user include files

// forward declarations
namespace edmreftest {
struct Doodad
{
   Doodad() : a() {}
   int a;
};
}

#endif /* FWCOREINTEGRATION_DOODAD_H */
