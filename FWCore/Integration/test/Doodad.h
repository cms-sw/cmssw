#ifndef Integration_Doodad_h
#define Integration_Doodad_h
// -*- C++ -*-
//
// Package:     Integration
// Class  :     Doodad
// 
/**\class Doodad Doodad.h FWCore/Integration/interface/Doodad.h

 Description: Dummy class used to test the EventSetup in the integration test

 Usage:
    Isn't actually usable :)

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 13:23:03 EDT 2005
// $Id: Doodad.h,v 1.2 2006/02/13 23:01:31 wmtan Exp $
//

// system include files

// user include files

// forward declarations
namespace edmtest {
struct Doodad
{
   Doodad() : a() {}
   int a;
};
}

#endif
