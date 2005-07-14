#ifndef FWCOREINTEGRATION_WHATSIT_H
#define FWCOREINTEGRATION_WHATSIT_H
// -*- C++ -*-
//
// Package:     Integration
// Class  :     WhatsIt
// 
/**\class WhatsIt WhatsIt.h FWCore/Integration/interface/WhatsIt.h

 Description: Dummy class used to test the EventSetup in the integration test

 Usage:
    Not actually usable :)

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 13:23:12 EDT 2005
// $Id: WhatsIt.h,v 1.1 2005/06/25 01:20:41 chrjones Exp $
//

// system include files

// user include files

// forward declarations

namespace edmreftest {
struct WhatsIt
{
   WhatsIt() : a() {}
   int a;
};
}

#endif /* FWCOREINTEGRATION_WHATSIT_H */
