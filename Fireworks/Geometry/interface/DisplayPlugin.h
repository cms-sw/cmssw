#ifndef Fireworks_Geometry_DisplayPlugin_h
#define Fireworks_Geometry_DisplayPlugin_h
// -*- C++ -*-
//
// Package:     Geometry
// Class  :     DisplayPlugin
// 
/**\class DisplayPlugin DisplayPlugin.h Fireworks/Geometry/interface/DisplayPlugin.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Mar 18 04:08:58 CDT 2010
// $Id: DisplayPlugin.h,v 1.1 2010/04/01 21:57:59 chrjones Exp $
//

// system include files

// user include files

// forward declarations
namespace edm { class EventSetup; }

namespace fireworks {
  namespace geometry {
    class DisplayPlugin
    {

    public:
      DisplayPlugin();
      virtual ~DisplayPlugin();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void run(const edm::EventSetup&) = 0;

    private:
      DisplayPlugin(const DisplayPlugin&); // stop default

      const DisplayPlugin& operator=(const DisplayPlugin&); // stop default

      // ---------- member data --------------------------------

    };
  }
}

#endif
