#ifndef Fireworks_Core_FWDataProxyBuilder_h
#define Fireworks_Core_FWDataProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDataProxyBuilder
// 
/**\class FWDataProxyBuilder FWDataProxyBuilder.h Fireworks/Core/interface/FWDataProxyBuilder.h

 Description: Builds Proxies of Data used by a View

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Dec  6 17:49:40 PST 2007
// $Id$
//

// system include files

// user include files

// forward declarations
namespace fwlite {
  class Event;
}

class TEveElementList;

class FWDataProxyBuilder
{

   public:
      FWDataProxyBuilder();
      virtual ~FWDataProxyBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void build(const fwlite::Event* iEvent,
			 TEveElementList** oList) = 0 ;

   private:
      FWDataProxyBuilder(const FWDataProxyBuilder&); // stop default

      const FWDataProxyBuilder& operator=(const FWDataProxyBuilder&); // stop default

      // ---------- member data --------------------------------

};


#endif
