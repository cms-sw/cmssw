#ifndef Fireworks_Electron_ElectronsGlimpseProxyBuilder_h
#define Fireworks_Electron_ElectronsGlimpseProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     ElectronsGlimpseProxyBuilder
//
/**\class ElectronsGlimpseProxyBuilder ElectronsGlimpseProxyBuilder.h Fireworks/Calo/interface/ElectronsGlimpseProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: ElectronsGlimpseProxyBuilder.h,v 1.1 2008/06/19 06:57:28 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"

// forward declarations
class ElectronsGlimpseProxyBuilder : public FWGlimpseDataProxyBuilder
{

   public:
      ElectronsGlimpseProxyBuilder();
      virtual ~ElectronsGlimpseProxyBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      ElectronsGlimpseProxyBuilder(const ElectronsGlimpseProxyBuilder&); // stop default

      const ElectronsGlimpseProxyBuilder& operator=(const ElectronsGlimpseProxyBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
