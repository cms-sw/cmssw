#ifndef Fireworks_Electron_PatElectronsGlimpseProxyBuilder_h
#define Fireworks_Electron_PatElectronsGlimpseProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     PatElectronsGlimpseProxyBuilder
//
/**\class PatElectronsGlimpseProxyBuilder PatElectronsGlimpseProxyBuilder.h Fireworks/Calo/interface/PatElectronsGlimpseProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PatElectronsGlimpseProxyBuilder.h,v 1.1 2008/09/26 07:42:08 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"

// forward declarations
class PatElectronsGlimpseProxyBuilder : public FWGlimpseDataProxyBuilder
{

   public:
      PatElectronsGlimpseProxyBuilder();
      virtual ~PatElectronsGlimpseProxyBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      PatElectronsGlimpseProxyBuilder(const PatElectronsGlimpseProxyBuilder&); // stop default

      const PatElectronsGlimpseProxyBuilder& operator=(const PatElectronsGlimpseProxyBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
