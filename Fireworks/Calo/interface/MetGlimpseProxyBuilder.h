#ifndef Fireworks_Calo_MetGlimpseProxyBuilder_h
#define Fireworks_Calo_MetGlimpseProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     MetGlimpseProxyBuilder
// 
/**\class MetGlimpseProxyBuilder MetGlimpseProxyBuilder.h Fireworks/Calo/interface/MetGlimpseProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: MetGlimpseProxyBuilder.h,v 1.1 2008/06/19 06:57:28 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"

// forward declarations
class MetGlimpseProxyBuilder : public FWGlimpseDataProxyBuilder
{

   public:
      MetGlimpseProxyBuilder();
      virtual ~MetGlimpseProxyBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();
   
      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product);

      MetGlimpseProxyBuilder(const MetGlimpseProxyBuilder&); // stop default

      const MetGlimpseProxyBuilder& operator=(const MetGlimpseProxyBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
