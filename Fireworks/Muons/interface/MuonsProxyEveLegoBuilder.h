#ifndef Fireworks_Muons_MuonsProxyEveLegoBuilder_h
#define Fireworks_Muons_MuonsProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     MuonsProxyEveLegoBuilder
//
/**\class MuonsProxyEveLegoBuilder MuonsProxyEveLegoBuilder.h Fireworks/Muons/interface/MuonsProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: MuonsProxyEveLegoBuilder.h,v 1.2 2008/07/08 06:59:21 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"

// forward declarations
class MuonsProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{

   public:
      MuonsProxyEveLegoBuilder();
      virtual ~MuonsProxyEveLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      MuonsProxyEveLegoBuilder(const MuonsProxyEveLegoBuilder&); // stop default

      const MuonsProxyEveLegoBuilder& operator=(const MuonsProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
