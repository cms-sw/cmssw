#ifndef Fireworks_Core_FW3DLegoEveHistProxyBuilder_h
#define Fireworks_Core_FW3DLegoEveHistProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoEveHistProxyBuilder
//
/**\class FW3DLegoEveHistProxyBuilder FW3DLegoEveHistProxyBuilder.h Fireworks/Core/interface/FW3DLegoEveHistProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Jul  5 11:26:06 EDT 2008
// $Id: FW3DLegoEveHistProxyBuilder.h,v 1.2 2008/07/09 20:05:28 chrjones Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

// forward declarations
class TH2F;
class TEveCaloDataHist;

class FW3DLegoEveHistProxyBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      FW3DLegoEveHistProxyBuilder();
      virtual ~FW3DLegoEveHistProxyBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void attach(TEveElement* iElement,
                          TEveCaloDataHist* iHist);
      virtual void build();

   private:
      virtual void modelChangesImp(const FWModelIds&);
      virtual void itemChangedImp(const FWEventItem*);

      virtual void itemBeingDestroyedImp(const FWEventItem*);

      //virtual void applyChangesToAllModels() = 0;
      virtual void build(const FWEventItem* iItem, TH2F** product) = 0;

      FW3DLegoEveHistProxyBuilder(const FW3DLegoEveHistProxyBuilder&); // stop default

      const FW3DLegoEveHistProxyBuilder& operator=(const FW3DLegoEveHistProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      TH2F* m_hist;
      TEveCaloDataHist* m_data;
      Int_t m_sliceIndex;
};


#endif
