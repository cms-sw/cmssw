#ifndef Fireworks_Core_FW3DLegoDataProxyBuilder_h
#define Fireworks_Core_FW3DLegoDataProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoDataProxyBuilder
//
/**\class FW3DLegoDataProxyBuilder FW3DLegoDataProxyBuilder.h Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sat Jan  5 15:02:03 EST 2008
// $Id: FW3DLegoDataProxyBuilder.h,v 1.10 2008/11/06 22:05:22 amraktad Exp $
//

// system include files
#include <vector>

// user include files
// #include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilderFactory.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"

// forward declarations
class FWEventItem;
class TH2;
class TObject;
class TEveElementList;
class TEveElement;
class FWModelId;
class TEveCaloDataHist;

namespace fw3dlego
{
  extern const double xbins[83];
}

class FW3DLegoDataProxyBuilder
{

   public:
      FW3DLegoDataProxyBuilder();
      virtual ~FW3DLegoDataProxyBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
      static std::string typeOfBuilder();

      // ---------- member functions ---------------------------
      void setItem(const FWEventItem* iItem);
      void setHaveAWindow(bool iFlag);
      virtual void build() = 0;

      //virtual void message( int type, int xbin, int ybin ){}
      void modelChanges(const FWModelIds&);
      void itemChanged(const FWEventItem*);

      virtual void attach(TEveElement* iElement,
                          TEveCaloDataHist* iHist)  = 0;

   protected:
      int legoRebinFactor() const {return 1;}
      const FWEventItem* item() const { return m_item; }
      std::vector<FWModelId>& ids() {
         return m_ids;
      }

      virtual void applyChangesToAllModels()=0;
   private:
        virtual void modelChangesImp(const FWModelIds&)=0;
        virtual void itemChangedImp(const FWEventItem*)=0;
        virtual void itemBeingDestroyedImp(const FWEventItem*);
      //virtual void build(const FWEventItem* iItem, TH2** product){}
      //virtual void build(const FWEventItem* iItem,
	///		 TEveElementList** product){}


      //Override this if you need to special handle selection or other changes
      virtual void itemBeingDestroyed(const FWEventItem*);

      FW3DLegoDataProxyBuilder(const FW3DLegoDataProxyBuilder&); // stop default

      const FW3DLegoDataProxyBuilder& operator=(const FW3DLegoDataProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      const FWEventItem* m_item;
      std::vector<FWModelId> m_ids;

      bool m_modelsChanged;
      bool m_haveViews;
      bool m_mustBuild;
};


#endif
