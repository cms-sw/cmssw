#ifndef Fireworks_Core_FWRPZDataProxyBuilderBase_h
#define Fireworks_Core_FWRPZDataProxyBuilderBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZDataProxyBuilderBase
//
/**\class FWRPZDataProxyBuilderBase FWRPZDataProxyBuilderBase.h Fireworks/Core/interface/FWRPZDataProxyBuilderBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sat Jun 28 09:51:27 PDT 2008
// $Id: FWRPZDataProxyBuilderBase.h,v 1.3 2008/11/26 01:55:31 chrjones Exp $
//

// system include files
#include <set>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "TEveElement.h"

// user include files
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"

// forward declarations
class FWEventItem;
class FWRhoPhiZView;

class FWRPZDataProxyBuilderBase
{

public:
   FWRPZDataProxyBuilderBase();
   virtual ~FWRPZDataProxyBuilderBase();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setItem(const FWEventItem* iItem);

   void itemChanged(const FWEventItem*);
   void itemBeingDestroyed(const FWEventItem*);
   void modelChanges(const FWModelIds&);

   void setViews(std::vector<boost::shared_ptr<FWRhoPhiZView> >* iRhoPhoViews,
                 std::vector<boost::shared_ptr<FWRhoPhiZView> >* iRhoZViews);

   void attachToRhoPhiView(boost::shared_ptr<FWRhoPhiZView>);
   void attachToRhoZView(boost::shared_ptr<FWRhoPhiZView>);

   float layer() const {
      return m_layer;
   }

   static
   void setUserData(const FWEventItem* iItem,
                    TEveElementList* iElements,
                    std::vector<FWModelId>& iIds);

   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static
   std::string typeOfBuilder();

protected:
   std::vector<FWModelId>& ids() {
      return m_ids;
   }
   const FWEventItem* item() const {
      return m_item;
   }

   //Override these two functions if you need to handle model changes in a unique way
   virtual void modelChanges(const FWModelIds& iIds,
                             TEveElement* iElements );
   virtual void applyChangesToAllModels(TEveElement* iElements);
private:
   FWRPZDataProxyBuilderBase(const FWRPZDataProxyBuilderBase&); // stop default

   const FWRPZDataProxyBuilderBase& operator=(const FWRPZDataProxyBuilderBase&); // stop default

   virtual void itemChangedImp(const FWEventItem*) = 0;
   virtual void itemBeingDestroyedImp(const FWEventItem*) = 0;
   virtual void modelChangesImp(const FWModelIds&) = 0;
   virtual TEveElementList* getRhoPhiProduct() const =0;
   virtual TEveElementList* getRhoZProduct() const = 0;
   void addRhoPhiProj(TEveElement*);
   void addRhoZProj(TEveElement*);


   // ---------- member data --------------------------------
   const FWEventItem* m_item;
   std::vector<boost::shared_ptr<FWRhoPhiZView> >* m_rpviews;
   std::vector<boost::shared_ptr<FWRhoPhiZView> >* m_rzviews;

   TEveElementList m_rhoPhiProjs;
   TEveElementList m_rhoZProjs;
   std::vector<FWModelId> m_ids;

   float m_layer;

   bool m_modelsChanged;
};


#endif
