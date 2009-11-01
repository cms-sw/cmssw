#ifndef Fireworks_Core_FW3DViewManager_h
#define Fireworks_Core_FW3DViewManager_h
// -*- C++ -*-
// $Id: FW3DViewManager.h,v 1.8 2009/04/07 14:05:46 chrjones Exp $

// system include files
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class TList;
class FW3DDataProxyBuilder;
class FWEventItem;
class FWGUIManager;
class FW3DView;
class FWViewBase;
class TEveElementList;
class TEveSelection;
class FWSelectionManager;
class TEveCaloDataHist;
class TEveCalo3D;
class TEveWindowSlot;

class FW3DViewManager : public FWViewManagerBase
{

public:
   FW3DViewManager(FWGUIManager*);
   virtual ~FW3DViewManager();

   // ---------- const member functions ---------------------
   FWTypeToRepresentations supportedTypesAndRepresentations() const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void newItem(const FWEventItem*);

   FWViewBase* buildView(TEveWindowSlot* iParent);

   //connect to ROOT signals
   void selectionAdded(TEveElement*);
   void selectionRemoved(TEveElement*);
   void selectionCleared();

protected:
   virtual void modelChangesComing();
   virtual void modelChangesDone();
   virtual void colorsChanged();

private:
   FW3DViewManager(const FW3DViewManager&);    // stop default

   const FW3DViewManager& operator=(const FW3DViewManager&);    // stop default

   void makeProxyBuilderFor(const FWEventItem* iItem);
   void beingDestroyed(const FWViewBase*);

   // ---------- member data --------------------------------
   typedef  std::map<std::string,std::vector<std::string> > TypeToBuilders;
   TypeToBuilders m_typeToBuilders;
   std::vector<boost::shared_ptr<FW3DDataProxyBuilder> > m_builders;

   std::vector<boost::shared_ptr<FW3DView> > m_views;
   FWEvePtr<TEveElementList> m_elements;
   TEveCaloDataHist* m_caloData;
   TEveCalo3D* m_calo3d;

   TEveSelection* m_eveSelection;
   FWSelectionManager* m_selectionManager;
};

#endif

