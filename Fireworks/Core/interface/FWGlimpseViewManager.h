#ifndef Fireworks_Core_FWGlimpseViewManager_h
#define Fireworks_Core_FWGlimpseViewManager_h
// -*- C++ -*-
// $Id: FWGlimpseViewManager.h,v 1.8 2009/03/11 21:16:21 amraktad Exp $

// system include files
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"

// forward declarations
class TList;
class FWGlimpseDataProxyBuilder;
class FWEventItem;
class FWGUIManager;
class TGFrame;
class FWGlimpseView;
class FWViewBase;
class TEveElementList;
class TEveSelection;
class FWSelectionManager;
class TEveWindowSlot;

class FWGlimpseViewManager : public FWViewManagerBase
{

public:
   FWGlimpseViewManager(FWGUIManager*);
   virtual ~FWGlimpseViewManager();

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
   FWGlimpseViewManager(const FWGlimpseViewManager&);    // stop default

   const FWGlimpseViewManager& operator=(const FWGlimpseViewManager&);    // stop default

   void makeProxyBuilderFor(const FWEventItem* iItem);
   void beingDestroyed(const FWViewBase*);

   // ---------- member data --------------------------------
   typedef  std::map<std::string,std::vector<std::string> > TypeToBuilders;
   TypeToBuilders m_typeToBuilders;
   std::vector<boost::shared_ptr<FWGlimpseDataProxyBuilder> > m_builders;

   std::vector<boost::shared_ptr<FWGlimpseView> > m_views;
   TEveElementList m_elements;

   TEveSelection* m_eveSelection;
   FWSelectionManager* m_selectionManager;

   FWEveValueScaler m_scaler;
};

#endif

