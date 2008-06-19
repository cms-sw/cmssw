#ifndef Fireworks_Core_FWGlimpseViewManager_h
#define Fireworks_Core_FWGlimpseViewManager_h
// -*- C++ -*-
// $Id: FWGlimpseViewManager.h,v 1.4 2008/06/10 19:28:01 chrjones Exp $

// system include files
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"

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

struct FWGlimpseModelProxy
{
   boost::shared_ptr<FWGlimpseDataProxyBuilder>   builder;
   TEveElementList*                           product; //owned by builder
   bool ignore;
   FWGlimpseModelProxy():product(0),ignore(false){}
   FWGlimpseModelProxy(boost::shared_ptr<FWGlimpseDataProxyBuilder> iBuilder):
    builder(iBuilder),product(0),ignore(false){}
};

class FWGlimpseViewManager : public FWViewManagerBase
{

   public:
      FWGlimpseViewManager(FWGUIManager*);
      virtual ~FWGlimpseViewManager();

      // ---------- const member functions ---------------------
      std::vector<std::string> purposeForType(const std::string& iTypeName) const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void newEventAvailable();

      virtual void newItem(const FWEventItem*);

      FWViewBase* buildView(TGFrame* iParent);
      
      //connect to ROOT signals
      void selectionAdded(TEveElement*);
      void selectionRemoved(TEveElement*);
      void selectionCleared();

   protected:
      virtual void modelChangesComing();
      virtual void modelChangesDone();

   private:
      FWGlimpseViewManager(const FWGlimpseViewManager&); // stop default

      const FWGlimpseViewManager& operator=(const FWGlimpseViewManager&); // stop default
   
      void makeProxyBuilderFor(const FWEventItem* iItem);
      void itemChanged(const FWEventItem*);

      // ---------- member data --------------------------------
      typedef  std::map<std::string,std::vector<std::string> > TypeToBuilders;
      TypeToBuilders m_typeToBuilders;
      std::vector<FWGlimpseModelProxy> m_modelProxies;

      std::vector<boost::shared_ptr<FWGlimpseView> > m_views;
      TEveElementList* m_elements;
      
      bool m_itemChanged;
      TEveSelection* m_eveSelection;
      FWSelectionManager* m_selectionManager;
};

#endif

