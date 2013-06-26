#ifndef Fireworks_Core_FWEveViewManager_h
#define Fireworks_Core_FWEveViewManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveViewManager
// 
/**\class FWEveViewManager FWEveViewManager.h Fireworks/Core/interface/FWEveViewManager.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones, Alja Mrak-Tadel
//         Created:  Thu Mar 18 14:12:45 CET 2010
// $Id: FWEveViewManager.h,v 1.14 2013/01/21 20:33:28 chrjones Exp $
//

// system include files
#include <vector>
#include <map>
#include <set>
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWViewType.h"

// forward declarations
class TEveCompund;
class TEveScene;
class TEveWindowSlot;
class FWViewBase;
class FWEveView;
class FWProxyBuilderBase;
class FWGUIManager;
class FWInteractionList;

typedef std::set<FWModelId> FWModelIds;

class FWEveViewManager : public FWViewManagerBase
{
private:
   struct BuilderInfo
   {
      std::string m_name;
      int         m_viewBit;

      BuilderInfo(std::string name, int viewBit) :
         m_name(name),
         m_viewBit(viewBit)
      {}
   };

public:
   FWEveViewManager(FWGUIManager*);
   virtual ~FWEveViewManager();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void newItem(const FWEventItem*);
   virtual void removeItem(const FWEventItem*);
   virtual void eventBegin();
   virtual void eventEnd();
   virtual void setContext(const fireworks::Context*);

   void highlightAdded(TEveElement*);
   void selectionAdded(TEveElement*);
   void selectionRemoved(TEveElement*);
   void selectionCleared();

   FWTypeToRepresentations supportedTypesAndRepresentations() const;

protected:
   virtual void modelChangesComing();
   virtual void modelChangesDone();
   virtual void colorsChanged();

private:
   FWEveViewManager(const FWEveViewManager&); // stop default
   const FWEveViewManager& operator=(const FWEveViewManager&); // stop default

   FWViewBase* buildView(TEveWindowSlot* iParent, const std::string& type);
   FWEveView*  finishViewCreate     (boost::shared_ptr<FWEveView>);

   void beingDestroyed(const FWViewBase*);
   void modelChanges(const FWModelIds& iIds);
   void itemChanged(const FWEventItem*);
   bool haveViewForBit (int) const;
   void globalEnergyScaleChanged();

   // ---------- member data --------------------------------
   
   typedef std::map<std::string,  std::vector<BuilderInfo> >  TypeToBuilder;
   typedef std::vector<boost::shared_ptr<FWProxyBuilderBase> >  BuilderVec;   
   typedef BuilderVec::iterator BuilderVec_it;
   typedef std::vector<boost::shared_ptr<FWEveView > >::iterator EveViewVec_it;
   
   TypeToBuilder            m_typeToBuilder;

   std::map<int, BuilderVec> m_builders; // key is viewer bit

   std::vector< std::vector<boost::shared_ptr<FWEveView> > >  m_views;

   std::map<const FWEventItem*,FWInteractionList*>  m_interactionLists;
};


#endif
