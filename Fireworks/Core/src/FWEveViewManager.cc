// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveViewManager
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones, Alja Mrak-Tadel
//         Created:  Thu Mar 18 14:11:32 CET 2010
//

// system include files

#include <boost/bind.hpp>


// user include files

// For optimized redraw of Eve views
#define protected public
#define private   public
#include "TEveManager.h"
#undef private
#undef protected

#include "TEveSelection.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TEveCalo.h"
#include "TEveGedEditor.h"
#include "TGListTree.h"
#include "TGeoManager.h"
#include "TExMap.h"
#include "TEnv.h"

#include "Fireworks/Core/interface/FWEveViewManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWInteractionList.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"

// PB
#include "Fireworks/Core/interface/FWEDProductRepresentationChecker.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWProxyBuilderFactory.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"

// viewes
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWISpyView.h"
#include "Fireworks/Core/interface/FW3DView.h"
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/FWEveLegoView.h"
#include "Fireworks/Core/interface/FWHFView.h"
#include "Fireworks/Core/interface/FWRPZView.h"

#include "Fireworks/Core/interface/FWTGLViewer.h"
bool FWEveViewManager::s_syncAllViews = false;

class FWViewContext;

// sentry class block TEveSelection signals when call TEveSelection::Remove/AddElement and
// when process its callbacks
class EveSelectionSentry {
public:
   EveSelectionSentry()
   {
      m_blocked = gEve->GetSelection()->BlockSignals(true);
   }
   ~EveSelectionSentry()
   {
     gEve->GetSelection()->BlockSignals(m_blocked);
   }
private:
   bool m_blocked;
};

//
//
// constants, enums and typedefs
//
//
// constructors and destructor
//
FWEveViewManager::FWEveViewManager(FWGUIManager* iGUIMgr) :
   FWViewManagerBase()
{
     
   // builders
   std::set<std::string> builders;
   
   std::vector<edmplugin::PluginInfo> available = FWProxyBuilderFactory::get()->available();
   std::transform(available.begin(),
                  available.end(),
                  std::inserter(builders,builders.begin()),
                  boost::bind(&edmplugin::PluginInfo::name_,_1));
   
   if(edmplugin::PluginManager::get()->categoryToInfos().end()!=edmplugin::PluginManager::get()->categoryToInfos().find(FWProxyBuilderFactory::get()->category())) {
      available = edmplugin::PluginManager::get()->categoryToInfos().find(FWProxyBuilderFactory::get()->category())->second;
      std::transform(available.begin(),
                     available.end(),
                     std::inserter(builders,builders.begin()),
                     boost::bind(&edmplugin::PluginInfo::name_,_1));
   }
   
   
   for(std::set<std::string>::iterator it = builders.begin(), itEnd=builders.end();
       it!=itEnd;
       ++it) {
      std::string::size_type first = it->find_first_of('@')+1;
      std::string purpose = it->substr(first,it->find_last_of('@')-first);
      
      first = it->find_last_of('@')+1;
      std::string view_str =  it->substr(first,it->find_last_of('#')-first);
      int viewTypes = atoi(view_str.c_str());
      std::string fullName = *it;
      m_typeToBuilder[purpose].push_back(BuilderInfo(*it, viewTypes));
   }
   
   m_views.resize(FWViewType::kTypeSize); 
   
   // view construction called via GUI mng
   FWGUIManager::ViewBuildFunctor f =  boost::bind(&FWEveViewManager::buildView, this, _1, _2);
   for (int i = 0; i < FWViewType::kTypeSize; i++)
   {
      if ( i == FWViewType::kTable || i == FWViewType::kTableHLT || i == FWViewType::kTableL1)
         continue;
      iGUIMgr->registerViewBuilder(FWViewType::idToName(i), f);
   }
   
   // signal
   gEve->GetHighlight()->SetPickToSelect(TEveSelection::kPS_Master);
   TEveSelection* eveSelection = gEve->GetSelection();
   eveSelection->SetPickToSelect(TEveSelection::kPS_Master);  
   eveSelection->Connect("SelectionAdded(TEveElement*)","FWEveViewManager",this,"selectionAdded(TEveElement*)");
   eveSelection->Connect("SelectionRepeated(TEveElement*)","FWEveViewManager",this,"selectionAdded(TEveElement*)");
   eveSelection->Connect("SelectionRemoved(TEveElement*)","FWEveViewManager",this,"selectionRemoved(TEveElement*)");
   eveSelection->Connect("SelectionCleared()","FWEveViewManager",this,"selectionCleared()");

   gEve->GetHighlight()->Connect("SelectionAdded(TEveElement*)","FWEveViewManager",this,"highlightAdded(TEveElement*)");
   gEve->GetHighlight()->Connect("SelectionRepeated(TEveElement*)","FWEveViewManager",this,"highlightAdded(TEveElement*)");

   TGeoManager::SetVerboseLevel(0);
}

FWEveViewManager::~FWEveViewManager()
{
}

//
// member functions
//

//______________________________________________________________________________

/**
   Helper function to add products from a given builder to a given view.
   FWRPZView is a base class for all projected views 
    (name derives from  FWViewType::RPhi and FWViewType::RhoZ projection).
  */
void
addElements(const FWEventItem *item, FWEveView *view, 
            int viewType, TEveElementList* product)
{
   if (FWViewType::isProjected(viewType))
   {
      FWRPZView* rpzView = dynamic_cast<FWRPZView*>(view);
      assert(rpzView);
      rpzView->importElements(product, item->layer(), rpzView->eventScene());
   }
   else
   {
      view->eventScene()->AddElement(product);
   }
}

/** This  is  invoked  when  a  new  item  is  created
    by  the  FWEventItemsManager.  The  workflow  is  the  following

   1.  First  we  check  if  we  have  a  builder  info  for  the  given  purpose  of  the
       item.  We  return  simply  if  we  don't.
   2.  We  iterate  over  all  the  proxy  builder  registered  for  the  given
       purpose  and  create  a  new  one  for  this  given  item.
   3.  Interaction  lists  are  set  up  in  case  the  proxy  builder  does  not  handle
       interaction  by  itself.
   4.  We  then  iterate  on  the  various  supported  views  and  add  elements  to  them,
       making  sure  that  we  handle  the  case  in  which  those  elements  are  not
       unique  among  all  the  views.
  */

void
FWEveViewManager::newItem(const FWEventItem* iItem)
{
   TypeToBuilder::iterator itFind = m_typeToBuilder.find(iItem->purpose());
   
   if (itFind == m_typeToBuilder.end())
      return;

   std::vector<BuilderInfo>& blist = itFind->second;

   std::string bType; bool bIsSimple;
   for (size_t bii = 0, bie = blist.size(); bii != bie; ++bii)
   {
      // 1.
      BuilderInfo &info = blist[bii];
      info.classType(bType, bIsSimple);
      if (bIsSimple) 
      {
         unsigned int distance=1;
         edm::TypeWithDict modelType( *(iItem->modelType()->GetTypeInfo()));
         if (!FWSimpleRepresentationChecker::inheritsFrom(modelType, bType,distance))
         {
            // printf("PB does not matche itemType (%s) !!! EDproduct %s %s\n", info.m_name.c_str(), iItem->modelType()->GetTypeInfo()->name(), bType.c_str() );
            continue;
         }
      }
      else {
         std::string itype = iItem->type()->GetTypeInfo()->name();
         if (itype != bType) {
            // printf("PB does not match modeType (%s)!!! EDproduct %s %s\n", info.m_name.c_str(), itype.c_str(), bType.c_str() );
            continue;
         }
      }

      std::string builderName = info.m_name;
      int builderViewBit =  info.m_viewBit;
      
      FWProxyBuilderBase* builder = 0;
      try
      {
         builder = FWProxyBuilderFactory::get()->create(builderName);

      }
      catch (std::exception& exc)
      {
         fwLog(fwlog::kWarning) << "FWEveViewManager::newItem ignoring the following exception (probably edmplugincache mismatch):"
                                << std::endl << exc.what();
      }
      if (!builder)
         continue;

      // 2.
      // printf("FWEveViewManager::makeProxyBuilderFor NEW builder %s \n", builderName.c_str());
      
      boost::shared_ptr<FWProxyBuilderBase> pB(builder);
      builder->setItem(iItem);
      iItem->changed_.connect(boost::bind(&FWEveViewManager::modelChanges,this,_1));
      iItem->goingToBeDestroyed_.connect(boost::bind(&FWEveViewManager::removeItem,this,_1));
      iItem->itemChanged_.connect(boost::bind(&FWEveViewManager::itemChanged,this,_1));

      // 3.
      // This calud be opaque to the user. I would pass a reference to the m_interactionLists to
      // FWProxyBuilderBase::setInteractionList and handle different case differently.
      if (builder->willHandleInteraction() == false)
      {
         typedef std::map<const FWEventItem*, FWInteractionList*>::iterator Iterator;
         std::pair<Iterator, bool> t = m_interactionLists.insert(std::make_pair(iItem,
                                                                                (FWInteractionList*)0));

         if (t.second == true)
            t.first->second = new FWInteractionList(iItem);
         //  printf(">>> builder %s add list %p \n", iItem->name().c_str(), il); fflush(stdout);
         builder->setInteractionList(t.first->second, iItem->purpose());
      }
      
      builder->setHaveWindow(haveViewForBit(builderViewBit));
      
      // 4.
      for (size_t viewType = 0; viewType < FWViewType::kTypeSize; ++viewType)
      {
         if (((1 << viewType) & builderViewBit) == 0)
            continue;
         
         FWViewType::EType type = (FWViewType::EType) viewType;
         
         // printf("%s builder %s supportsd view %s \n",  iItem->name().c_str(), builderName.c_str(), FWViewType::idToName(viewType).c_str());
         if (builder->havePerViewProduct((FWViewType::EType) viewType))
         { 
            for (size_t i = 0, e = m_views[viewType].size(); i != e; ++i)
            {
               FWEveView *view = m_views[viewType][i].get();
               TEveElementList* product = builder->createProduct(type, 
                                                                 view->viewContext());
               addElements(iItem, view, viewType, product);
            }
         }
         else 
         {
            TEveElementList* product = builder->createProduct(type, 0);
         
            for (size_t i = 0, e = m_views[viewType].size(); i != e; ++i)
               addElements(iItem, m_views[viewType][i].get(), viewType, product);
         }
      }
      
      m_builders[builderViewBit].push_back(pB);
   } // loop views
}

//______________________________________________________________________________
FWViewBase*
FWEveViewManager::buildView(TEveWindowSlot* iParent, const std::string& viewName)
{
   FWViewType::EType type = FWViewType::kTypeSize;
   for (int i = 0; i < FWViewType::kTypeSize; ++i)
   {
      if (viewName == FWViewType::idToName(i))
      {
         type = FWViewType::EType(i);
         break;
      }
   }

   boost::shared_ptr<FWEveView> view;
   switch(type)
   {
      case FWViewType::k3D:
         view.reset(new FW3DView(iParent, type));
         break;
      case FWViewType::kISpy:
         view.reset(new FWISpyView(iParent, type));
         break;
      case FWViewType::kRhoPhi:
      case FWViewType::kRhoZ:
      case FWViewType::kRhoPhiPF:
         view.reset(new FWRPZView(iParent, type));
         break;
      case FWViewType::kLego:
      case FWViewType::kLegoPFECAL:
         view.reset(new FWEveLegoView(iParent, type));
         break;
      case FWViewType::kLegoHF:
         view.reset(new FWHFView(iParent, type));
         break;
      case FWViewType::kGlimpse:
         view.reset(new FWGlimpseView(iParent, type));
         break;
      default:
         break;
   }

   m_views[type].push_back(boost::shared_ptr<FWEveView> (view));
   return finishViewCreate(m_views[type].back());
}

FWEveView*
FWEveViewManager::finishViewCreate(boost::shared_ptr<FWEveView> view)
{
   // printf("new view %s added \n", view->typeName().c_str());
   gEve->DisableRedraw();

   // set geometry and calo data
   view->setContext(context()); 

   FWColorManager::setColorSetViewer(view->viewerGL(),  context().colorManager()->background());

   // set proxies have a window falg
   int viewerBit = 1 << view->typeId();   
   if (m_views[view->typeId()].size() == 1)
   {
      for ( std::map<int, BuilderVec>::iterator i = m_builders.begin(); i!=  m_builders.end(); ++i)
      {
         int builderViewBit = i->first;
         BuilderVec& bv =  i->second;
         if (viewerBit == (builderViewBit & viewerBit))
         {
            for(BuilderVec_it bIt = bv.begin(); bIt != bv.end(); ++bIt)
            {
               (*bIt)->setHaveWindow(true);
            }
         }
      }
   }
    
   FWRPZView* rpzView = dynamic_cast<FWRPZView*>(view.get());
   for ( std::map<int, BuilderVec>::iterator i = m_builders.begin(); i!=  m_builders.end(); ++i)
   {
      int builderViewBit = i->first;
      BuilderVec& bv =  i->second;
      if (viewerBit == (builderViewBit & viewerBit))
      { 
         for(BuilderVec_it bIt = bv.begin(); bIt != bv.end(); ++bIt)
         {
            // it is ok to call create even for shared productsm since
            // builder map key garanties that
            TEveElementList* product = (*bIt)->createProduct(view->typeId(), view->viewContext());

            if ((*bIt)->havePerViewProduct((FWViewType::EType)view->typeId()))
            {
               // view owned
               (*bIt)->build();
               if (rpzView)
               {
                  rpzView->importElements(product, (*bIt)->item()->layer(), rpzView->ownedProducts());
               }
               else
               {
                  view->ownedProducts()->AddElement(product);
               }
            }
            else
            {
               // shared
               if (rpzView)
               {
                  rpzView->importElements(product, (*bIt)->item()->layer(), rpzView->eventScene());
               }
               else
               {
                  view->eventScene()->AddElement(product);
               }
                 
            }
         }
      }
   }

   view->beingDestroyed_.connect(boost::bind(&FWEveViewManager::beingDestroyed,this,_1));

   view->setupEnergyScale(); // notify PB for energy scale

   gEve->EnableRedraw();
   view->viewerGL()->UpdateScene();
   gEve->Redraw3D();   

   return view.get();
}

void
FWEveViewManager::beingDestroyed(const FWViewBase* vb)
{
   FWEveView* view = (FWEveView*) vb;
   int typeId = view->typeId();
  
   int viewerBit = 1 << typeId;
   int nviews = m_views[typeId].size(); 
   for ( std::map<int, BuilderVec>::iterator i = m_builders.begin(); i!= m_builders.end(); ++i)
   {
      int builderBit = i->first;
      if (viewerBit == (builderBit & viewerBit)) // check only in case if connected
      {
         BuilderVec& bv =  i->second;

         // remove view-owned product
         if (viewerBit == (builderBit & viewerBit))
         {
            for(BuilderVec_it bIt = bv.begin(); bIt != bv.end(); ++bIt)
               (*bIt)->removePerViewProduct(view->typeId(), view->viewContext());
         }

         // and setup proxy builders have-a-window flag
         if (nviews == 1)
         {
            if (!haveViewForBit(builderBit))
            {
               if (viewerBit == (builderBit & viewerBit))
               {
                  for(BuilderVec_it bIt = bv.begin(); bIt != bv.end(); ++bIt)
                     (*bIt)->setHaveWindow(false);
               }
            }
         }
      }
   }
  

   for(EveViewVec_it i= m_views[typeId].begin(); i != m_views[typeId].end(); ++i) {
      if(i->get() == vb) {
         m_views[typeId].erase(i);
         break;
      }
   }
}

//______________________________________________________________________________

void
FWEveViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}

void
FWEveViewManager::modelChangesDone()
{
   gEve->EnableRedraw();
}

/** Callback of event item changed_ signal.*/
void
FWEveViewManager::modelChanges(const FWModelIds& iIds)
{
   FWModelId id = *(iIds.begin());
   const FWEventItem* item = id.item();

   // in standard case new elements can be build in case of change of visibility
   // and in non-standard case (e.g. calo towers) PB's modelChages handles all changes
   bool itemHaveWindow = false;
   for (std::map<int, BuilderVec>::iterator i = m_builders.begin(); 
        i != m_builders.end(); ++i)
   {
      for (size_t bi = 0, be = i->second.size(); bi != be; ++bi)
      {
         FWProxyBuilderBase *builder = i->second[bi].get();
         if (builder->getHaveWindow() && builder->item() == item)
         {
            builder->modelChanges(iIds);
            itemHaveWindow = true;
         }
      }
   }

   if (!itemHaveWindow)
      return;

   EveSelectionSentry();

   std::map<const FWEventItem*, FWInteractionList*>::iterator it = m_interactionLists.find(item);
   if (it != m_interactionLists.end())
   {
      if (!it->second->empty())
         it->second->modelChanges(iIds);
   }
}

/** Callback of itemChanged_ signal.
    Iterate over all the builders for all the views and call itemChanged
    for any of the builders.
    If any of the builder also has at least one view, also update the interaction list.
*/
void
FWEveViewManager::itemChanged(const FWEventItem* item)
{
   if (!item)
      return;

   bool itemHaveWindow = false;

   for (std::map<int, BuilderVec>::iterator i = m_builders.begin(); 
        i != m_builders.end(); ++i)
   {
      for(size_t bi = 0, be = i->second.size(); bi != be; ++bi)
      {
         FWProxyBuilderBase *builder = i->second[bi].get();
         
         if (builder->item() != item)
            continue;

         builder->itemChanged(item);
         itemHaveWindow |= builder->getHaveWindow();
      }
   }
   
   if (!itemHaveWindow)
      return;

   std::map<const FWEventItem*, FWInteractionList*>::iterator it = m_interactionLists.find(item);
   if (it != m_interactionLists.end())
   {
      if (!it->second->empty())
         it->second->itemChanged();
   }
}

/** Remove an item from the given view.
  */
void
FWEveViewManager::removeItem(const FWEventItem* item)
{
   EveSelectionSentry();

   std::map<const FWEventItem*, FWInteractionList*>::iterator it =  m_interactionLists.find(item);
   if (it != m_interactionLists.end())
   {
      delete it->second;
      m_interactionLists.erase(it);
   }
  
   for (std::map<int, BuilderVec>::iterator i = m_builders.begin();
        i != m_builders.end(); ++i)
   {
      BuilderVec_it bIt = i->second.begin();
      while( bIt != i->second.end() )
      {
         if ((*bIt)->item() == item)
         { 
            // TODO caching of proxy builders
            (*bIt)->itemBeingDestroyed(item);
            bIt = i->second.erase(bIt);
         }
         else
         {
            ++bIt;
         }
      }
   }
}

void
FWEveViewManager::setContext(const fireworks::Context* x)
{
   FWViewManagerBase::setContext(x);
   x->commonPrefs()->getEnergyScale()->parameterChanged_.connect(boost::bind(&FWEveViewManager::globalEnergyScaleChanged,this));

}

void
FWEveViewManager::globalEnergyScaleChanged()
{
   for (int t = 0 ; t < FWViewType::kTypeSize; ++t)
   {
      for(EveViewVec_it i = m_views[t].begin(); i != m_views[t].end(); ++i) 
      {
         if ((*i)->isEnergyScaleGlobal())
         {
            (*i)->setupEnergyScale();
         }
      }

   }
}

void
FWEveViewManager::colorsChanged()
{
   for (int t = 0 ; t < FWViewType::kTypeSize; ++t)
   {
      for(EveViewVec_it i = m_views[t].begin(); i != m_views[t].end(); ++i) 
         (*i)->setBackgroundColor(colorManager().background());
   }
}

//______________________________________________________________________________
void
FWEveViewManager::eventBegin()
{
   // Prevent registration of redraw timer, full redraw is done in
   // FWEveViewManager::eventEnd().
   gEve->fTimerActive = kTRUE;
   gEve->DisableRedraw();

   context().resetMaxEtAndEnergy();

   for (int t = 0 ; t < FWViewType::kTypeSize; ++t)
   {
      for(EveViewVec_it i = m_views[t].begin(); i != m_views[t].end(); ++i)   
         (*i)->eventBegin();
   }
}

void
FWEveViewManager::eventEnd()
{
   for (int t = 0 ; t < FWViewType::kTypeSize; ++t)
   {
      for(EveViewVec_it i = m_views[t].begin(); i != m_views[t].end(); ++i)   
         (*i)->eventEnd();
   }

   // What follows is a copy of TEveManager::DoRedraw3D() with the difference that
   // we have full control over execution of GL view rendering. In particular:
   // - optionally delay buffer swapping so they can all be swapped together;
   // - we could render into FBO once and then use this to be put on screen
   //   and saved into an image file.

   {
      TEveElement::List_t scenes;
      Long64_t   key, value;
      TExMapIter stamped_elements(gEve->fStampedElements);
      while (stamped_elements.Next(key, value))
      {
         TEveElement *el = reinterpret_cast<TEveElement*>(key);
         if (el->GetChangeBits() & TEveElement::kCBVisibility)
         {
            el->CollectSceneParents(scenes);
         }
      }
      gEve->ScenesChanged(scenes);
   }

   // Process changes in scenes.
   gEve->GetScenes()->ProcessSceneChanges(gEve->fDropLogicals, gEve->fStampedElements);

   // To synchronize buffer swapping set swap_on_render to false.
   // Note that this costs 25-40% extra time with 4 views, depending on V-sync settings.
   // Tested with NVIDIA 343.22.
   const bool swap_on_render = !s_syncAllViews;

   // Loop over viewers, swap buffers if swap_on_render is true.
   for (int t = 0 ; t < FWViewType::kTypeSize; ++t)
   {
      for(EveViewVec_it i = m_views[t].begin(); i != m_views[t].end(); ++i)   
         (*i)->fwViewerGL()->DrawHiLod(swap_on_render);
   }

   // Swap buffers if they were not swapped before.
   if ( ! swap_on_render)
   {
      for (int t = 0 ; t < FWViewType::kTypeSize; ++t)
      {
         for(EveViewVec_it i = m_views[t].begin(); i != m_views[t].end(); ++i)   
            (*i)->fwViewerGL()->JustSwap();
      }
   }

   gEve->GetViewers()->RepaintChangedViewers(gEve->fResetCameras, gEve->fDropLogicals);

   {
      Long64_t   key, value;
      TExMapIter stamped_elements(gEve->fStampedElements);
      while (stamped_elements.Next(key, value))
      {
         TEveElement *el = reinterpret_cast<TEveElement*>(key);
         if (gEve->GetEditor()->GetModel() == el->GetEditorObject("FWEveViewManager::eventEnd"))
            gEve->EditElement(el);
         TEveGedEditor::ElementChanged(el);

         el->ClearStamps();
      }
   }
   gEve->fStampedElements->Delete();

   gEve->GetListTree()->ClearViewPort(); // Fix this when several list-trees can be added.

   gEve->fResetCameras = kFALSE;
   gEve->fDropLogicals = kFALSE;

   gEve->EnableRedraw();
   gEve->fTimerActive = kFALSE;
}

//______________________________________________________________________________

/** Helper function to extract the FWFromEveSelectorBase * from an TEveElement.
  */
FWFromEveSelectorBase *getSelector(TEveElement *iElement)
{
   if (!iElement)
      return 0;

   //std::cout <<"  non null"<<std::endl;
   void* userData = iElement->GetUserData();
   //std::cout <<"  user data "<<userData<<std::endl;
   if (!userData)
      return 0;

   //std::cout <<"    have userData"<<std::endl;
   //std::cout <<"      calo"<<std::endl;
   EveSelectionSentry();
   return reinterpret_cast<FWFromEveSelectorBase*> (userData);   
}

void
FWEveViewManager::selectionAdded(TEveElement* iElement)
{
   FWFromEveSelectorBase* selector = getSelector(iElement);
   if (selector)
      selector->doSelect();
}

void
FWEveViewManager::selectionRemoved(TEveElement* iElement)
{
   FWFromEveSelectorBase* selector = getSelector(iElement);
   if (selector)
      selector->doUnselect();
}

void
FWEveViewManager::selectionCleared()
{
   context().selectionManager()->clearSelection();
}


//
// const member functions
//

FWTypeToRepresentations
FWEveViewManager::supportedTypesAndRepresentations() const
{
   // needed for add collection GUI
   FWTypeToRepresentations returnValue;
   const static std::string kFullFrameWorkPBExtension = "FullFramework";
   for(TypeToBuilder::const_iterator it = m_typeToBuilder.begin(), itEnd = m_typeToBuilder.end();
       it != itEnd;
       ++it) 
   {
      std::vector<BuilderInfo> blist = it->second;
      for (size_t bii = 0, bie = blist.size(); bii != bie; ++bii)
      {
         BuilderInfo &info = blist[bii];
         
         if (context().getHidePFBuilders()) {
            const static std::string pfExt = "PF ";
            if (std::string::npos != info.m_name.find(pfExt))
               continue;
               }

         unsigned int bitPackedViews = info.m_viewBit;
         bool representsSubPart = (info.m_name.substr(info.m_name.find_first_of('@')-1, 1)=="!");
         size_t extp = info.m_name.rfind(kFullFrameWorkPBExtension);
         bool FFOnly = (extp != std::string::npos);

         std::string name;
         bool isSimple;
         info.classType(name, isSimple);
         if(isSimple) 
         {
            returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>(new FWSimpleRepresentationChecker(name, it->first,bitPackedViews,representsSubPart, FFOnly)) );
         }
         else
         {
            returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>(new FWEDProductRepresentationChecker(name, it->first,bitPackedViews,representsSubPart, FFOnly)) );
         }
      }
   }
   return returnValue;
}


/** Checks whether any of the views */
bool
FWEveViewManager::haveViewForBit(int bit) const
{
   for (int t = 0; t < FWViewType::kTypeSize; ++t)
   {
      if ((bit & (1 << t)) && m_views[t].size())
         return true;
   }
   // printf("have %d view for bit %d \n", haveView, bit);
   return false;
}


void
FWEveViewManager::BuilderInfo::classType(std::string& typeName, bool& simple) const
{
   const std::string kSimple("simple#");
   simple = (m_name.substr(0,kSimple.size()) == kSimple);
   if (simple)
   {
      typeName = m_name.substr(kSimple.size(), m_name.find_first_of('@')-kSimple.size()-1);
   }
   else
   {
      typeName = m_name.substr(0, m_name.find_first_of('@')-1);
   }
}

/*
AMT: temporary workaround for using TEveCaloDataHist instead of
TEveCaloDataVec. 

 */

#include "TH2F.h"
#include "TAxis.h"
#include "TEveCaloData.h"

void
FWEveViewManager::highlightAdded(TEveElement* iElement)
{

   bool blocked = gEve->GetHighlight()->BlockSignals(true);


   if (iElement == context().getCaloData())
   {
      TEveCaloData::vCellId_t& hlist =  context().getCaloData()->GetCellsHighlighted();
      std::set<TEveCaloData::CellId_t> hset;

      int etaBin, phiBin, w, newPhiBin, tower;
      TH2F* hist =  context().getCaloData()->GetHist(0);
      TAxis* etaAxis = hist->GetXaxis();
      int nBinsX = etaAxis->GetNbins() + 2;

      for (TEveCaloData::vCellId_i i = hlist.begin(); i != hlist.end(); ++i)
      {
         hist->GetBinXYZ((*i).fTower, etaBin, phiBin, w);
         if (TMath::Abs(etaAxis->GetBinCenter(etaBin)) > 4.71475)
         {
            newPhiBin = ((phiBin + 1) / 4) * 4 - 1;
            if (newPhiBin <= 0) newPhiBin = 71;
               
            tower = etaBin + newPhiBin*nBinsX;
            hset.insert(TEveCaloData::CellId_t( tower, (*i).fSlice, (*i).fFraction));
            tower += nBinsX;
            hset.insert(TEveCaloData::CellId_t(tower, (*i).fSlice, (*i).fFraction));
            tower += nBinsX;

            if (newPhiBin == 71)
               tower = etaBin + 1*nBinsX;

            hset.insert(TEveCaloData::CellId_t(tower, (*i).fSlice, (*i).fFraction));
            tower += nBinsX;
            hset.insert(TEveCaloData::CellId_t(tower, (*i).fSlice, (*i).fFraction));
         }
         else if (TMath::Abs(etaAxis->GetBinCenter(etaBin)) > 1.747650)
         {  
            newPhiBin =  ((phiBin + 1)/2)*2 - 1;
            tower = etaBin + newPhiBin*nBinsX;
            hset.insert(TEveCaloData::CellId_t( tower, (*i).fSlice, (*i).fFraction));
            tower += nBinsX;
            hset.insert(TEveCaloData::CellId_t(tower, (*i).fSlice, (*i).fFraction));
         }
         else
         {
            hset.insert(*i);
         }
      }
 
      // edit calo data list
      hlist.clear();
      for(std::set<TEveCaloData::CellId_t>::iterator it = hset.begin(); it != hset.end(); ++it)
      {
         hlist.push_back(*it);
      }    
      context().getCaloData()->CellSelectionChanged();

   }

   gEve->GetHighlight()->BlockSignals(blocked);
}
