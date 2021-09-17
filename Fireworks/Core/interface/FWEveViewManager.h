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
//

// system include files
#include <vector>
#include <map>
#include <set>
#include <memory>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWViewType.h"

// forward declarations
class TEveCompund;
class TEveScene;
class TEveElement;
class TEveWindowSlot;
class FWViewBase;
class FWEveView;
class FWProxyBuilderBase;
class FWGUIManager;
class FWInteractionList;

typedef std::set<FWModelId> FWModelIds;

class FWEveViewManager : public FWViewManagerBase {
public:
  struct BuilderInfo {
    std::string m_name;
    int m_viewBit;

    void classType(std::string&, bool&) const;

    BuilderInfo(std::string name, int viewBit) : m_name(name), m_viewBit(viewBit) {}
  };

  FWEveViewManager(FWGUIManager*);
  ~FWEveViewManager() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void newItem(const FWEventItem*) override;
  virtual void removeItem(const FWEventItem*);
  void eventBegin() override;
  void eventEnd() override;
  void setContext(const fireworks::Context*) override;

  void highlightAdded(TEveElement*);
  void selectionAdded(TEveElement*);
  void selectionRemoved(TEveElement*);
  void selectionCleared();

  FWTypeToRepresentations supportedTypesAndRepresentations() const override;

  static void syncAllViews() { s_syncAllViews = true; }

protected:
  void modelChangesComing() override;
  void modelChangesDone() override;
  void colorsChanged() override;

public:
  FWEveViewManager(const FWEveViewManager&) = delete;                   // stop default
  const FWEveViewManager& operator=(const FWEveViewManager&) = delete;  // stop default

private:
  FWViewBase* buildView(TEveWindowSlot* iParent, const std::string& type);
  FWEveView* finishViewCreate(std::shared_ptr<FWEveView>);

  void beingDestroyed(const FWViewBase*);
  void modelChanges(const FWModelIds& iIds);
  void itemChanged(const FWEventItem*);
  bool haveViewForBit(int) const;
  void globalEnergyScaleChanged();
  void eventCenterChanged();

  // ---------- member data --------------------------------

  typedef std::map<std::string, std::vector<BuilderInfo> > TypeToBuilder;
  typedef std::vector<std::shared_ptr<FWProxyBuilderBase> > BuilderVec;
  typedef BuilderVec::iterator BuilderVec_it;
  typedef std::vector<std::shared_ptr<FWEveView> >::iterator EveViewVec_it;

  TypeToBuilder m_typeToBuilder;

  static bool s_syncAllViews;

  std::map<int, BuilderVec> m_builders;  // key is viewer bit

  std::vector<std::vector<std::shared_ptr<FWEveView> > > m_views;

  std::map<const FWEventItem*, FWInteractionList*> m_interactionLists;
};

#endif
