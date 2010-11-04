#ifndef Fireworks_Core_FWL1TriggerTableViewManager_h
#define Fireworks_Core_FWL1TriggerTableViewManager_h

#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWConfigurable.h"

#include <string>
#include <vector>

class FWGUIManager;
class FWL1TriggerTableView;
class FWTypeToRepresentations;
class FWViewBase;
class TEveWindowSlot;

class FWL1TriggerTableViewManager : public FWViewManagerBase, public FWConfigurable
{
   friend class FWL1TriggerTableView;
   friend class FWL1TriggerTableViewTableManager;

public:
   FWL1TriggerTableViewManager(FWGUIManager *manager);
   virtual ~FWL1TriggerTableViewManager(void);

   virtual FWTypeToRepresentations supportedTypesAndRepresentations(void) const;

   virtual void 	newItem(const FWEventItem *item);
   void 		destroyItem(const FWEventItem *item);
   FWViewBase *		buildView(TEveWindowSlot *iParent, const std::string& type);
   const std::vector<const FWEventItem *> &items(void) const {
      return m_items;
   }
   void 		addTo(FWConfiguration&) const;
   void 		addToImpl(FWConfiguration&) const;
   void 		setFrom(const FWConfiguration&);

   static const std::string kConfigTypeNames;
   static const std::string kConfigColumns;

protected:
   FWL1TriggerTableViewManager(void);

   /** called when models have changed and so the display must be updated */
   virtual void 	modelChangesComing(void);
   virtual void 	modelChangesDone(void);
   virtual void 	colorsChanged(void);
   void 		dataChanged(void);

   std::vector<boost::shared_ptr<FWL1TriggerTableView> > m_views;
   std::vector<const FWEventItem *> m_items;

private:
   FWL1TriggerTableViewManager(const FWL1TriggerTableViewManager&);      // stop default
   const FWL1TriggerTableViewManager& operator=(const FWL1TriggerTableViewManager&);      // stop default

   void beingDestroyed(const FWViewBase*);
};

#endif
