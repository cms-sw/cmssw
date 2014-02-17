// -*- C++ -*-
#ifndef Fireworks_Core_FWTableViewManager_h
#define Fireworks_Core_FWTableViewManager_h
//
// Package:     Core
// Class  :     FWTableViewManager
//
/**\class FWTableViewManager FWTableViewManager.h Fireworks/Core/interface/FWTableViewManager.h

   Description: Base class for a Manger for a specific type of View

   Usage:
   <usage>

*/
//
// Original Author:
//         Created:  Sat Jan  5 10:29:00 EST 2008
// $Id: FWTableViewManager.h,v 1.10 2012/08/03 18:20:27 wmtan Exp $
//

// system include files
#include <string>
#include <vector>
#include <set>
#include <map>
#include "FWCore/Utilities/interface/TypeWithDict.h"

// user include files

#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWTableView.h"
#include "Fireworks/Core/interface/FWConfigurable.h"

class FWViewBase;
class FWGUIManager;
class TEveWindowSlot;

class FWTableViewManager : public FWViewManagerBase, public FWConfigurable 
{
   friend class FWTableView;
   friend class FWTableViewTableManager;
public:
   struct TableEntry {
      enum { INT = 0, INT_HEX = -1, BOOL = -2 };
      std::string    expression;
      std::string    name;
      int            precision;
   };

   /** Container for the event items which have a table. */
   typedef std::vector<const FWEventItem *>                Items;
   /** Container for the description of the columns of a given table. */
   typedef std::vector<TableEntry> TableEntries;
   /** Type for the collection specific (i.e. those that do not use
       default) table definition. */
   typedef std::map<std::string, TableEntries> TableSpecs;

   FWTableViewManager(FWGUIManager*);
   virtual ~FWTableViewManager();

   // ---------- const member functions ---------------------
   virtual FWTypeToRepresentations supportedTypesAndRepresentations() const;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void            newItem(const FWEventItem*);
   void                    destroyItem(const FWEventItem *item);
   void                    removeAllItems(void);
   FWViewBase *            buildView(TEveWindowSlot *iParent, const std::string& type);
   const Items &           items() const { return m_items; }
   TableSpecs::iterator    tableFormats(const edm::TypeWithDict &key);
   TableSpecs::iterator    tableFormats(const TClass &key);
   void                    addTo(FWConfiguration&) const;
   void                    addToImpl(FWConfiguration&) const;
   void                    setFrom(const FWConfiguration&);

   void                    notifyViews();

   static const std::string kConfigTypeNames;
   static const std::string kConfigColumns;

protected:
   FWTableViewManager();

   /** Called when models have changed and so the display must be updated. */
   virtual void modelChangesComing();
   virtual void modelChangesDone();
   virtual void colorsChanged();
   void dataChanged ();

   typedef std::vector<boost::shared_ptr<FWTableView> >    Views;

   Views       m_views;
   Items       m_items;
   TableSpecs  m_tableFormats;
private:
   TableSpecs::iterator tableFormatsImpl(const edm::TypeWithDict &key);
   FWTableViewManager(const FWTableViewManager&);    // stop default
   const FWTableViewManager& operator=(const FWTableViewManager&);    // stop default

   void beingDestroyed(const FWViewBase*);
   
   class TableHandle
   {
   public: 
      TableHandle &column(const char *formula, int precision, const char *name);
      TableHandle &column(const char *label, int precision)
         {
            return column(label, precision, label);
         }

      TableHandle(const char *name, TableSpecs &specs)
         :m_name(name), m_specs(specs) 
         {
            m_specs[name].clear();
         }
   private:
      std::string  m_name;
      TableSpecs  &m_specs;
   };

   TableHandle table(const char *collection);
};

#endif
