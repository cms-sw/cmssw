#ifndef Fireworks_FWInterface_FWPSetTableManager_h
#define Fireworks_FWInterface_FWPSetTableManager_h
// -*- C++ -*-
//
// Package:     FWInterface
// Class  :     FWPSetTableManager
// 
/**\class FWPSetTableManager FWPSetTableManager.h Fireworks/FWInterface/interface/FWPSetTableManager.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon Feb 28 17:06:50 CET 2011
// $Id: FWPSetTableManager.h,v 1.11 2012/09/08 06:13:57 amraktad Exp $
//

// system include files


#include <sigc++/sigc++.h>
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Fireworks/TableWidget/interface/FWTextTreeCellRenderer.h"

class FWPSetCellEditor;
namespace edm 
{
   class ScheduleInfo;
}



class FWPSetTableManager : public FWTableManagerBase 
{
   friend class FWPathsPopup;

public: 
   /** Custom structure for holding the table contents */
   struct PSetData
   {
      PSetData() :level(-1),
         tracked(false),

         type(-1),
   
         parent(-1),

         module(-1),
         path(-1),

         expandedUser(false),
         expandedFilter(false),

         visible(false),

         matches(false),
         childMatches(false),

       editable(false) , pset(0){}

      std::string label;
      std::string value;

      int         level;
      bool        tracked;
      char        type;
      size_t      parent;
   
      size_t      module;
      size_t      path;

      bool        expandedUser;
      bool        expandedFilter;

      bool        visible;
      bool        matches;
      bool        childMatches;
      bool        editable;

      edm::ParameterSet *pset;
   };

   FWPSetTableManager();
   virtual ~FWPSetTableManager();


   virtual int unsortedRowNumber(int unsorted) const ;
   virtual int numberOfRows() const;
   virtual int numberOfColumns() const;
   virtual std::vector<std::string> getTitles() const;
   virtual const std::string title() const;
   virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const;

   int selectedRow() const;
   int selectedColumn() const;
   virtual bool rowIsSelected(int row) const;

   virtual void implSort(int, bool);
   virtual bool cellDataIsSortable() const { return false ; }

   virtual void updateFilter(const char *filter);

   virtual std::vector<unsigned int> maxWidthForColumns() const;

   std::vector<PSetData> &data()  { return m_entries; }
   std::vector<int> &rowToIndex() { return m_row_to_index; }
  
   //______________________________________________________________________________

protected:
   struct ModuleInfo
   {
      /** The path this info is associated to, as ordered in
          availablePath(); 
      */
      size_t            pathIndex;
      size_t            path;
      size_t            entry;
      bool              passed;
      /** Whether or not the pset was modified since last time the 
          looper reloaded. 
      */
      bool              dirty;

     edm::ParameterSet *orig_pset;
     edm::ParameterSet *current_pset;
   };

   /** Datum for updating the path status information */
   struct PathUpdate
   {
      std::string pathName;
      bool passed;
      size_t  choiceMaker;
   };


   void setExpanded(int row);
   void updateSchedule(const edm::ScheduleInfo *info);
   void update(std::vector<PathUpdate> &pathUpdates);

   bool applyEditor();
   void cancelEditor();

   std::vector<ModuleInfo> &modules() { return m_modules; }
   std::vector<PSetData>   &entries() { return m_entries; }

   void  setSelection (int row, int column, int mask);
   //______________________________________________________________________________

private: 

   /** Model for additional path information */
   struct PathInfo
   {
      std::string pathName;
      size_t      entryId;
      int         modulePassed;
      size_t      moduleStart;
      size_t      moduleEnd;
      bool        passed;
   };

   FWPSetTableManager(const FWPSetTableManager&); // stop default
   const FWPSetTableManager& operator=(const FWPSetTableManager&); // stop default

   void recalculateVisibility();

   template <class T> void createScalarString(PSetData &data, T v);
   template <typename T> void createVectorString(FWPSetTableManager::PSetData &data, const T &v, bool quotes);

   void setCellValueEditor(FWPSetCellEditor *editor);

   void handleEntry(const edm::Entry &entry,const std::string &key);
   void handlePSetEntry(edm::ParameterSetEntry& entry, const std::string& key);
   void handleVPSetEntry(edm::VParameterSetEntry& entry, const std::string& key);
   void handlePSet(edm::ParameterSet *psp);

   std::vector<PSetData>           m_entries;
   /** Index in m_entries where to find paths */
   std::vector<PathInfo>           m_paths;
   std::vector<ModuleInfo>         m_modules;
   std::map<std::string, size_t>   m_pathIndex;
   std::vector<size_t>             m_parentStack;
   std::vector<int>                m_row_to_index;
   int                             m_selectedRow;
   int                             m_selectedColumn;
   std::string                     m_filter;
   FWPSetCellEditor               *m_editor;
   std::vector<std::string>        m_availablePaths;

   mutable FWTextTreeCellRenderer m_renderer;  
};


#endif
