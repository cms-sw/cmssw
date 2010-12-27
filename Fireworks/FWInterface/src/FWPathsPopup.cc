#include "Fireworks/FWInterface/src/FWPathsPopup.h"
#include "Fireworks/FWInterface/interface/FWFFLooper.h"
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/src/FWTabularWidget.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"
#include "Fireworks/Core/interface/FWGUIManager.h"

#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/Parse.h"

#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "TGLabel.h"
#include "TGTextEdit.h"
#include "TGText.h"
#include "TSystem.h"
#include "TGFont.h"
#include "TGTextEntry.h"
#include "KeySymbols.h"
#include "TGPicture.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <map>

#include "GuiTypes.h"

// FIXME: copied from Entry.cc should find a way to use the original
//        table.
struct TypeTrans {
  TypeTrans();

  typedef std::vector<std::string> CodeMap;
  CodeMap table_;
  std::map<std::string, char> type2Code_;
};

TypeTrans::TypeTrans():table_(255) {
  table_['b'] = "vBool";
  table_['B'] = "bool";
  table_['i'] = "vint32";
  table_['I'] = "int32";
  table_['u'] = "vuint32";
  table_['U'] = "uint32";
  table_['l'] = "vint64";
  table_['L'] = "int64";
  table_['x'] = "vuint64";
  table_['X'] = "uint64";
  table_['s'] = "vstring";
  table_['S'] = "string";
  table_['d'] = "vdouble";
  table_['D'] = "double";
  table_['p'] = "vPSet";
  table_['P'] = "PSet";
  table_['T'] = "path";
  table_['F'] = "FileInPath";
  table_['t'] = "InputTag";
  table_['v'] = "VInputTag";
  table_['g'] = "ESInputTag";
  table_['G'] = "VESInputTag";
  table_['e'] = "VEventID";
  table_['E'] = "EventID";
  table_['m'] = "VLuminosityBlockID";
  table_['M'] = "LuminosityBlockID";
  table_['a'] = "VLuminosityBlockRange";
  table_['A'] = "LuminosityBlockRange";
  table_['r'] = "VEventRange";
  table_['R'] = "EventRange";

  for(CodeMap::const_iterator itCode = table_.begin(), itCodeEnd = table_.end();
       itCode != itCodeEnd;
       ++itCode) {
     type2Code_[*itCode] = (itCode - table_.begin());
  }
}

static TypeTrans const sTypeTranslations;

//Where to find the icons
static const TString& coreIcondir() {
   static TString path = Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE"));
   if ( gSystem->AccessPathName(path.Data()) ){ // cannot find directory
      assert(gSystem->Getenv("CMSSW_RELEASE_BASE"));
      path = Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_RELEASE_BASE"));
   }

   return path;
}

static
const TGPicture* closedImage()
{
   static const TGPicture* s_picture=gClient->GetPicture(coreIcondir()+"arrow-black-right-whitebg.png");
   return s_picture;
}

static
const TGPicture* openedImage()
{
   static const TGPicture* s_picture=gClient->GetPicture(coreIcondir()+"arrow-black-down-whitebg.png");
   return s_picture;
}


class FWTextTreeCellRenderer : public FWTextTableCellRenderer
{
public:
   FWTextTreeCellRenderer(const TGGC* iContext = &(getDefaultGC()),
                          const TGGC* iHighlightContext = &(getHighlightGC()),
                          Justify iJustify = kJustifyLeft)
      : FWTextTableCellRenderer(iContext, iHighlightContext, iJustify),
        m_indentation(0),
        m_editor(0),
        m_showEditor(false),
        m_isParent(false),
        m_isOpen(false)
      {}

   virtual void setIndentation(int indentation = 0) { m_indentation = indentation; }
   virtual void setCellEditor(TGTextEntry *editor) { m_editor = editor; }
   virtual void showEditor(bool value) { m_showEditor = value; }
   void setIsParent(bool value) {m_isParent = value; }
   void setIsOpen(bool value) {m_isOpen = value; }
   virtual UInt_t width() const { return FWTextTableCellRenderer::width() + 15 + m_indentation + 
     (m_isParent ?  closedImage()->GetWidth() + 2: 0  ); }
   virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
      {
         if (m_showEditor && m_editor)
         {
            m_editor->MoveResize(iX-3, iY-3, iWidth + 6 , iHeight + 6);
            m_editor->MapWindow();
            m_editor->SetText(data().c_str());
            return;
         }

         if (selected())
         {
            GContext_t c = highlightContext()->GetGC();
            gVirtualX->FillRectangle(iID, c, iX, iY, iWidth, iHeight);
            
            gVirtualX->DrawLine(iID,graphicsContext()->GetGC(),iX-1,iY-1,iX-1,iY+iHeight);
            gVirtualX->DrawLine(iID,graphicsContext()->GetGC(),iX+iWidth,iY-1,iX+iWidth,iY+iHeight);
            gVirtualX->DrawLine(iID,graphicsContext()->GetGC(),iX-1,iY-1,iX+iWidth,iY-1);
            gVirtualX->DrawLine(iID,graphicsContext()->GetGC(),iX-1,iY+iHeight,iX+iWidth,iY+iHeight);
         }
         int xOffset = 0;
         if(m_isParent) {
            if(m_isOpen) {
              openedImage()->Draw(iID,graphicsContext()->GetGC(),m_indentation+iX,iY);
              xOffset += openedImage()->GetWidth() + 2;
            } else {
              closedImage()->Draw(iID,graphicsContext()->GetGC(),m_indentation+iX,iY);
              xOffset += closedImage()->GetWidth() + 2;
            }
         }
         FontMetrics_t metrics;
         font()->GetFontMetrics(&metrics);
         gVirtualX->DrawString(iID, graphicsContext()->GetGC(),
                               iX+m_indentation+xOffset, iY+metrics.fAscent, 
                               data().c_str(),data().size());
      }
private:
   int            m_indentation;
   TGTextEntry    *m_editor;
   bool           m_showEditor;
   bool           m_isParent;
   bool           m_isOpen;
};

/** Custom structure for holding the table contents */
struct PSetData
{
   PSetData()
   : type(-1)
   {}

   std::string label;
   std::string value;
   int         level;
   bool        tracked;
   char        type;
   size_t      parent;
   
   size_t      module;
   size_t      path;
   // Whether or not it matches the filter.
   bool        matches;
   // Whether or not it is expanded.
   bool        expanded;
   // Whether or not it is visibile.  Being visible is given by either matching
   // a non-null filter, or  or the parent being visibible and expanded.
   bool        visible;
   // Whether or not any of the children matches the filter.
   bool        childMatches;
   // Whether or not the contents of the GUI can be edited.
   bool        editable;
   // Copy of the parameter set associated with this item.
   // We need to keep a copy, because updating a parameter
   // in a parameter set means actually creating a new one,
   // because the checksum changes.
   edm::ParameterSet pset;
};

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
};

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

/** Datum for updating the path status information */
struct PathUpdate
{
   std::string pathName;
   bool passed;
   size_t  choiceMaker;
};

/** Attempt to create a table based editor for the PSET. */
class FWPSetTableManager : public FWTableManagerBase 
{
public:
 
  /*
    From TGFont.h

  enum EFontWeight {
   kFontWeightNormal = 0,
   kFontWeightMedium = 0,
   kFontWeightBold = 1,
   kFontWeightLight = 2,
   kFontWeightDemibold = 3,
   kFontWeightBlack = 4,
   kFontWeightUnknown = -1
  };
  */


   // All this GC stuff is getting very clunky
   const TGGC&
   boldGC()
   {
      static TGGC s_boldGC(*gClient->GetResourcePool()->GetFrameGC());
 
      TGFontPool *pool = gClient->GetFontPool();
      TGFont *font = pool->FindFontByHandle(s_boldGC.GetFont());
      FontAttributes_t attributes = font->GetFontAttributes();

      // This doesn't seem to work:
      //attributes.fWeight = 1; 
      //TGFont *newFont = pool->GetFont(attributes.fFamily, 9,
      //                                attributes.fWeight, attributes.fSlant);

      // But this does:
      TGFont* newFont = pool->GetFont("-*-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
      
      if ( ! newFont )
        return s_boldGC;

      //std::cout<<"boldGC: "<< std::endl;
      //newFont->Print();

      s_boldGC.SetFont(newFont->GetFontHandle());

      return s_boldGC;
   }

   const TGGC&
   italicGC()
   {
      static TGGC s_italicGC(*gClient->GetResourcePool()->GetFrameGC());
 
      TGFontPool *pool = gClient->GetFontPool();
      TGFont *font = pool->FindFontByHandle(s_italicGC.GetFont());
      FontAttributes_t attributes = font->GetFontAttributes();

      attributes.fSlant = 1;
      TGFont *newFont = pool->GetFont(attributes.fFamily, 9,
                                      attributes.fWeight, attributes.fSlant);

      //std::cout<<"italicGC: "<< std::endl;
      //newFont->Print();

      s_italicGC.SetFont(newFont->GetFontHandle());

      return s_italicGC;
   }

   const TGGC &
   pathBackgroundGC()
   {
      static TGGC s_pathBackgroundGC(*gClient->GetResourcePool()->GetFrameGC());
      s_pathBackgroundGC.SetBackground(gVirtualX->GetPixel(kGray));
      return s_pathBackgroundGC;
   }


   // Is there a simpler way to handle colors?   
   const TGGC &
   boldRedGC()
   {
      static TGGC s_boldRedGC(boldGC());
      s_boldRedGC.SetForeground(gVirtualX->GetPixel(kRed-5));
      return s_boldRedGC;
   }

   const TGGC &
   italicRedGC()
   {
      static TGGC s_italicRedGC(italicGC());
      s_italicRedGC.SetForeground(gVirtualX->GetPixel(kRed-5));
      return s_italicRedGC;
   }

   const TGGC&
   italicGray()
   {
      static TGGC s_italicGrayGC(italicGC());
      s_italicGrayGC.SetForeground(gVirtualX->GetPixel(kGray+1));
      return s_italicGrayGC;
   }
  
   const TGGC&
   redGC()
   {
      static TGGC s_redGC(*gClient->GetResourcePool()->GetFrameGC());
      s_redGC.SetForeground(gVirtualX->GetPixel(kRed-5));
      return s_redGC;
   }
 
   const TGGC&
   boldGreenGC()
   {
      static TGGC s_boldGreenGC(boldGC());
      s_boldGreenGC.SetForeground(gVirtualX->GetPixel(kGreen-5));
      return s_boldGreenGC;
   }
 
   const TGGC &
   italicGreenGC()
   {
      static TGGC s_italicGreenGC(italicGC());
      s_italicGreenGC.SetForeground(gVirtualX->GetPixel(kGreen-5));
      return s_italicGreenGC;
   }

   const TGGC&
   greenGC()
   {
      static TGGC s_greenGC(*gClient->GetResourcePool()->GetFrameGC());
      s_greenGC.SetForeground(gVirtualX->GetPixel(kGreen-5));
      return s_greenGC;
   }
  

   FWPSetTableManager()
      : m_selectedRow(-1)
   {  
      m_italicRenderer.setGraphicsContext(&italicGC());
      m_boldRenderer.setGraphicsContext(&boldGC());

      m_pathPassedRenderer.setGraphicsContext(&boldGreenGC());
      m_pathPassedRenderer.setHighlightContext(&pathBackgroundGC());
      m_pathPassedRenderer.setIsParent(true);

      m_pathFailedRenderer.setGraphicsContext(&boldRedGC());
      m_pathFailedRenderer.setHighlightContext(&pathBackgroundGC());
      m_pathFailedRenderer.setIsParent(true);
      
      m_editingDisabledRenderer.setGraphicsContext(&italicGray());
      m_editingDisabledRenderer.setHighlightContext(&pathBackgroundGC());

      // Italic color doesn't seem to show up well event though
      // modules are displayed in italic
      m_modulePassedRenderer.setGraphicsContext(&boldGreenGC());
      m_modulePassedRenderer.setIsParent(true);
      m_moduleFailedRenderer.setGraphicsContext(&boldRedGC());
      m_moduleFailedRenderer.setIsParent(true);

      // Debug stuff to dump font list.
//      std::cout << "Available fonts: " << std::endl;
//      gClient->GetFontPool()->Print();
       
      reset();
   }

   /** Update the internal model for the schedule. Notice that
       the actual structure of the model will not change, only
       its contents, because of the way CMSSW is designed,
       hence this method only needs to be called once.
      */
   void updateSchedule(const edm::ScheduleInfo *info)
      {
         if (!m_entries.empty())
            return;
         // Execute only once since the schedule itself
         // cannot be altered.
         assert(m_availablePaths.empty());
         info->availablePaths(m_availablePaths);
         
         for (size_t i = 0, e = m_availablePaths.size(); i != e; ++i)
         {
            PSetData pathEntry;
            const std::string &pathName = m_availablePaths[i];
            pathEntry.label = pathName;
            m_pathIndex.insert(std::make_pair(pathName, m_paths.size()));

            pathEntry.value = "Path";
            pathEntry.level= 0;
            pathEntry.parent = -1;
            pathEntry.path = i;
            pathEntry.editable = false;

            PathInfo pathInfo;
            pathInfo.entryId = m_entries.size();
            pathInfo.passed = false;
            pathInfo.moduleStart = m_modules.size();
            m_paths.push_back(pathInfo);

            m_parentStack.push_back(m_entries.size());
            m_entries.push_back(pathEntry);

            std::vector<std::string> pathModules;
            info->modulesInPath(pathName, pathModules);

            for (size_t mi = 0, me = pathModules.size(); mi != me; ++mi)
            {
               PSetData moduleEntry;

               const edm::ParameterSet* ps = info->parametersForModule(pathModules[mi]);
               const edm::ParameterSet::table& pst = ps->tbl();
               const edm::ParameterSet::table::const_iterator ti = pst.find("@module_edm_type");

               if (ti == pst.end())
                 moduleEntry.value = "Unknown module name";
               else
                 moduleEntry.value = ti->second.getString();

               moduleEntry.label = pathModules[mi];
               moduleEntry.parent = m_parentStack.back();
               moduleEntry.level = m_parentStack.size();
               moduleEntry.module = mi;
               moduleEntry.path = i;
               moduleEntry.pset = *ps;
               moduleEntry.editable = false;
               ModuleInfo moduleInfo;
               moduleInfo.path = m_paths.size() - 1;
               moduleInfo.entry = m_entries.size();
               moduleInfo.passed = false;
               moduleInfo.dirty = false;
               m_modules.push_back(moduleInfo);
               m_parentStack.push_back(m_entries.size());
               m_entries.push_back(moduleEntry);
               handlePSet(moduleEntry.pset);
               m_parentStack.pop_back();
            }
            m_paths.back().moduleEnd = m_modules.size();
            m_parentStack.pop_back();
         }

         // Nothing is expanded by default.
         for (size_t i = 0, e = m_entries.size(); i != e; ++i)
            m_entries[i].expanded = false;
         m_filter = "";
      }


   /** Update the status of a given path. This is the information 
       that changes on event by event basis.
     */
   void update(std::vector<PathUpdate> &pathUpdates)
      {
         // Reset all the path / module status information, so that
         // by default paths and modules are considered "not passed".
         for (size_t pi = 0, pe = m_paths.size(); pi != pe; ++pi)
            m_paths[pi].passed = false;
         for (size_t mi = 0, me = m_modules.size(); mi != me; ++mi)
            m_modules[mi].passed = false;
         
         // Update whether or not a given path / module passed selection.
         for (size_t pui = 0, pue = pathUpdates.size(); pui != pue; ++pui)
         {
            PathUpdate &update = pathUpdates[pui];
            std::map<std::string, size_t>::const_iterator index = m_pathIndex.find(update.pathName);
            if (index == m_pathIndex.end())
            {
               std::cerr << "Path " << update.pathName << "cannot be found!" << std::endl;
               continue;
            }
            PathInfo &pathInfo = m_paths[index->second];
            pathInfo.passed = update.passed;
            
            for (size_t mi = pathInfo.moduleStart, me = pathInfo.moduleEnd; mi != me; ++mi)
            {
               ModuleInfo &moduleInfo = m_modules[mi];
               moduleInfo.passed = update.passed || ((mi-pathInfo.moduleStart) < update.choiceMaker);
            }
         }

         implSort(-1, true);
      }

   std::vector<ModuleInfo> &modules() {return m_modules; }
   std::vector<PSetData> &entries() {return m_entries; }

   virtual void implSort(int, bool)
   {
      // Decide whether or not items match the filter.
      for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      {
         PSetData &data = m_entries[i];
         // First of all decide whether or not we match
         // the filter.
         if (m_filter.empty())
            data.matches = false;
         else if (strstr(data.label.c_str(), m_filter.c_str()))
            data.matches = true;
         else
            data.matches = false;
      }

      // We reset whether or not a given parent has children that match the
      // filter, and we recompute the whole information by checking all the
      // children.
      for (size_t i = 0, e = m_entries.size(); i != e; ++i)
         m_entries[i].childMatches = false;

      std::vector<int> stack;
      int previousLevel = 0;
      for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      {
         PSetData &data = m_entries[i];
         // Top level.
         if (data.parent == (size_t)-1)
         {
            previousLevel = 0;
            continue;
         }
         // If the level is greater than the previous one,
         // it means we are among the children of the 
         // previous level, hence we push the parent to
         // the stack.
         // If the level is not greater than the previous
         // one it means we have popped out n levels of
         // parents, where N is the difference between the 
         // new and the old level. In this case we
         // pop up N parents from the stack.
         if (data.level > previousLevel)
            stack.push_back(data.parent);
         else
            for (size_t pi = 0, pe = previousLevel - data.level; pi != pe; ++pi)
               stack.pop_back();
 
         if (data.matches)
            for (size_t pi = 0, pe = stack.size(); pi != pe; ++pi)
               m_entries[stack[pi]].childMatches = true;

         previousLevel = data.level;
      }
       
      recalculateVisibility();
    }

   void recalculateVisibility()
      {
         m_row_to_index.clear();

         // Decide about visibility.
         // * If the items are toplevel and they match the filter, they get shown
         //   in any case.
         // * If the item or any of its children match the filter, the item
         //   is visible.
         // * If the filter is empty and the parent is expanded.
         for (size_t i = 0, e = m_entries.size(); i != e; ++i)
         {
            PSetData &data = m_entries[i];
            if (data.parent == ((size_t) -1))
               data.visible = data.childMatches || data.matches || m_filter.empty();
            else
               data.visible = data.matches || data.childMatches || (m_filter.empty() && m_entries[data.parent].expanded && m_entries[data.parent].visible);
         }

         // Put in the index only the entries which are visible.
         for (size_t i = 0, e = m_entries.size(); i != e; ++i)
            if (m_entries[i].visible)
               m_row_to_index.push_back(i);
      }

   virtual int unsortedRowNumber(int unsorted) const
   {
      return unsorted;
   }

   virtual int numberOfRows() const {
      return m_row_to_index.size();
   }

   virtual int numberOfColumns() const {
      return 2;
   }
   
   virtual std::vector<std::string> getTitles() const 
   {
      std::vector<std::string> returnValue;
      returnValue.reserve(numberOfColumns());
      returnValue.push_back("Label");
      returnValue.push_back("Value");
      return returnValue;
   }
  
   virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const
   {
      // If the index is outside the table, we simply return an empty cell.
      // FIXME: how is this actually possible???
      if (static_cast<int>(m_row_to_index.size()) <= iSortedRowNumber)
      {
         m_renderer.setData(std::string(), false);
         return &m_renderer;
      }

      // Do the actual rendering.
      FWTextTreeCellRenderer* renderer;

      int unsortedRow =  m_row_to_index[iSortedRowNumber];
      const PSetData& data = m_entries[unsortedRow];

      std::string value;
      std::string label;

      if (data.level == 0)
      {
         const PathInfo &path = m_paths[data.path];
         label = data.label + " (" + data.value + ")";
       
         value = "";

         if (path.passed)
           renderer = &m_pathPassedRenderer;
         else 
           renderer = &m_pathFailedRenderer;
         if( 0 == iCol ) {
           renderer->setIsParent(true);
         } else {
           renderer->setIsParent(false);
         }
      }
      else if (data.level == 1)
      {
         const ModuleInfo &module = m_modules[m_paths[data.path].moduleStart + data.module];

         label = data.label + " (" + data.value + ")";
         value = "";

         // "passed" means if module made decision on path 
         // passing or failing
         if (module.passed)
           renderer = &m_modulePassedRenderer;
         else
           renderer = &m_moduleFailedRenderer;
         if( 0 == iCol ) {
           renderer->setIsParent(true);
         } else {
           renderer->setIsParent(false);
         }
      }
      else
      {
         if (data.type > 0)
            label = data.label + " (" + sTypeTranslations.table_[data.type] + ")";
         else
            label = data.label;
         value = data.value;
         
         if (data.editable)
            renderer = &m_renderer;
         else
            renderer = &m_editingDisabledRenderer;
      }

      renderer->setIndentation(0);
      if(data.expanded) {
        renderer->setIsOpen(true);
      } else {
        renderer->setIsOpen(false);        
      }

      if (iCol == 0)
      {
         renderer->setIndentation(data.level * 10);
         renderer->setData(label, false);
      }
      else if (iCol == 1)
         renderer->setData(value, false);
      else
         renderer->setData(std::string(), false);

      // If we are rendering the selected cell,
      // we show the editor.
      if (iCol == 1 && iSortedRowNumber == m_selectedRow && iCol == m_selectedColumn)
         renderer->showEditor(data.editable);
      else
         renderer->showEditor(false);

      return renderer;
   }

   void setExpanded(int row)
      {
         if (row == -1)
            return;
         // We do not want to handle expansion
         // events while in filtering mode.
         if (!m_filter.empty())
            return;
         int index = rowToIndex()[row];
         PSetData& data = m_entries[index];

         data.expanded = !data.expanded;
         recalculateVisibility();
         dataChanged();
         visualPropertiesChanged();
      }

   template <class T>
   bool editNumericParameter(edm::ParameterSet &ps, bool tracked, 
                             const std::string &label, 
                             const std::string &value) 
      {
         std::stringstream  str(value);
         T v;
         str >> v;
         bool fail = str.fail();
         if (tracked)
            ps.addParameter(label, v);
         else
            ps.addUntrackedParameter(label, v);
         
         return fail;
      }

   void editStringParameter(edm::ParameterSet &ps, bool tracked,
                            const std::string &label,
                            const std::string &value)
      {
         if (tracked)
            ps.addParameter(label, value);
         else
            ps.addUntrackedParameter(label, value);
      }

   void editFileInPath(edm::ParameterSet &ps, bool tracked,
                       const std::string &label,
                       const std::string &value)
      {
        if (tracked)
          ps.addParameter(label, edm::FileInPath(value));
        else
          ps.addUntrackedParameter(label, edm::FileInPath(value));
      }

   bool editVInputTag(edm::ParameterSet &ps, bool tracked,
                      const std::string &label,
                      const std::string &value)
      { 
        std::vector<edm::InputTag> inputTags;
        std::stringstream iss(value);
        std::string vitem;
        bool fail = false;
        size_t fst, lst;

        while (getline(iss, vitem, ','))
        {
          fst = vitem.find("[");
          lst = vitem.find("]");
        
          if ( fst != std::string::npos )
            vitem.erase(fst,1);
          if ( lst != std::string::npos )
            vitem.erase(lst,1);
        
          std::vector<std::string> tokens = edm::tokenize(vitem, ":");
          size_t nwords = tokens.size();
        
          if ( nwords > 3 )
          {
            fail = true;
            return fail;
          }
          else 
          {
            std::string it_label("");
            std::string it_instance("");
            std::string it_process("");

            if ( nwords > 0 ) 
              it_label = tokens[0];
            if ( nwords > 1 ) 
              it_instance = tokens[1];
            if ( nwords > 2 ) 
              it_process  = tokens[2];
        
            inputTags.push_back(edm::InputTag(it_label, it_instance, it_process));
          }
        }
     
        if (tracked)
          ps.addParameter(label, inputTags);
        else
          ps.addUntrackedParameter(label, inputTags);

        return fail;
      }
  

   bool editInputTag(edm::ParameterSet &ps, bool tracked,
                     const std::string &label,
                     const std::string &value)
      {
        std::vector<std::string> tokens = edm::tokenize(value, ":");
        size_t nwords = tokens.size();
     
        bool fail;

        if ( nwords > 3 ) 
        {
          fail = true;
        }
        else
        {           
          std::string it_label("");
          std::string it_instance("");
          std::string it_process("");

          if ( nwords > 0 ) 
            it_label = tokens[0];
          if ( nwords > 1 ) 
            it_instance = tokens[1];
          if ( nwords > 2 ) 
            it_process  = tokens[2];

          if ( tracked )
            ps.addParameter(label, edm::InputTag(it_label, it_instance, it_process));
          else
            ps.addUntrackedParameter(label, edm::InputTag(it_label, it_instance, it_process));
            
          fail = false;
        }
           
        return fail;
      }

   bool editESInputTag(edm::ParameterSet &ps, bool tracked,
                       const std::string &label,
                       const std::string &value)
      {
        std::vector<std::string> tokens = edm::tokenize(value, ":");
        size_t nwords = tokens.size();
      
        bool fail;
  
        if ( nwords > 2 )
        {
          fail = true;    
        }
        else
        {             
          std::string it_module("");
          std::string it_data("");

          if ( nwords > 0 ) 
            it_module = tokens[0];
          if ( nwords > 1 ) 
            it_data = tokens[1];

          if ( tracked )
            ps.addParameter(label, edm::ESInputTag(it_module, it_data));
          else
            ps.addUntrackedParameter(label, edm::ESInputTag(it_module, it_data));
        
          fail = false;
        }

        return fail;
      }
  
  template <typename T>
  void editVectorParameter(edm::ParameterSet &ps, bool tracked,
                           const std::string &label,
                           const std::string &value)
    {
      std::vector<T> valueVector;
      
      std::stringstream iss(value);
      std::string vitem;
      
      size_t fst, lst;

      while (getline(iss, vitem, ','))
      {
        fst = vitem.find("[");
        lst = vitem.find("]");
        
        if ( fst != std::string::npos )
          vitem.erase(fst,1);
        if ( lst != std::string::npos )
          vitem.erase(lst,1);
        
        std::stringstream oss(vitem);
        T on;
        oss >> on;

        valueVector.push_back(on);
      }
     
      if (tracked)
        ps.addParameter(label, valueVector);
      else
        ps.addUntrackedParameter(label, valueVector);
    }
  
   /** Does not apply changes and closes window. */
   void cancelEditor()
      {
         if (!m_editor)
            return;
         
         m_editor->UnmapWindow(); 
         
      }

   /** This is invoked every single time the
       editor contents must be applied to the selected entry in the pset. 
       @return true on success. 
     */
   bool applyEditor()
      {
         if (!m_editor)
            return false;
         
         if (m_selectedRow == -1)
            return false;
         if (m_selectedColumn != 1)
         {
            m_editor->UnmapWindow();
            return false;
         }
         
         PSetData &data = m_entries[m_row_to_index[m_selectedRow]];
         PSetData &parent = m_entries[data.parent];
         try
         {
            switch (data.type)
            {
               case 'I':
                  editNumericParameter<int32_t>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'U':
                  editNumericParameter<uint32_t>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'D':
                  editNumericParameter<double>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'L':
                  editNumericParameter<long long>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'X':
                  editNumericParameter<unsigned long long>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'S':
                  editStringParameter(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'i':
                  editVectorParameter<int32_t>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'u':
                  editVectorParameter<uint32_t>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'l':
                  editVectorParameter<long long>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'x':
                  editVectorParameter<unsigned long long>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'd':
                  editVectorParameter<double>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 's':
                  editVectorParameter<std::string>(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break; 
               case 't':
                  editInputTag(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'g':
                  editESInputTag(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'v':
                  editVInputTag(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               case 'F':
                  editFileInPath(parent.pset, data.tracked, data.label, m_editor->GetText());
                  break;
               default:
                  std::cerr << "unsupported parameter" << std::endl;
                  m_editor->UnmapWindow();
                  return false;
            }
            data.value = m_editor->GetText();
            m_modules[data.module].dirty = true;
            m_editor->UnmapWindow();
         }
         catch(cms::Exception &e)
         {
            m_editor->SetForegroundColor(gVirtualX->GetPixel(kRed));
         }
         return true;
      }

   void setSelection (int row, int column, int mask) {
      if(mask == 4) {
         if( row == m_selectedRow) {
            row = -1;
         }
      }
      changeSelection(row, column);
   }

   virtual const std::string title() const {
      return "Modules & their parameters";
   }

   int selectedRow() const {
      return m_selectedRow;
   }

   int selectedColumn() const {
      return m_selectedColumn;
   }
   //virtual void sort (int col, bool reset = false);
   virtual bool rowIsSelected(int row) const 
   {
      return m_selectedRow == row;
   }

   void reset() 
   {
      changeSelection(-1, -1);
      recalculateVisibility();
      dataChanged();
      visualPropertiesChanged();
   }

   void handlePSetEntry(const edm::ParameterSetEntry& entry, const std::string& key)
   {
      PSetData data;
      data.label = key;
      data.tracked = entry.isTracked();
      data.level = m_parentStack.size();
      data.parent = m_parentStack.back();
      data.type = 'P';
      data.module = m_modules.size() - 1;
      data.path = m_paths.size() - 1;
      data.pset = entry.pset();
      data.editable = false;
      m_parentStack.push_back(m_entries.size());
      m_entries.push_back(data);

      handlePSet(entry.pset());
      m_parentStack.pop_back();
   }

   void handleVPSetEntry(const edm::VParameterSetEntry& entry,
                         const std::string& key)
   {
      PSetData data;
      data.label = key;
      data.tracked = entry.isTracked();
      data.level = m_parentStack.size();
      data.parent = m_parentStack.back();
      data.type = 'p';
      data.module = m_modules.size() - 1;
      data.path = m_paths.size() - 1;
      data.editable = false;
      m_parentStack.push_back(m_entries.size());
      m_entries.push_back(data);

      std::stringstream ss;

      for (size_t i = 0, e = entry.vpset().size(); i != e; ++i)
      {
          ss.str("");
          ss << key << "[" << i << "]";
          PSetData vdata;
          vdata.label = ss.str();
          vdata.tracked = entry.isTracked();
          vdata.level = m_parentStack.size();
          vdata.parent = m_parentStack.back();
          vdata.module = m_modules.size() - 1;
          vdata.path = m_paths.size() - 1;
          vdata.editable = false;
          m_parentStack.push_back(m_entries.size());
          m_entries.push_back(vdata);
          handlePSet(entry.vpset()[i]);
          m_parentStack.pop_back();
      }
      m_parentStack.pop_back();
   }

   void handlePSet(const edm::ParameterSet &ps)
   {
      typedef edm::ParameterSet::table::const_iterator TIterator;
      for (TIterator i = ps.tbl().begin(), e = ps.tbl().end(); i != e; ++i)
         handleEntry(i->second, i->first);

      typedef edm::ParameterSet::psettable::const_iterator PSIterator;
      for (PSIterator i = ps.psetTable().begin(), e = ps.psetTable().end(); i != e; ++i)
         handlePSetEntry(i->second, i->first);

      typedef edm::ParameterSet::vpsettable::const_iterator VPSIterator;
      for (VPSIterator i = ps.vpsetTable().begin(), e = ps.vpsetTable().end(); i != e; ++i)
         handleVPSetEntry(i->second, i->first);
   }
    
   template <class T>
   void createScalarString(PSetData &data, T v)
   {
      std::stringstream ss;
      ss << v;
      data.value = ss.str();
      m_entries.push_back(data);
   }

   template <typename T>
   void createVectorString(PSetData &data, const T &v, bool quotes)
   {
      std::stringstream ss;
      ss << "[";
      for (size_t ii = 0, ie = v.size(); ii != ie; ++ii)
      {
         if (quotes)
            ss << "\""; 
         ss << v[ii];
         if (quotes)
            ss << "\"";
         if (ii + 1 != ie) 
            ss << ", ";
      }
      ss << "]";
      data.value = ss.str();
      m_entries.push_back(data);
   }

   virtual void sortWithFilter(const char *filter)
      {
         m_filter = filter;
         sort(-1, true);
         dataChanged();
      }

   void setCellValueEditor(TGTextEntry *editor)
      {
         m_editor = editor;
         m_renderer.setCellEditor(editor);
      }

   void handleEntry(const edm::Entry &entry,const std::string &key)
   {
      std::stringstream ss;
      PSetData data;
      data.label = key;
      data.tracked = entry.isTracked();
      data.type = entry.typeCode();
      data.level = m_parentStack.size();
      data.parent = m_parentStack.back();
      data.module = m_modules.size() - 1;
      data.type = entry.typeCode();
      if (data.label[0] == '@' || data.level > 2)
         data.editable = false;
      else
         data.editable = true;

      switch(entry.typeCode())
      {
      case 'b':
        {
          data.value = entry.getBool() ? "True" : "False";
          m_entries.push_back(data);
          break;
        }
      case 'B':
        {
          data.value = entry.getBool() ? "True" : "False";
          m_entries.push_back(data);
          break;
        }
      case 'i':
        {
          createVectorString(data, entry.getVInt32(), false);
          break;
        }
      case 'I':
         {
           createScalarString(data, entry.getInt32());
           break;
         }
      case 'u':
         {
           createVectorString(data, entry.getVUInt32(), false);
           break;
         }
      case 'U':
         {
           createScalarString(data, entry.getUInt32());
           break;
         }
      case 'l':
         {
           createVectorString(data, entry.getVInt64(), false);
           break;
         }
      case 'L':
         {
            createScalarString(data, entry.getInt32());
            break;
         }
      case 'x':
         {
           createVectorString(data, entry.getVUInt64(), false);
           break;
         }
      case 'X':
         {
           createScalarString(data, entry.getUInt64());
           break;
         }
      case 's':
         {
           createVectorString(data, entry.getVString(), false);
           break;
         }
      case 'S':
         {
           createScalarString(data, entry.getString());
           break;
         }
      case 'd':
         {
           createVectorString(data, entry.getVDouble(), false);
           break;
         }
      case 'D':
         { 
           createScalarString(data, entry.getDouble());
           break;
         }
      case 'p':
        {
          std::vector<edm::ParameterSet> psets = entry.getVPSet();
          for (size_t psi = 0, pse = psets.size(); psi != pse; ++psi)
            handlePSet(psets[psi]);
          break;
        }
      case 'P':
        {    
          handlePSet(entry.getPSet());
          break;
        }
      case 't':
         {
           data.value = entry.getInputTag().encode();
           m_entries.push_back(data);
           break;
         } 
      case 'v':
         {
           std::vector<std::string> tags;
           tags.resize(entry.getVInputTag().size());
           for (size_t iti = 0, ite = tags.size(); iti != ite; ++iti) 
             tags[iti] = entry.getVInputTag()[iti].encode();
           createVectorString(data, tags, true);
           break;
         }        
      case 'g':
        {
          data.value = entry.getESInputTag().encode();
          m_entries.push_back(data);
          break;
        }
      case 'G':
        {
          std::vector<std::string> tags;
          tags.resize(entry.getVESInputTag().size());
          for (size_t iti = 0, ite = tags.size(); iti != ite; ++iti) 
            tags[iti] = entry.getVESInputTag()[iti].encode();
          createVectorString(data, tags, true);
          break;
        }
      case 'F':
        {
          createScalarString(data, entry.getFileInPath().relativePath());
          break;
        }
      case 'e':
        {
          data.editable = false;
          std::vector<edm::EventID> ids;
          ids.resize(entry.getVEventID().size());
          for ( size_t iri = 0, ire = ids.size(); iri != ire; ++iri )
            ids[iri] = entry.getVEventID()[iri];
          createVectorString(data, ids, true);
          break;
        }
      case 'E':
        {
          data.editable = false;
          createScalarString(data, entry.getEventID());
          break;
        }
      case 'm':
        {
          data.editable = false;
          std::vector<edm::LuminosityBlockID> ids;
          ids.resize(entry.getVLuminosityBlockID().size());
          for ( size_t iri = 0, ire = ids.size(); iri != ire; ++iri )
            ids[iri] = entry.getVLuminosityBlockID()[iri];
          createVectorString(data, ids, true);
          break;
        }
      case 'M':
        {
          data.editable = false;
          createScalarString(data, entry.getLuminosityBlockID());
          break;
        }
      case 'a':
        {
          data.editable = false;
          std::vector<edm::LuminosityBlockRange> ranges;
          ranges.resize(entry.getVLuminosityBlockRange().size());
          for ( size_t iri = 0, ire = ranges.size(); iri != ire; ++iri )
            ranges[iri] = entry.getVLuminosityBlockRange()[iri];
          createVectorString(data, ranges, true);
          break;
        }
      case 'A':
        {
          data.editable = false;
          createScalarString(data, entry.getLuminosityBlockRange());
          break;
        }
      case 'r':
        {
          data.editable = false;
          std::vector<edm::EventRange> ranges;
          ranges.resize(entry.getVEventRange().size());
          for ( size_t iri = 0, ire = ranges.size(); iri != ire; ++iri )
            ranges[iri] = entry.getVEventRange()[iri];
          createVectorString(data, ranges, true);
          break;
        }
      case 'R':
        {
          data.editable = false;
          createScalarString(data, entry.getEventRange());
          break;          
        }
      default:
        {
          break;
        }
      }
   }

   std::vector<PSetData> &data()  { return m_entries; }
   std::vector<int> &rowToIndex() { return m_row_to_index; }
  
   sigc::signal<void,int,int> indexSelected_;

private:
   void changeSelection(int iRow, int iColumn)
   {
      // Nothing changes if we clicked selected
      // twice the same cell.
      if (iRow == m_selectedRow && iColumn == m_selectedColumn)
         return;

      // Otherwise update the selection information
      // and notify observers.
      m_selectedRow = iRow;
      m_selectedColumn = iColumn;

      indexSelected_(iRow, iColumn);
      visualPropertiesChanged();
   }

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
   TGTextEntry                    *m_editor;
   std::vector<std::string>        m_availablePaths;

   mutable FWTextTreeCellRenderer m_renderer;  
   mutable FWTextTreeCellRenderer m_italicRenderer;
   mutable FWTextTreeCellRenderer m_boldRenderer;

   mutable FWTextTreeCellRenderer m_pathPassedRenderer;
   mutable FWTextTreeCellRenderer m_pathFailedRenderer;

   // To be used to renderer cells that should appear as disabled.
   mutable FWTextTreeCellRenderer m_editingDisabledRenderer;

   mutable FWTextTreeCellRenderer m_modulePassedRenderer;
   mutable FWTextTreeCellRenderer m_moduleFailedRenderer;
};

void
FWPathsPopup::windowIsClosing()
{
   UnmapWindow();
   DontCallClose();
}

FWPathsPopup::FWPathsPopup(FWFFLooper *looper, FWGUIManager *guiManager)
   : TGMainFrame(gClient->GetRoot(), 400, 600),
     m_info(0),
     m_looper(looper),
     m_hasChanges(false),
     m_moduleLabel(0),
     m_moduleName(0),
     m_apply(0),
     m_psTable(new FWPSetTableManager()),
     m_guiManager(guiManager)
{
   gVirtualX->SelectInput(GetId(), kKeyPressMask | kKeyReleaseMask | kExposureMask |
                                   kPointerMotionMask | kStructureNotifyMask | kFocusChangeMask |
                                   kEnterWindowMask | kLeaveWindowMask);
   m_psTable->indexSelected_.connect(boost::bind(&FWPathsPopup::newIndexSelected,this,_1,_2));
   this->Connect("CloseWindow()","FWPathsPopup",this,"windowIsClosing()");

   FWDialogBuilder builder(this);
   builder.indent(4)
          .spaceDown(10)
          .addLabel("Filter:").floatLeft(4).expand(false, false)
          .addTextEntry("", &m_search).expand(true, false)
          .spaceDown(10)
          .addTable(m_psTable, &m_tableWidget).expand(true, true)
          .addTextButton("Apply changes and reload", &m_apply);

   TGTextEntry *editor = new TGTextEntry(m_tableWidget->body(), "");
   editor->SetBackgroundColor(gVirtualX->GetPixel(kYellow-7));
   m_psTable->setCellValueEditor(editor);
   editor->Connect("ReturnPressed()", "FWPathsPopup", this, "applyEditor()");

   m_apply->Connect("Clicked()", "FWPathsPopup", this, "scheduleReloadEvent()");
   m_apply->SetEnabled(false);
   m_search->SetEnabled(true);
   m_search->Connect("TextChanged(const char *)", "FWPathsPopup",
                     this, "updateFilterString(const char *)");
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);
   m_tableWidget->Connect("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",
                          "FWPathsPopup",this,
                          "cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)");
   m_tableWidget->Connect("Clicked()", "FWPathsPopup", this, "applyEditor()");
   MapSubwindows();
   editor->UnmapWindow();
   Layout();
}

/** Handle pressing of esc.
    FIXME: this still does not work if the cursor is on the editor widget.
 */
Bool_t
FWPathsPopup::HandleEvent(Event_t*event)
{
   if (event->fType != (int) kGKeyPress)
      return TGMainFrame::HandleEvent(event);

   UInt_t keysym = event->fCode;

   if (keysym == (UInt_t) gVirtualX->KeysymToKeycode(kKey_Escape))
   {
      m_psTable->cancelEditor();
      m_psTable->setSelection(-1, -1, 0);
   }
   return TGMainFrame::HandleEvent(event);
}

/** Proxies the applyEditor() method in the model so that it can be connected to GUI, signals.
  */
void
FWPathsPopup::applyEditor()
{
   bool applied = m_psTable->applyEditor();
   m_psTable->setSelection(-1, -1, 0);
   if (applied)
      m_apply->SetEnabled(true);
}

/** Handles clicking on the table cells.
    
    * Clicking on a cell in the first column opens / closes a given node. 
    * Clicking on a cell in the second column moves the editor to that cell. 
 
  */
void 
FWPathsPopup::cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t, Int_t)
{
   if (iButton != kButton1)
      return;
   
   if (iColumn == 0)
   {
      m_psTable->setExpanded(iRow);

      // Save and close the previous editor, if required.
      if (m_psTable->selectedColumn() == 1
          && m_psTable->selectedRow() != -1)
      {
         int oldIndex = m_psTable->rowToIndex()[m_psTable->selectedRow()];
         PSetData& oldData = m_psTable->data()[oldIndex];

         if (oldData.editable)
            applyEditor();
      }

      m_psTable->setSelection(iRow, iColumn, iKeyMod);
   }
   else if (iColumn == 1)
   {
      // If we selected a new cell, save the previous, if required.
      if (m_psTable->selectedColumn() == 1 
          && m_psTable->selectedRow() != -1)
      {
         int oldIndex = m_psTable->rowToIndex()[m_psTable->selectedRow()];
         PSetData& oldData = m_psTable->data()[oldIndex];
         if (oldData.editable)
            applyEditor();
      }
         
      // Clear text on new row click
      // FIXME: int index = m_psTable->rowToIndex()[iRow];
      m_psTable->setSelection(iRow, iColumn, iKeyMod);
   }
}

void
FWPathsPopup::newIndexSelected(int iSelectedRow, int iSelectedColumn)
{
   if (iSelectedRow == -1)
      return;

   m_psTable->sortWithFilter(m_search->GetText());
   m_psTable->dataChanged();
}

void
FWPathsPopup::updateFilterString(const char *str)
{
   m_psTable->applyEditor();
   m_psTable->setSelection(-1, -1, 0);
   m_psTable->sortWithFilter(str);
}

/** Finish the setup of the GUI */
void
FWPathsPopup::setup(const edm::ScheduleInfo *info)
{
   assert(info);
   m_info = info;
}

/** Gets called by CMSSW as modules are about to be processed. **/
void
FWPathsPopup::postModule(edm::ModuleDescription const& description)
{
   m_guiManager->updateStatus((description.moduleName() + " processed.").c_str());
   gSystem->ProcessEvents();
}

/** Gets called by CMSSW as we process modules. **/
void
FWPathsPopup::preModule(edm::ModuleDescription const& description)
{
   m_guiManager->updateStatus(("Processing " + description.moduleName() + "...").c_str());
   gSystem->ProcessEvents();
}


void
FWPathsPopup::postProcessEvent(edm::Event const& event, edm::EventSetup const& eventSetup)
{
   m_guiManager->updateStatus("Done processing.");
   gSystem->ProcessEvents();

   // Get the last process name from the process history:
   // this should be the one specified in the cfg file
   edm::ProcessHistory::const_iterator pi = event.processHistory().end() - 1;
   std::string processName = pi->processName();
   
   // It's called TriggerResults but actually contains info on all paths
   edm::InputTag tag("TriggerResults", "", processName);
   edm::Handle<edm::TriggerResults> triggerResults;
   event.getByLabel(tag, triggerResults);

   std::vector<PathUpdate> pathUpdates;

   if (triggerResults.isValid())
   {
      edm::TriggerNames triggerNames = event.triggerNames(*triggerResults);
     
      for (size_t i = 0, e = triggerResults->size(); i != e; ++i)
      {
         PathUpdate update;
         update.pathName = triggerNames.triggerName(i);
         update.passed = triggerResults->accept(i);
         update.choiceMaker = triggerResults->index(i);
         pathUpdates.push_back(update);
      }
   }
   m_psTable->updateSchedule(m_info);
   m_psTable->update(pathUpdates);
   m_psTable->dataChanged();
   m_tableWidget->body()->DoRedraw();
}

#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace boost::python;


/** Modifies the module and asks the looper to reload the event.
 
    1. Read the configuration snippet from the GUI,
    2. Use the python interpreter to parse it and get the new
      parameter set.
    3. Notify the looper about the changes.

    FIXME: implement 2 and 3.
  */
void
FWPathsPopup::scheduleReloadEvent()
{
   applyEditor();
   try
   {
      for (size_t mni = 0, mne = m_psTable->modules().size(); mni != mne; ++mni)
      {
         ModuleInfo &module = m_psTable->modules()[mni];
         if (module.dirty == false)
            continue;
         PSetData &data = m_psTable->entries()[module.entry];
         m_looper->requestChanges(data.label, data.pset);
      }
      m_hasChanges = true;
      m_apply->SetEnabled(false);
      gSystem->ExitLoop();
   }
   catch (boost::python::error_already_set)
   {
      edm::pythonToCppException("Configuration");
      Py_Finalize();
   }
   catch (cms::Exception &exception)
   {
      std::cout << exception.what() << std::endl;
   }
   // Return control to the FWFFLooper so that it can decide what to do next.
}
