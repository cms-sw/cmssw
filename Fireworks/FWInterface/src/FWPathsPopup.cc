#include "Fireworks/FWInterface/src/FWPathsPopup.h"
#include "Fireworks/FWInterface/interface/FWFFLooper.h"
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"

#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/TriggerNames.h"

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

#include <iostream>
#include <sstream>
#include <cstring>
#include <map>

class FWTextTreeCellRenderer : public FWTextTableCellRenderer
{
public:
   FWTextTreeCellRenderer(const TGGC* iContext = &(getDefaultGC()),
                          const TGGC* iHighlightContext = &(getHighlightGC()),
                          Justify iJustify = kJustifyLeft)
      : FWTextTableCellRenderer(iContext, iHighlightContext, iJustify),
        m_indentation(0),
        m_editor(0),
        m_showEditor(false)
      {}

   virtual void setIndentation(int indentation = 0) { m_indentation = indentation; }
   virtual void setCellEditor(TGTextEntry *editor) { m_editor = editor; }
   virtual void showEditor(bool value) { m_showEditor = value; }
   virtual UInt_t width() { return FWTextTableCellRenderer::width() + m_indentation * 2; }
   virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
      {
         if (m_showEditor && m_editor)
         {
            m_editor->MoveResize(iX-3, iY+iHeight+3, iWidth + 6 , iHeight + 6);
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
         FontMetrics_t metrics;
         font()->GetFontMetrics(&metrics);
         gVirtualX->DrawString(iID, graphicsContext()->GetGC(),
                               iX+m_indentation, iY+metrics.fAscent, 
                               data().c_str(),data().size());
      }
private:
   int            m_indentation;
   TGTextEntry    *m_editor;
   bool           m_showEditor;
};

/** Custom structure for holding the table contents */
struct PSetData
{
   std::string label;
   std::string value;
   int         level;
   bool        tracked;
   std::string type;
   size_t      parent;
   // Whether or not it matches the filter.
   bool        matches;
   // Whether or not it is expanded.
   bool        expanded;
   // Whether or not it is visibile.  Being visible is given by either matching
   // a non-null filter, or  or the parent being visibible and expanded.
   bool        visible;
   // Whether or not any of the children matches the filter.
   bool        childMatches;
   // For paths (i.e. level == 0), whether or not it "passed"
   // For modules, whether or not it made the "decision" 
   bool        passed;
   // For modules, if the parent (i.e. the path) passed
   bool        parentPassed;
};

/** string is whether or not a path passed and 
    int is index of module that made decision **/
typedef std::pair<bool, int> PathData;

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

      m_pathFailedRenderer.setGraphicsContext(&boldRedGC());
      m_pathFailedRenderer.setHighlightContext(&pathBackgroundGC());

      // Italic color doesn't seem to show up well event though
      // modules are displayed in italic
      m_modulePassedRenderer.setGraphicsContext(&boldGreenGC());
      m_moduleFailedRenderer.setGraphicsContext(&boldRedGC());

      std::cout << "Available fonts: " << std::endl;
      gClient->GetFontPool()->Print();
       
      reset();
   }
   
  void update(const edm::ScheduleInfo *info, std::map<std::string, PathData> &pathStatus)
   {
      std::map<std::string, PathData>::const_iterator pi;
      std::map<std::string, PathData>::const_iterator pe = pathStatus.end();

      m_entries.clear();
      std::vector<std::string> availablePaths;
      info->availablePaths(availablePaths);
      for (size_t i = 0, e = availablePaths.size(); i != e; ++i)
      {
         PSetData pathEntry;
         pathEntry.label = availablePaths[i];

         pi = pathStatus.find(availablePaths[i]);

         pathEntry.parentPassed = false; // Always for a path
         size_t mindex;
   
         if ( pi == pe )
         {
           pathEntry.passed = false; // What is sensible to do here?
           mindex = -1;
         }
         else
         {
           pathEntry.passed = (pi->second).first;
           mindex = (pi->second).second;
         }

         pathEntry.value = "Path";
         pathEntry.level= 0;
         pathEntry.parent = -1;
         m_parentStack.push_back(m_entries.size());
         m_entries.push_back(pathEntry);

         std::vector<std::string> pathModules;
         info->modulesInPath(availablePaths[i], pathModules);

         for (size_t mi = 0, me = pathModules.size(); mi != me; ++mi)
         {
            PSetData moduleEntry;
         
            if ( mi == mindex )
              moduleEntry.passed = true;
            else
              moduleEntry.passed = false;
  
            moduleEntry.parentPassed = pathEntry.passed;

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
            m_parentStack.push_back(m_entries.size());
            m_entries.push_back(moduleEntry);
            handlePSet(*ps);
            m_parentStack.pop_back();
         }
         m_parentStack.pop_back();
      }
      
      // Nothing is expanded by default.
      for (size_t i = 0, e = m_entries.size(); i != e; ++i)
         m_entries[i].expanded = false;
      m_filter = "";
      implSort(-1, true);
   }

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
            data.visible = data.matches || data.childMatches || (m_filter.empty() && m_entries[data.parent].expanded);
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
         label = data.label + " (" + data.value + ")";
       
         value = "";

         if ( data.passed )
           renderer = &m_pathPassedRenderer;
         else 
           renderer = &m_pathFailedRenderer;
      }
      else if (data.level == 1)
      {
         label = data.label + " (" + data.value + ")";
         value = "";

         // "passed" means if module made decision on path 
         // passing or failing
         if ( data.passed )
         {
            if ( data.parentPassed )
              renderer = &m_modulePassedRenderer;
            else
              renderer = &m_moduleFailedRenderer;
         }
         else 
            renderer = &m_italicRenderer;
      }
      else
      {
         if (!data.type.empty())
            label = data.label + " (" + data.type + ")";
         else
            label = data.label;
         value = data.value;
         renderer = &m_renderer;
      }

      renderer->setIndentation(0);

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
         renderer->showEditor(true);
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

   /** This is invoked every single time the
       editor contents must */
   void applyEditor()
      {
         if (!m_editor)
            return;
         m_editor->UnmapWindow();
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
      data.type = "PSet";
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
      data.type = "vPSet";
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
         if (ii + 1 != ie) 
            ss << ", ";
         if (quotes)
            ss << "\"";
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

      switch(entry.typeCode())
      {
      case 'b':
        {
          data.type  = "Bool";
          data.value = entry.getBool() ? "True" : "False";
          m_entries.push_back(data);
          break;
        }
      case 'B':
        {
          data.type  = "Bool";
          data.value = entry.getBool() ? "True" : "False";
          m_entries.push_back(data);
          break;
        }
      case 'i':
        {
          data.type = "vint32";
          createVectorString(data, entry.getVInt32(), false);
          break;
        }
      case 'I':
         {
           data.type = "int32";
           createScalarString(data, entry.getInt32());
           break;
         }
      case 'u':
         {
           data.type = "vuint32";
           createVectorString(data, entry.getVUInt32(), false);
           break;
         }
      case 'U':
         {
           data.type = "uint32";
           createScalarString(data, entry.getUInt32());
           break;
         }
      case 'l':
         {
           data.type = "vint64";
           createVectorString(data, entry.getVInt64(), false);
           break;
         }
      case 'L':
         {
           data.type = "int64";
            createScalarString(data, entry.getInt32());
            break;
         }
      case 'x':
         {
           data.type = "vuint64";
           createVectorString(data, entry.getVUInt64(), false);
           break;
         }
      case 'X':
         {
           data.type = "uint64";
           createScalarString(data, entry.getUInt64());
           break;
         }
      case 's':
         {
           data.type = "vstring";
           createVectorString(data, entry.getVString(), false);
           break;
         }
      case 'S':
         {
           data.type = "string";
           createScalarString(data, entry.getString());
           break;
         }
      case 'd':
         {
           data.type = "vdouble";
           createVectorString(data, entry.getVDouble(), false);
           break;
         }
      case 'D':
         { 
           data.type = "double";
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
           data.type = "InputTag";
           data.value = entry.getInputTag().encode();
           break;
         } 
      case 'v':
         {
           data.type = "VInputTag";
           std::vector<std::string> tags;
           tags.resize(entry.getVInputTag().size());
           for (size_t iti = 0, ite = tags.size(); iti != ite; ++iti) 
             tags[iti] = entry.getVInputTag()[iti].encode();
           createVectorString(data, tags, true);
           break;
         }        
      case 'F':
        {
          data.type = "FileInPath";
          entry.getFileInPath().write(ss);
          createScalarString(data, ss.str());
          break;
        }
      case 'e':
        {
          data.type = "VEventID";
          std::vector<edm::MinimalEventID> ids;
          ids.resize(entry.getVEventID().size());
          for ( size_t iri = 0, ire = ids.size(); iri != ire; ++iri )
            ids[iri] = entry.getVEventID()[iri];
          createVectorString(data, ids, true);
          break;
        }
      case 'E':
        {
          data.type = "EventID";
          createScalarString(data, entry.getEventID());
          break;
        }
      case 'm':
        {
          data.type = "VLuminosityBlockID";
          std::vector<edm::LuminosityBlockID> ids;
          ids.resize(entry.getVLuminosityBlockID().size());
          for ( size_t iri = 0, ire = ids.size(); iri != ire; ++iri )
            ids[iri] = entry.getVLuminosityBlockID()[iri];
          createVectorString(data, ids, true);
          break;
        }
      case 'M':
        {
          data.type = "LuminosityBlockID";
          createScalarString(data, entry.getLuminosityBlockID());
          break;
        }
      case 'a':
        {
          data.type = "VLuminosityBlockRange";
          std::vector<edm::LuminosityBlockRange> ranges;
          ranges.resize(entry.getVLuminosityBlockRange().size());
          for ( size_t iri = 0, ire = ranges.size(); iri != ire; ++iri )
            ranges[iri] = entry.getVLuminosityBlockRange()[iri];
          createVectorString(data, ranges, true);
          break;
        }
      case 'A':
        {
          data.type = "LuminosityBlockRange";
          createScalarString(data, entry.getLuminosityBlockRange());
          break;
        }
      case 'r':
        {
          data.type = "VEventRange";
          std::vector<edm::EventRange> ranges;
          ranges.resize(entry.getVEventRange().size());
          for ( size_t iri = 0, ire = ranges.size(); iri != ire; ++iri )
            ranges[iri] = entry.getVEventRange()[iri];
          createVectorString(data, ranges, true);
          break;
        }
      case 'R':
        {
          data.type = "EventRange";
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
   std::vector<size_t>             m_parentStack;
   std::vector<int>                m_row_to_index;
   int                             m_selectedRow;
   int                             m_selectedColumn;
   std::string                     m_filter;
   TGTextEntry                    *m_editor;

   mutable FWTextTreeCellRenderer m_renderer;  
   mutable FWTextTreeCellRenderer m_italicRenderer;
   mutable FWTextTreeCellRenderer m_boldRenderer;

   mutable FWTextTreeCellRenderer m_pathPassedRenderer;
   mutable FWTextTreeCellRenderer m_pathFailedRenderer;

   mutable FWTextTreeCellRenderer m_modulePassedRenderer;
   mutable FWTextTreeCellRenderer m_moduleFailedRenderer;
};

FWPathsPopup::FWPathsPopup(FWFFLooper *looper)
   : TGMainFrame(gClient->GetRoot(), 400, 600),
     m_info(0),
     m_looper(looper),
     m_hasChanges(false),
     m_moduleLabel(0),
     m_moduleName(0),
     m_apply(0),
     m_psTable(new FWPSetTableManager())
{
   m_psTable->indexSelected_.connect(boost::bind(&FWPathsPopup::newIndexSelected,this,_1,_2));

   FWDialogBuilder builder(this);
   builder.indent(4)
          .spaceDown(10)
          .addLabel("Filter:").floatLeft(4).expand(false, false)
          .addTextEntry("", &m_search).expand(true, false)
          .spaceDown(10)
          .addTable(m_psTable, &m_tableWidget).expand(true, true)
          .addTextButton("Apply changes and reload", &m_apply);

   TGTextEntry *editor = new TGTextEntry(m_tableWidget, "");
   m_psTable->setCellValueEditor(editor);

   m_apply->Connect("Clicked()", "FWPathsPopup", this, "scheduleReloadEvent()");
   m_apply->SetEnabled(true);
   m_search->SetEnabled(true);
   m_search->Connect("TextChanged(const char *)", "FWPathsPopup",
                     this, "updateFilterString(const char *)");
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);
   m_tableWidget->Connect("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",
                          "FWPathsPopup",this,
                          "cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)");
   MapSubwindows();
   editor->UnmapWindow();
   Layout();
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
      m_psTable->applyEditor();
      m_psTable->setSelection(iRow, iColumn, iKeyMod);
   }   
   else if (iColumn == 1)
   {
      // Clear text on new row click
      int index = m_psTable->rowToIndex()[iRow];
      PSetData& data = m_psTable->data()[index];
      std::cerr << data.label << "clicked" << std::endl;
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

/** Gets called by CMSSW as we process events. **/
void
FWPathsPopup::postModule(edm::ModuleDescription const& description)
{
   gSystem->ProcessEvents();
}

void
FWPathsPopup::postProcessEvent(edm::Event const& event, edm::EventSetup const& eventSetup)
{
   gSystem->ProcessEvents();

   // Get the last process name from the process history:
   // this should be the one specified in the cfg file
   edm::ProcessHistory::const_iterator pi = event.processHistory().end() - 1;
   std::string processName = pi->processName();
   
   // It's called TriggerResults but actually contains info on all paths
   edm::InputTag tag("TriggerResults", "", processName);
   edm::Handle<edm::TriggerResults> triggerResults;
   event.getByLabel(tag, triggerResults);

   // The string is the path name, the bool is whether it passed,
   // and the int is the index of the module that made the decision
   std::map<std::string, PathData> pathStatus;

   if ( triggerResults.isValid() )
   {
     edm::TriggerNames triggerNames = event.triggerNames(*triggerResults);
     
     for ( size_t i = 0, ie = triggerResults->size(); i != ie; ++i )
     {
       PathData pd(triggerResults->accept(i), triggerResults->index(i));
       pathStatus.insert(std::pair<std::string, PathData>(triggerNames.triggerName(i), pd));

       std::vector<std::string> pathModules;
       m_info->modulesInPath(triggerNames.triggerName(i), pathModules);
     }
   }

   m_psTable->update(m_info, pathStatus);
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
   /*
   PythonProcessDesc desc;
   std::string pythonSnippet("import FWCore.ParameterSet.Config as cms\n"
                             "process=cms.Process('Dummy')\n");
   for (size_t li = 0, le = text->RowCount(); li != le; ++li)
   {
      char *buf = text->GetLine(TGLongPosition(0, li), text->GetLineLength(li));
      if (!buf)
         continue;
      pythonSnippet += buf;
      free(buf);
   }

   try
   {
      PythonProcessDesc pydesc(pythonSnippet);
      boost::shared_ptr<edm::ProcessDesc> desc = pydesc.processDesc();
      boost::shared_ptr<edm::ParameterSet> ps = desc->getProcessPSet();
      const edm::ParameterSet::table &pst = ps->tbl();
      const edm::ParameterSet::table::const_iterator &mi= pst.find("@all_modules");
      if (mi == pst.end())
         throw cms::Exception("cmsShow") << "@all_modules not found";
      // FIXME: we are actually interested in "@all_modules" entry.
      std::vector<std::string> modulesInConfig(mi->second.getVString());
      std::vector<std::string> parameterNames;

      for (size_t mni = 0, mne = modulesInConfig.size(); mni != mne; ++mni)
      {
         const std::string &moduleName = modulesInConfig[mni];
         std::cout << moduleName << std::endl;
         const edm::ParameterSet *modulePSet(ps->getPSetForUpdate(moduleName));
         parameterNames = modulePSet->getParameterNames();
         for (size_t pi = 0, pe = parameterNames.size(); pi != pe; ++pi)
            std::cout << "  " << parameterNames[pi] << std::endl;
         m_looper->requestChanges(moduleName, *modulePSet);
      }
      m_hasChanges = true;
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
   */
}
