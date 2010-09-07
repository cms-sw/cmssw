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

#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "TGLabel.h"
#include "TGTextEdit.h"
#include "TGText.h"
#include "TSystem.h"
#include "TGHtml.h"
#include "TGFont.h"

#include <iostream>
#include <sstream>
#include <cstring>

class FWTextTreeCellRenderer : public FWTextTableCellRenderer
{
public:
   FWTextTreeCellRenderer(const TGGC* iContext = &(getDefaultGC()),
                          const TGGC* iHighlightContext = &(getHighlightGC()),
                          Justify iJustify = kJustifyLeft)
      : FWTextTableCellRenderer(iContext, iHighlightContext, iJustify),
        m_indentation(0)
      {}

   virtual void setIndentation(int indentation = 0) { m_indentation = indentation; }
   virtual UInt_t width() { return FWTextTableCellRenderer::width() + m_indentation * 2; }
   virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
      {
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
         gVirtualX->DrawString(iID, graphicsContext()->GetGC(), iX+m_indentation, iY+metrics.fAscent, data().c_str(),data().size());
      }
private:
   int   m_indentation;
};

/** Attempt to create a table based editor for the PSET. */
class FWPSetTableManager : public FWTableManagerBase 
{
public:
   /** Custom structure for holding the table contents */
   struct PSetData
   {
      std::string label;
      std::string value;
      int         level;
      bool        tracked;
      std::string type;
      size_t      parent;
   };

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


   const TGGC&
   boldGC()
   {
      static TGGC s_boldGC(*gClient->GetResourcePool()->GetFrameGC());
 
      TGFontPool *pool = gClient->GetFontPool();
      TGFont *font = pool->FindFontByHandle(s_boldGC.GetFont());
      FontAttributes_t attributes = font->GetFontAttributes();

      attributes.fWeight = 1; // This doesn't seem to work
      //TGFont *newFont = pool->GetFont(attributes.fFamily, 9,
      //                                attributes.fWeight, attributes.fSlant);

      // But this does:
      TGFont* newFont = pool->GetFont("-*-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
      
      if ( ! newFont )
        return s_boldGC;

      std::cout<<"boldGC: "<< std::endl;
      newFont->Print();

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

      std::cout<<"italicGC: "<< std::endl;
      newFont->Print();

      s_italicGC.SetFont(newFont->GetFontHandle());

      return s_italicGC;
   }

   const TGGC &
   pathGC()
   {
      static TGGC s_pathGC(boldGC());
      s_pathGC.SetForeground(gVirtualX->GetPixel(kRed-5));
      return s_pathGC;
   }
   
   const TGGC &
   pathBackgroundGC()
   {
      static TGGC s_pathBackgroundGC(*gClient->GetResourcePool()->GetFrameGC());
      s_pathBackgroundGC.SetBackground(gVirtualX->GetPixel(kGray));
      return s_pathBackgroundGC;
   }

  const TGGC&
  redGC()
    {
      static TGGC s_redGC(*gClient->GetResourcePool()->GetFrameGC());
      s_redGC.SetForeground(gVirtualX->GetPixel(kRed));
      return s_redGC;
    }
 
 
  const TGGC&
  greenGC()
    {
      static TGGC s_greenGC(*gClient->GetResourcePool()->GetFrameGC());
      s_greenGC.SetForeground(gVirtualX->GetPixel(kGreen));
      return s_greenGC;
    }
  

   FWPSetTableManager()
      : m_selectedRow(-1)
   {
      m_boldRenderer.setGraphicsContext(&boldGC());
      m_italicRenderer.setGraphicsContext(&italicGC());
      m_pathRenderer.setGraphicsContext(&pathGC());
      m_pathRenderer.setHighlightContext(&pathBackgroundGC());

      std::cout << "Available fonts: " << std::endl;
      gClient->GetFontPool()->Print();
        
      reset();
   }
   
   void update(const edm::ScheduleInfo *info)
   {
      m_entries.clear();
      std::vector<std::string> availablePaths;
      info->availablePaths(availablePaths);
      for (size_t i = 0, e = availablePaths.size(); i != e; ++i)
      {
         PSetData pathEntry;
         pathEntry.label = availablePaths[i];
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
      reset();
   }

   virtual void implSort(int, bool)
   {
      m_row_to_index.clear();
      m_matches.resize(m_entries.size());

      // First of all show all the elements that match,
      // and all their children.
      for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      {
         PSetData &data = m_entries[i];
         if (data.parent != ((size_t) -1) && m_matches[data.parent])
         {
            m_matches[i] = true;
            continue;
         } 

         if (strstr(data.label.c_str(), m_filter.c_str()))
            m_matches[i] = true;
         else
            m_matches[i] = false;
      }

      // Show also parents
      std::vector<int> stack;
      int previousLevel = 0;
      for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      {
         PSetData &data = m_entries[i];
         // Top level.
         if (data.parent == (size_t)-1)
            continue;

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
         
         if (m_matches[i])
            for (size_t pi = 0, pe = stack.size(); pi != pe; ++pi)
               m_matches[stack[pi]] = true;

         previousLevel = data.level;
      }  

      for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      {
         if (m_matches[i])
            m_row_to_index.push_back(i);
      }
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
      returnValue.reserve(2);
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

      if ( data.level == 0 )
      {
         label = data.label + "(" + data.value + ")";
         value = "";
         renderer = &m_pathRenderer;
      }
      else if (data.level == 1)
      {
         label = data.label + "(" + data.value + ")";
         value = "";
         renderer = &m_italicRenderer;
      }
      else 
      {
         value = data.value;
         label = data.label;
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

      return renderer;
   }

   void setSelection (int row, int mask) {
      if(mask == 4) {
         if( row == m_selectedRow) {
            row = -1;
         }
      }
      changeSelection(row);
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
      changeSelection(-1);
      m_row_to_index.clear();
      m_row_to_index.reserve(m_entries.size());
      for(size_t i = 0, e = m_entries.size(); i != e; ++i)
         m_row_to_index.push_back(i);
      
      dataChanged();
   }

   void handlePSetEntry(const edm::ParameterSetEntry& entry, const std::string& key)
   {
      PSetData data;
      data.label = key;
      data.tracked = entry.isTracked();
      data.level = m_parentStack.size();
      data.parent = m_parentStack.back();
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
      case 'F':
        {
          entry.getFileInPath().write(ss);
          createVectorString(data, ss.str(), true);
          break;
        }
      case 'e':
        {
          std::vector<edm::MinimalEventID> ids;
          ids.resize(entry.getVEventID().size());
          for ( size_t iri = 0, ire = ids.size(); iri != ire; ++iri )
            ids[iri] = entry.getVEventID()[iri];
          createVectorString(data, ids, true);
          break;
        }
      case 'E':
        {
          createScalarString(data, entry.getEventID());
          break;
        }
      case 'm':
        {
          std::vector<edm::LuminosityBlockID> ids;
          ids.resize(entry.getVLuminosityBlockID().size());
          for ( size_t iri = 0, ire = ids.size(); iri != ire; ++iri )
            ids[iri] = entry.getVLuminosityBlockID()[iri];
          createVectorString(data, ids, true);
          break;
        }
      case 'M':
        {
          createScalarString(data, entry.getLuminosityBlockID());
          break;
        }
      case 'a':
        {
          std::vector<edm::LuminosityBlockRange> ranges;
          ranges.resize(entry.getVLuminosityBlockRange().size());
          for ( size_t iri = 0, ire = ranges.size(); iri != ire; ++iri )
            ranges[iri] = entry.getVLuminosityBlockRange()[iri];
          createVectorString(data, ranges, true);
          break;
        }
      case 'A':
        {
          createScalarString(data, entry.getLuminosityBlockRange());
          break;
        }
      case 'r':
        {
          std::vector<edm::EventRange> ranges;
          ranges.resize(entry.getVEventRange().size());
          for ( size_t iri = 0, ire = ranges.size(); iri != ire; ++iri )
            ranges[iri] = entry.getVEventRange()[iri];
          createVectorString(data, ranges, true);
          break;
        }
      case 'R':
        {
          createScalarString(data, entry.getEventRange());
          break;          
        }
      default:
        {
          break;
        }
      }
   }

   std::vector<PSetData> &data() { return m_entries; }
    
   sigc::signal<void,int> indexSelected_;
private:
   void changeSelection(int iRow)
   {
      if(iRow != m_selectedRow) {
         m_selectedRow=iRow;
         if(-1 == iRow) {
            indexSelected_(-1);
         } else {
            indexSelected_(iRow);
         }
         visualPropertiesChanged();
      }
   }

   std::vector<PSetData>           m_entries;
   std::vector<size_t>             m_parentStack;
   std::vector<int>                m_row_to_index;
   std::vector<bool>               m_matches;
   int                             m_selectedRow;
   std::string                     m_filter;

   mutable FWTextTreeCellRenderer m_renderer;  
   mutable FWTextTreeCellRenderer m_pathRenderer;
   mutable FWTextTreeCellRenderer m_boldRenderer;
   mutable FWTextTreeCellRenderer m_italicRenderer;
};

FWPathsPopup::FWPathsPopup(FWFFLooper *looper)
   : TGMainFrame(gClient->GetRoot(), 400, 600),
     m_info(0),
     m_looper(looper),
     m_hasChanges(false),
     m_moduleLabel(0),
     m_moduleName(0),
     m_modulePathsHtml(0),
     m_textEdit(0),
     m_apply(0),
     m_psTable(new FWPSetTableManager())
{
   FWDialogBuilder builder(this);
   builder.indent(4)
          .addLabel("Available paths", 10)
          .spaceDown(10)
          .addLabel("Filter:").floatLeft(4).expand(false, false)
          .addTextEntry("", &m_search).expand(true, false)
          .spaceDown(10)
          .addTable(m_psTable, &m_tableWidget).expand(true, true)
          .addTextEdit("", &m_textEdit)
          .addTextButton("Apply changes and reload", &m_apply);

   m_apply->Connect("Clicked()", "FWPathsPopup", this, "scheduleReloadEvent()");
   m_apply->SetEnabled(true);
   m_search->SetEnabled(true);
   m_search->Connect("TextChanged(const char *)", "FWPathsPopup",
                     this, "updateFilterString(const char *)");


   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);

   MapSubwindows();
   Layout();
}

void
FWPathsPopup::updateFilterString(const char *str)
{
   m_psTable->sortWithFilter(str);
}

/** Finish the setup of the GUI */
void
FWPathsPopup::setup(const edm::ScheduleInfo *info)
{
   assert(info);
   m_info = info;

   m_info->availableModuleLabels(m_availableModuleLabels);
   m_info->availablePaths(m_availablePaths);

   // makePathsView();
}

// It would be nice if we could use some of the 
// utilities from Entry. 
// Why couldn't the type just be the type?
const char*
FWPathsPopup::typeCodeToChar(char typeCode)
{
  switch(typeCode)
  {
  case 'b':  return "Bool";
  case 'B':  return "Bool";
  case 'i' : return "vint32";
  case 'I' : return "int32";
  case 'u' : return "vuint32";
  case 'U' : return "uint32";
  case 'l' : return "vint64";
  case 'L' : return "int64";
  case 'x' : return "vuint64";
  case 'X' : return "uint64";
  case 's' : return "vstring";
  case 'S' : return "string";
  case 'd' : return "vdouble";
  case 'D' : return "double";
  case 'p' : return "vPSet";
  case 'P' : return "PSet";
  case 'T' : return "path";
  case 'F' : return "FileInPath";
  case 't' : return "InputTag";
  case 'v' : return "VInputTag";
  case 'e' : return "VEventID";
  case 'E' : return "EventID";
  case 'm' : return "VLuminosityBlockID";
  case 'M' : return "LuminosityBlockID";
  case 'a' : return "VLuminosityBlockRange";    
  case 'A' : return "LuminosityBlockRange";
  case 'r' : return "VEventRange";
  case 'R' : return "EventRange";
  default:   return "Type not known";
  }
}

// Crikey! I'm ending up writing a ParameterSet parser here!
// I probably could use the results from the << operator
// in Entry but it's not quite what I want for format.
// Also, it's not clear to me how to break it up into html
// elements.
void
FWPathsPopup::handleEntry(const edm::Entry& entry, 
                          const std::string& key, TString& html)
{
  html += "<li>" + key + "    " 
          + (entry.isTracked() ? "tracked    " : "untracked    ")
          + typeCodeToChar(entry.typeCode());

  std::stringstream ss;
  switch(entry.typeCode())
  {
  case 'b':
    {
      ss << entry.getBool();
      html += "    " + ss.str();
      break;
    }
  case 'B':
    { 
      ss << entry.getBool();
      html += "    " + ss.str();
      break;
    }
  case 'i':
    {
      html += "    ";
      std::vector<int> ints = entry.getVInt32();
      for ( std::vector<int>::const_iterator ii = ints.begin(), iiEnd = ints.end();
            ii != iiEnd; ++ii )
      {
        ss << *ii <<"  ";
        html += ss.str();
      }
      break;
    }
  case 'I':
    {
      ss << entry.getInt32();
      html += "   " + ss.str();
      break;
    }
  case 'u':
    {
      html += "    ";
      std::vector<unsigned> us = entry.getVUInt32();
      for ( std::vector<unsigned>::const_iterator ui = us.begin(), uiEnd = us.end();
            ui != uiEnd; ++ui )
      {
        ss << *ui <<" ";
        html += ss.str();
      } 
      break;
    }
  case 'U':
    {
      ss << entry.getUInt32();
      html += "   " + ss.str();
      break;
    }
  case 'l':
    {
      html += "    ";
      std::vector<long long> ints = entry.getVInt64();
      for ( std::vector<long long>::const_iterator ii = ints.begin(), iiEnd = ints.end();
            ii != iiEnd; ++ii )
      {
        ss << *ii << " ";
        html += ss.str();
      }
      break;
    }
  case 'L':
    {
      ss << entry.getInt64();
      html += "   " + ss.str();
      break;
    }
  case 'x':
    {
      html += "    ";
      // This the 1st time in my life I have written "unsigned long long"! Exciting.
      std::vector<unsigned long long> us = entry.getVUInt64();
      for ( std::vector<unsigned long long>::const_iterator ui = us.begin(), uiEnd = us.end();
            ui != uiEnd; ++ui )
      {
        ss << *ui <<" ";
        html += ss.str();
      }
      break;
    }
  case 'X':
    {
      ss << entry.getUInt64();
      html += "    " + ss.str();
      break;
    }
  case 's':
    {
      std::vector<std::string> strs = entry.getVString();
      html += "    ";
      for ( std::vector<std::string>::const_iterator si = strs.begin(), siEnd = strs.end();
            si != siEnd; ++si )
      {
        html += *si + " ";
      }
      break;
    }
  case 'S':
    {
      html += "    " + entry.getString();
      break;
    }
  case 'd':
    {
      html += "    ";
      std::vector<double> ds = entry.getVDouble();
      for ( std::vector<double>::const_iterator di = ds.begin(), diEnd = ds.end();
            di != diEnd; ++di )
      {
        ss << *di <<" ";
        html += ss.str();
      }
      break;
    }
  case 'D':
    {   
      ss << entry.getDouble();
      html += "    " + ss.str();
      break;
    }
  case 'p':
    {
      std::vector<edm::ParameterSet> psets = entry.getVPSet();
      html += "    ";
      for ( std::vector<edm::ParameterSet>::const_iterator psi = psets.begin(), psiEnd = psets.end();
            psi != psiEnd; ++psi )
        handlePSet(&(*psi), html);
      break;
    }
  case 'P':
    {    
      edm::ParameterSet psets = entry.getPSet();
      handlePSet(&psets, html);
      break;
    }
  case 'v':
    {
      html += "    ";
      std::vector<edm::InputTag> tags = entry.getVInputTag();
      for ( std::vector<edm::InputTag>::const_iterator ti = tags.begin(), tiEnd = tags.end();
            ti != tiEnd; ++ti )
      {
        ss << ti->encode() <<" ";
        html += ss.str();
      }
      break;
    }
  case 't':
    {
      ss << entry.getInputTag();
      html += "    " + ss.str();
      break;
    }
  case 'e':
    {
      html += "    ";
      std::vector<edm::MinimalEventID> ids = entry.getVEventID();
      for ( size_t iri = 0, ire = ids.size(); iri != ire; ++iri )
      {
        ss << ids[iri];
        html += " " + ss.str();
      }
      break;
    }
  case 'E':
    {
      ss << entry.getEventID();
      html += "   " + ss.str();
      break;
    }
  case 'm':
    {
      html += "    ";
      std::vector<edm::LuminosityBlockID> ids = entry.getVLuminosityBlockID();
      for ( size_t iri = 0, ire = ids.size(); iri != ire; ++iri )
      {
        ss << ids[iri];
        html += " " + ss.str();
      }
      break;
    }      
  case 'M':
    {
      ss << entry.getLuminosityBlockID();
      html += "   " + ss.str();
      break;
    }      
  case 'a':
    {
      html += "   ";
      std::vector<edm::LuminosityBlockRange> ranges = entry.getVLuminosityBlockRange();
      for ( size_t iri = 0, ire = ranges.size(); iri != ire; ++iri )
      {
        ss << ranges[iri];
        html += " " + ss.str();
      }
      break;
    }    
  case 'A':
    {
      ss << entry.getLuminosityBlockRange();
      html += "   " + ss.str();
      break;
    }    
  case 'r':
    {
      html += "   ";
      std::vector<edm::EventRange> ranges = entry.getVEventRange();
      for ( size_t iri = 0, ire = ranges.size(); iri != ire; ++iri )
      {
        ss << ranges[iri];
        html += " " + ss.str();
      }
      break;
    }
  case 'R':
    {
      ss << entry.getEventRange();
      html += "   " + ss.str();
      break;          
    }
  case 'F':
    {
      entry.getFileInPath().write(ss);
      html += "   " + ss.str();
      break;
    }
  default:
    {
      html += "   [Not supported yet. Are you sure you want this?]";
      break;
    }
  }
}

void 
FWPathsPopup::handlePSetEntry(const edm::ParameterSetEntry& entry, 
                              const std::string& key, TString& html)
{
  html += "<li>" + key + "    " 
          + (entry.isTracked() ? "tracked    " : "untracked    ")
          + "PSet";

  handlePSet(&(entry.pset()), html);
}

void 
FWPathsPopup::handleVPSetEntry(const edm::VParameterSetEntry& entry, 
                               const std::string& key, TString& html)
{
  html += "<li>" + key + "    " 
          + (entry.isTracked() ? "tracked    " : "untracked    ")
          + "vPSet";

  for ( std::vector<edm::ParameterSet>::const_iterator psi = entry.vpset().begin(),
                                                    psiEnd = entry.vpset().end();
        psi != psiEnd; ++psi )
  {
    handlePSet(&(*psi), html);
  }
}       

void 
FWPathsPopup::handlePSet(const edm::ParameterSet* ps, TString& html)
{
  html += "<ul>";

  for ( edm::ParameterSet::table::const_iterator ti = 
          ps->tbl().begin(), tiEnd = ps->tbl().end();
        ti != tiEnd; ++ti )
    handleEntry(ti->second, ti->first, html);
    
  for ( edm::ParameterSet::psettable::const_iterator pi = 
          ps->psetTable().begin(), piEnd = ps->psetTable().end();
        pi != piEnd; ++pi )
    handlePSetEntry(pi->second, pi->first, html);

  for ( edm::ParameterSet::vpsettable::const_iterator vpi = 
          ps->vpsetTable().begin(), vpiEnd = ps->vpsetTable().end();
        vpi != vpiEnd; ++vpi )
    handleVPSetEntry(vpi->second, vpi->first, html);

  html += "</ul>";
}

//#include <fstream>
//std::ofstream fout("path-view.html");

void
FWPathsPopup::makePathsView()
{
  m_modulePathsHtml->Clear();

  TString html;

  html = "<html><head><title>Available paths</title></head><body>";

  for ( std::vector<std::string>::iterator pi = m_availablePaths.begin(),
                                        piEnd = m_availablePaths.end();
        pi != piEnd; ++pi )
  {
    html += "<h1>"+ *pi + "</h1>";

    std::vector<std::string> modulesInPath;
    m_info->modulesInPath(*pi, modulesInPath);

    for ( std::vector<std::string>::iterator mi = modulesInPath.begin(),
                                          miEnd = modulesInPath.end();
          mi != miEnd; ++mi )
    {
      const edm::ParameterSet* ps = m_info->parametersForModule(*mi);

      // Need to get the module type from the parameter set before we handle the set itself
      const edm::ParameterSet::table& pst = ps->tbl();    
      const edm::ParameterSet::table::const_iterator ti = pst.find("@module_edm_type");
      if (ti == pst.end())
         html += "<h2>Unknown module type: " + *mi + "</h2>";
      else
         html += "<h2>" + ti->second.getString() + "  " + *mi  + "</h2>";
      handlePSet(ps, html); 
    }
  } 

  html += "</body></html>";
  //fout<< html <<std::endl;

  m_modulePathsHtml->ParseText((char*)html.Data());
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
   m_psTable->update(m_info);
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
   PythonProcessDesc desc;
   std::string pythonSnippet("import FWCore.ParameterSet.Config as cms\n"
                             "process=cms.Process('Dummy')\n");
   TGText *text = m_textEdit->GetText();
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
}
