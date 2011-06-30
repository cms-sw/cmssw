#ifndef Fireworks_Core_FWGeometryTableManager_h
#define Fireworks_Core_FWGeometryTableManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableManager
// 
/**\class FWGeometryTableManager FWGeometryTableManager.h Fireworks/Core/interface/FWGeometryTableManager.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel, Matevz Tadel
//         Created:  Thu Jan 27 14:50:40 CET 2011
// $Id: FWGeometryTableManager.h,v 1.12 2011/06/30 04:53:31 amraktad Exp $
//

#include <sigc++/sigc++.h>
#include <boost/tr1/unordered_map.hpp>

#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTextTreeCellRenderer.h"

#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/TableWidget/interface/FWTableCellRendererBase.h"

#include "TGeoNode.h"

class FWTableCellRendererBase;
class FWGeometryBrowser;
class TGeoManager;
class TGeoNode;

class FWGeometryTableManager : public FWTableManagerBase
{
   friend class FWGeometryBrowser;

public:
   enum   ECol { kName, kColor,  kVisSelf, kVisChild, kMaterial, kPosition, kBBoxSize, kNumCol };

   struct NodeInfo
   {
      NodeInfo():m_node(0), m_parent(-1), m_level(-1), 
                 m_visible(false), m_expanded(false)
      {}  

      TGeoNode*   m_node;
      Int_t       m_parent;
      Short_t     m_level;

      Bool_t      m_visible;
      Bool_t      m_expanded;

      // 3D
      Color_t     m_color;

      const char* name() const;
   };

   typedef std::vector<NodeInfo> Entries_v;
   typedef Entries_v::iterator Entries_i;
   
private:
   struct Match
   {
      bool m_matches;
      bool m_childMatches;
      
      Match() : m_matches(true), m_childMatches(true) {}
   
      bool accepted() { return m_matches || m_childMatches; }
   };

   
   // AMT: this could be a common base class with FWCollectionSummaryModelCellRenderer ..
   class ColorBoxRenderer : public FWTableCellRendererBase
   { 
   public:
      ColorBoxRenderer();
      virtual ~ColorBoxRenderer();
  
      virtual UInt_t width() const { return m_width; }
      virtual UInt_t height() const { return m_height; }
      void setData(Color_t c, bool);
      virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight);

      UInt_t  m_width;
      UInt_t  m_height;
      Pixel_t m_color;      
      bool    m_isSelected;
      TGGC*   m_colorContext;

   };

public:
   FWGeometryTableManager(FWGeometryBrowser*);
   virtual ~FWGeometryTableManager();

   // virtual functions of FWTableManagerBase
   
   virtual int unsortedRowNumber(int unsorted) const;
   virtual int numberOfRows() const;
   virtual int numberOfColumns() const;
   virtual std::vector<std::string> getTitles() const;
   virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const;

   virtual const std::string title() const;

   int selectedRow() const;
   int selectedColumn() const;
   virtual bool rowIsSelected(int row) const;

   std::vector<int> rowToIndex() { return m_row_to_index; }

   void setSelection(int row, int column, int mask); 
   virtual void implSort(int, bool) {}

   void printChildren(int) const;
   bool nodeImported(int idx) const;
   int    getAutoExpandEntry();
   // geo stuff
   NodeInfo& refSelected();
   Entries_v& refEntries() {return m_entries;}

   void loadGeometry();
   void setBackgroundToWhite(bool);
   void getNodePath(int, std::string&);

   int getTopGeoNodeIdx() const { return m_geoTopNodeIdx; }
   int getLevelOffset() const { return m_levelOffset; }

private:
   FWGeometryTableManager(const FWGeometryTableManager&); // stop default
   const FWGeometryTableManager& operator=(const FWGeometryTableManager&); // stop default

   
   void firstColumnClicked(int row);

   // table mng
   void changeSelection(int iRow, int iColumn);
   void redrawTable();
   void recalculateVisibility();
   
   // geo
   void checkUniqueVolume(TGeoVolume* v); 
   void checkChildMatches(TGeoVolume* v,  std::vector<TGeoVolume*>&);
   int  getNdaughtersLimited(TGeoNode*) const;
   void getNNodesTotal(TGeoNode* geoNode, int level,int& off, bool debug) const;
   void importChildren(int parent_idx, bool recurse);
   void checkHierarchy();


   // signal callbacks
   void updateFilter();
   void checkImportLevel();
   void topGeoNodeChanged(int);


   // ---------- member data --------------------------------
   
   typedef boost::unordered_map<TGeoVolume*, Match>  Volumes_t;
   typedef Volumes_t::iterator               Volumes_i;
   
   // table stuff
   mutable FWTextTreeCellRenderer m_renderer;  
   mutable ColorBoxRenderer       m_colorBoxRenderer;  

public:
   std::vector<int>  m_row_to_index;
   int               m_selectedRow;
   int               m_selectedIdx;
   int               m_selectedColumn;
   
   // geo stuff
   FWGeometryBrowser*   m_browser;
      
   mutable Volumes_t  m_volumes;
   Entries_v          m_entries;

   bool               m_filterOff; //cached
   int m_topGeoNodeIdx; 
   int m_levelOffset;
   int  m_geoTopNodeIdx;
};

#endif

inline int FWGeometryTableManager::getNdaughtersLimited(TGeoNode* geoNode) const
{
   // used for debugging of table and 3D view

   // return TMath::Min(geoNode->GetNdaughters(), m_browser->getMaxDaughters());
  return  geoNode->GetNdaughters();
}
