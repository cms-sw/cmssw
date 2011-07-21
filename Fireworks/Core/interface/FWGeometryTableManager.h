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
// $Id: FWGeometryTableManager.h,v 1.30 2011/07/20 22:11:49 amraktad Exp $
//

#include <sigc++/sigc++.h>
#include <boost/tr1/unordered_map.hpp>

#include "Fireworks/Core/interface/FWGeometryTableView.h"

#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTextTreeCellRenderer.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/TableWidget/interface/FWTableCellRendererBase.h"

#include "TGeoNode.h"
#include "TGeoVolume.h"

class FWTableCellRendererBase;
// class FWGeometryTableView;
//class TGeoManager;
class TGeoNode;

class FWGeometryTableManager : public FWTableManagerBase
{
   friend class FWGeometryTableView;

public:
   enum   ECol { kName, kColor,  kVisSelf, kVisChild, kMaterial, kPosX, kPosY, kPosZ /*, kDiagonal*/, kNumCol };

   enum Bits
   {
      kExpanded        =  BIT(0),
      kMatches         =  BIT(1),
      kChildMatches    =  BIT(2),
      kFilterCached    =  BIT(3),

      kVisNode         =  BIT(4),
      kVisNodeChld     =  BIT(5)
      //   kVisVol          =  BIT(6),
      //   kVisVolChld      =  BIT(7),

   };

   struct NodeInfo
   {
      NodeInfo():m_node(0), m_parent(-1), m_color(0), m_level(-1), 
                 m_flags(kVisNode|kVisNodeChld)
      {}  

      TGeoNode*   m_node;
      Int_t       m_parent;
      Color_t     m_color;
      UChar_t     m_level;
      UChar_t     m_flags;

      const char* name() const;
      const char* nameIndent() const;

      void setBit(UChar_t f)    { m_flags  |= f;}
      void resetBit(UChar_t f)  { m_flags &= ~f; }
      void setBitVal(UChar_t f, bool x) { x ? setBit(f) : resetBit(f);}
 
      bool testBit(UChar_t f) const  { return (m_flags & f) == f; }
      bool testBitAny(UChar_t f) const  { return (m_flags & f) != 0; }

      void switchBit(UChar_t f) { testBit(f) ? resetBit(f) : setBit(f); }
   };

   struct Match
   {
      bool m_matches;
      bool m_childMatches;
      Match() : m_matches(false), m_childMatches(false) {}
   
      bool accepted() { return m_matches || m_childMatches; }
   };

   typedef std::vector<NodeInfo> Entries_v;
   typedef Entries_v::iterator Entries_i;
   
   typedef boost::unordered_map<TGeoVolume*, Match>  Volumes_t;
   typedef Volumes_t::iterator               Volumes_i; 

private: 
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
   FWGeometryTableManager(FWGeometryTableView*);
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
   // geo stuff
   Entries_i refSelected();
   Entries_v& refEntries() {return m_entries;}

   void loadGeometry( TGeoNode* , TObjArray*);
   void setBackgroundToWhite(bool);
   void getNodePath(int, std::string&) const;

   int getLevelOffset() const { return m_levelOffset; }

   void assertNodeFilterCache(NodeInfo& data);

   void setDaughtersSelfVisibility(bool);

   void getNodeMatrix(const NodeInfo& nodeInfo, TGeoHMatrix& mat) const;

   void setVisibility(NodeInfo& nodeInfo, bool );
   void setVisibilityChld(NodeInfo& nodeInfo, bool);

   bool getVisibilityChld(const NodeInfo& nodeInfo) const;
   bool getVisibility (const NodeInfo& nodeInfo) const;

   static  void getNNodesTotal(TGeoNode* geoNode, int& off);

private:
   FWGeometryTableManager(const FWGeometryTableManager&); // stop default
   const FWGeometryTableManager& operator=(const FWGeometryTableManager&); // stop default

   
   void firstColumnClicked(int row);

   // table mng
   void changeSelection(int iRow, int iColumn);
   void redrawTable();

   void recalculateVisibility();
   void recalculateVisibilityNodeRec(int);
   void recalculateVisibilityVolumeRec(int);
   
   // geo
   void checkChildMatches(TGeoVolume* v,  std::vector<TGeoVolume*>&);
   void importChildren(int parent_idx);
   void checkHierarchy();


   // signal callbacks
   void updateFilter();
   void checkExpandLevel();
   void topGeoNodeChanged(int);
   void printMaterials();

   //   const std::string& getStatusMessage() const { return m_statusMessage; }
   // ---------- member data --------------------------------
   
   
   // table stuff
   mutable FWTextTreeCellRenderer m_renderer;  
   mutable ColorBoxRenderer       m_colorBoxRenderer;  

   std::vector<int>  m_row_to_index;
   int               m_selectedRow;
   int               m_selectedIdx;
   int               m_selectedColumn;
   
   // geo stuff
   FWGeometryTableView*   m_browser;
      
   mutable Volumes_t  m_volumes;
   Entries_v          m_entries;

   bool               m_filterOff; //cached
   int                m_numVolumesMatched; //cached

   int m_topGeoNodeIdx; 
   int m_levelOffset;
   //  int m_geoTopNodeIdx;

   //   std::string m_statusMessage;
};



inline void FWGeometryTableManager::getNNodesTotal(TGeoNode* geoNode, int& off)
{   
   int nD =  geoNode->GetNdaughters();
   off += nD;
   for (int i = 0; i < nD; ++i )
   {
      getNNodesTotal(geoNode->GetDaughter(i), off);
   }
}

#endif
