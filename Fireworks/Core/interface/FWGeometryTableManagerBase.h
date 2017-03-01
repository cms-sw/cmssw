#ifndef Fireworks_Core_FWGeometryTableManagerBase_h
#define Fireworks_Core_FWGeometryTableManagerBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableManagerBase
// 
/**\class FWGeometryTableManagerBase FWGeometryTableManagerBase.h Fireworks/Core/interface/FWGeometryTableManagerBase.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel, Matevz Tadel
//         Created:  Thu Jan 27 14:50:40 CET 2011
//

#include <sigc++/sigc++.h>

#include "Fireworks/Core/interface/FWGeometryTableViewBase.h"

#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTextTreeCellRenderer.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/TableWidget/interface/FWTableCellRendererBase.h"

#include "TGeoNode.h"

class FWTableCellRendererBase;
class TGeoNode;
class TEvePointSet;

class FWGeometryTableManagerBase : public FWTableManagerBase
{
   friend class FWGeometryTableViewBase;

public:
   //   enum   ESelectionState { kNone, kSelected, kHighlighted, kFiltered };

   enum Bits
   {
      kExpanded        =  BIT(0),

      kVisNodeSelf     =  BIT(1),
      kVisNodeChld     =  BIT(2),

      kHighlighted   =  BIT(3),
      kSelected      =  BIT(4)
   };

   struct NodeInfo
   {
      NodeInfo():m_node(0), m_parent(-1), m_color(0), m_level(-1), 
                 m_flags(kVisNodeSelf|kVisNodeChld) {}  

      NodeInfo(TGeoNode* n, Int_t p, Color_t col, Char_t l, UChar_t f = kVisNodeSelf|kVisNodeChld ):m_node(n), m_parent(p), m_color(col), m_level(l), 
                 m_flags(f) {}  

      TGeoNode*   m_node;
      Int_t       m_parent;
      Color_t     m_color;
      UChar_t     m_level;
      UChar_t     m_flags;
      UChar_t     m_transparency;


      const char* name() const;
      //  const char* nameIndent() const;

      void setBit(UChar_t f)    { m_flags  |= f;}
      void resetBit(UChar_t f)  { m_flags &= ~f; }
      void setBitVal(UChar_t f, bool x) { x ? setBit(f) : resetBit(f);}
 
      bool testBit(UChar_t f) const  { return (m_flags & f) == f; }
      bool testBitAny(UChar_t f) const  { return (m_flags & f) != 0; }

     void switchBit(UChar_t f) { testBit(f) ? resetBit(f) : setBit(f); }

     void copyColorTransparency(const NodeInfo& x) {
       m_color = x.m_color; m_transparency = x.m_transparency; 
       if (m_node->GetVolume()) { 
         m_node->GetVolume()->SetLineColor(x.m_color);
         m_node->GetVolume()->SetTransparency(x.m_transparency);
       }
     }
   };


   typedef std::vector<NodeInfo> Entries_v;
   typedef Entries_v::iterator Entries_i;
   
   int m_highlightIdx;

   //private: 
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

protected:
   virtual bool nodeIsParent(const NodeInfo&) const { return false; }
   //   virtual ESelectionState nodeSelectionState(int idx) const;

public:
   FWGeometryTableManagerBase();
   virtual ~FWGeometryTableManagerBase();
   //   virtual std::string& cellName(const NodeInfo& ) const { return &std::string("ddd");} 
   virtual const char* cellName(const NodeInfo& ) const { return 0;} 

   // virtual functions of FWTableManagerBase
   
   virtual int unsortedRowNumber(int unsorted) const;
   virtual int numberOfRows() const;
   virtual std::vector<std::string> getTitles() const;

   virtual const std::string title() const;

   //int selectedRow() const;
   //int selectedColumn() const;
   //virtual bool rowIsSelected(int row) const;

   std::vector<int> rowToIndex() { return m_row_to_index; }

   //   void setSelection(int row, int column, int mask); 
   virtual void implSort(int, bool) {}

   bool nodeImported(int idx) const;
   // geo stuff

   NodeInfo* getSelected();

   Entries_v& refEntries() {return m_entries;}
  NodeInfo& refEntry(int i) {return m_entries[i];}

   void loadGeometry( TGeoNode* , TObjArray*);

   void setBackgroundToWhite(bool);
   void getNodePath(int, std::string&) const;

   int getLevelOffset() const { return m_levelOffset; }
   void setLevelOffset(int x) { m_levelOffset =x; }

   void setDaughtersSelfVisibility(bool);

   void getNodeMatrix(const NodeInfo& nodeInfo, TGeoHMatrix& mat) const;

   
   virtual void setVisibility(NodeInfo&, bool );
   virtual void setVisibilityChld(NodeInfo&, bool);
   virtual void setDaughtersSelfVisibility(int selectedIdx, bool v);

   virtual bool getVisibilityChld(const NodeInfo& nodeInfo) const;
   virtual bool getVisibility (const NodeInfo& nodeInfo) const;

   virtual void applyColorTranspToDaughters(int selectedIdx, bool recurse);

   bool isNodeRendered(int idx, int top_node_idx) const;

   static  void getNNodesTotal(TGeoNode* geoNode, int& off);

   void showEditor(int);
   void cancelEditor(bool);
   void setCellValueEditor(TGTextEntry *editor);
   void applyTransparencyFromEditor();
   // protected:
   FWGeometryTableManagerBase(const FWGeometryTableManagerBase&); // stop default
   const FWGeometryTableManagerBase& operator=(const FWGeometryTableManagerBase&); // stop default

   
   bool firstColumnClicked(int row, int xPos);
   //   void changeSelection(int iRow, int iColumn);

   void redrawTable(bool setExpand = false);

   virtual void recalculateVisibility() = 0;

  
   virtual bool cellDataIsSortable() const { return false ; }
   // ---------- member data --------------------------------
   
   
   // table stuff
   mutable TGGC* m_highlightContext; 
   mutable FWTextTreeCellRenderer m_renderer;  
   mutable ColorBoxRenderer       m_colorBoxRenderer;  

   std::vector<int>  m_row_to_index;
   
   Entries_v          m_entries;

   int m_levelOffset;
   
   TGTextEntry* m_editor;
   int m_editTransparencyIdx;
};



inline void FWGeometryTableManagerBase::getNNodesTotal(TGeoNode* geoNode, int& off)
{   
   int nD =  geoNode->GetNdaughters();
   off += nD;
   for (int i = 0; i < nD; ++i )
   {
      getNNodesTotal(geoNode->GetDaughter(i), off);
   }
}

#endif
