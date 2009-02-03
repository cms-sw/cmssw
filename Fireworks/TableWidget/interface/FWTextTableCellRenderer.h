#ifndef Fireworks_TableWidget_FWTextTableCellRenderer_h
#define Fireworks_TableWidget_FWTextTableCellRenderer_h
// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWTextTableCellRenderer
// 
/**\class FWTextTableCellRenderer FWTextTableCellRenderer.h Fireworks/TableWidget/interface/FWTextTableCellRenderer.h

 Description: A Cell Renderer who draws text and can show selection of a cell

 Usage:
    Use when the cells of a table are simple text and you want to be able to also show that a cell has been selected.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:43:50 EST 2009
// $Id$
//

// system include files
#include <string>
#include "GuiTypes.h"
#include "TGResourcePool.h"
#include "TGGC.h"

// user include files
#include "Fireworks/TableWidget/interface/FWTableCellRendererBase.h"

// forward declarations

class FWTextTableCellRenderer : public FWTableCellRendererBase
{

   public:
      static FontStruct_t getDefaultFontStruct();
      static const TGGC&  getDefaultGC();
      static const TGGC&  getHighlightGC();   

      FWTextTableCellRenderer(GContext_t iContext=getDefaultGC()(),FontStruct_t iFontStruct = getDefaultFontStruct());
      virtual ~FWTextTableCellRenderer();

      // ---------- const member functions ---------------------
      GContext_t graphicsContext() const { return m_context;}
      virtual UInt_t width() const;
      virtual UInt_t height() const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setData(const std::string&, bool isSelected);
      void setGraphicsContext(GContext_t iContext) { m_context = iContext;}

      virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight);


   private:
      FWTextTableCellRenderer(const FWTextTableCellRenderer&); // stop default

      const FWTextTableCellRenderer& operator=(const FWTextTableCellRenderer&); // stop default

      // ---------- member data --------------------------------
      GContext_t m_context;
      FontStruct_t m_fontStruct;
      TGFont* m_font;
      std::string m_data;
      bool m_isSelected;

};


#endif
