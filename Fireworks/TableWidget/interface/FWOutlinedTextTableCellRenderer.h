#ifndef Fireworks_TableWidget_FWOutlinedTextTableCellRenderer_h
#define Fireworks_TableWidget_FWOutlinedTextTableCellRenderer_h
// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWOutlinedTextTableCellRenderer
// 
/**\class FWOutlinedTextTableCellRenderer FWOutlinedTextTableCellRenderer.h Fireworks/TableWidget/interface/FWOutlinedTextTableCellRenderer.h

 Description: A Cell Renderer who draws text with an outline and fills in the background

 Usage:
    The background color of the text graphics context will be used to draw an outline around the regular text
 and the color of the regular text is set by the foreground color of the text graphics context.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:43:50 EST 2009
// $Id: FWOutlinedTextTableCellRenderer.h,v 1.3 2009/05/05 08:36:06 elmer Exp $
//

// system include files
#include <string>
#include "GuiTypes.h"
#include "TGResourcePool.h"
#include "TGGC.h"

// user include files
#include "Fireworks/TableWidget/interface/FWTableCellRendererBase.h"

// forward declarations

class FWOutlinedTextTableCellRenderer : public FWTableCellRendererBase {
   
public:
   static const TGGC&  getDefaultGC();
   static const TGGC&  getFillGC();  
   
   enum Justify {
      kJustifyLeft,
      kJustifyRight,
      kJustifyCenter
   };
   
   FWOutlinedTextTableCellRenderer(const TGGC* iTextContext=&(getDefaultGC()), 
                           const TGGC* iFillContext=&(getFillGC()),
                           Justify iJustify=kJustifyLeft);
   virtual ~FWOutlinedTextTableCellRenderer();
   
   // ---------- const member functions ---------------------
   const TGGC* graphicsContext() const { return m_context;}
   virtual UInt_t width() const;
   virtual UInt_t height() const;
   
   const TGFont* font() const;
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void setData(const std::string&);
   void setGraphicsContext(const TGGC* iContext) { m_context = iContext;}
   void setJustify(Justify);
   
   virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight);
   
   
private:
   FWOutlinedTextTableCellRenderer(const FWOutlinedTextTableCellRenderer&); // stop default
   
   const FWOutlinedTextTableCellRenderer& operator=(const FWOutlinedTextTableCellRenderer&); // stop default
   
   // ---------- member data --------------------------------
   const TGGC* m_context;
   const TGGC* m_fillContext;
   TGFont* m_font;
   std::string m_data;
   Justify m_justify;
};


#endif
