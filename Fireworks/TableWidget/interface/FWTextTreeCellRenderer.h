#ifndef Fireworks_TableWidget_FWTextTreeCellRenderer_h
#define Fireworks_TableWidget_FWTextTreeCellRenderer_h

#include <cassert>
#include <iostream>

#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/TableWidget/interface/GlobalContexts.h"
#include "Fireworks/TableWidget/src/FWTabularWidget.h"

#include "TGTextEntry.h"
#include "TGPicture.h"
#include "TSystem.h"
#include "TGClient.h"

class FWTextTreeCellRenderer : public FWTextTableCellRenderer
{
protected:
   const static int  s_iconOffset  = 2;

public:

   FWTextTreeCellRenderer(const TGGC* iContext = &(getDefaultGC()),
                          const TGGC* iHighlightContext = &(getDefaultHighlightGC()),
                          Justify iJustify = kJustifyLeft)
      : FWTextTableCellRenderer(iContext, iHighlightContext, iJustify),
        m_indentation(0),
        m_editor(0),
        m_showEditor(false),
        m_isParent(false),
        m_isOpen(false),
        m_blackIcon(true)
   {}

   // Where to find the icons
   static const TString& coreIcondir() {
      static TString path = Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE"));
      if ( gSystem->AccessPathName(path.Data()) ){ // cannot find directory
         assert(gSystem->Getenv("CMSSW_RELEASE_BASE"));
         path = Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_RELEASE_BASE"));
      }

      return path;
   }

   static
   const TGPicture* closedImage(bool isBlack = true)
   {
      static const TGPicture* s_picture_white = gClient->GetPicture(coreIcondir()+"arrow-white-right-blackbg.png");
      static const TGPicture* s_picture_black = gClient->GetPicture(coreIcondir()+"arrow-black-right.png");

      return isBlack ? s_picture_black : s_picture_white;
   }

   static
   const TGPicture* openedImage(bool isBlack = true)
   {
      static const TGPicture* s_picture_white = gClient->GetPicture(coreIcondir()+"arrow-white-down-blackbg.png");
      static const TGPicture* s_picture_black = gClient->GetPicture(coreIcondir()+"arrow-black-down.png");

      return isBlack ? s_picture_black : s_picture_white;
   }


   static
   int iconWidth()
   {
      return  openedImage(true)->GetWidth() + s_iconOffset;
   }

   virtual void setIndentation(int indentation = 0) { m_indentation = indentation; }
   virtual void setCellEditor(TGTextEntry *editor) { m_editor = editor; }
   virtual void showEditor(bool value) { m_showEditor = value; }
 

   void setIsParent(bool value) {m_isParent = value; }
   void setIsOpen(bool value) {m_isOpen = value; }
   void setBlackIcon(bool value) { m_blackIcon = value; }

   virtual UInt_t width() const
   {
      int w = FWTextTableCellRenderer::width() + 15 + m_indentation;
      if (m_isParent)   w += iconWidth();
      return w;
   }

   virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
   {      
      if (m_showEditor && m_editor)
      {
         //  printf("renderer draw editor %d %d %d %d \n", iX, iY, m_editor->GetWidth(), m_editor->GetHeight());

         // fill to cover buffer offset
         static TGGC editGC(FWTextTableCellRenderer::getDefaultGC()); 
         editGC.SetForeground(m_editor->GetBackground());
         gVirtualX->FillRectangle(iID, editGC(), iX - FWTabularWidget::kTextBuffer, iY - FWTabularWidget::kTextBuffer,
                                  iWidth + 2*FWTabularWidget::kTextBuffer, iHeight + 2*FWTabularWidget::kTextBuffer);

         if ( iY > -2)
         {
            // redraw editor
            if (!m_editor->IsMapped())
            {
               m_editor->MapWindow();
               m_editor->SetFocus();
            }
            m_editor->MoveResize(iX , iY, m_editor->GetWidth(), m_editor->GetHeight());
            m_editor->SetCursorPosition( data().size());
            gClient->NeedRedraw(m_editor);
       
            return;
         }
         else
         {
            // hide editor if selected entry scrolled away
            if (m_editor->IsMapped()) m_editor->UnmapWindow();
         }
      }

      if (selected())
      {
         GContext_t c = highlightContext()->GetGC();
         gVirtualX->FillRectangle(iID, c, iX - FWTabularWidget::kTextBuffer, iY - FWTabularWidget::kTextBuffer,
                                  iWidth + 2*FWTabularWidget::kTextBuffer, iHeight + 2*FWTabularWidget::kTextBuffer);
      } 
      int xOffset = 0;
      if(m_isParent) {
         const TGPicture* img = m_isOpen ?  openedImage(m_blackIcon) : closedImage(m_blackIcon);         
         img->Draw(iID,graphicsContext()->GetGC(),m_indentation+iX,iY +2);
         xOffset += img->GetWidth() + s_iconOffset;
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
   bool           m_blackIcon;
};

#endif
