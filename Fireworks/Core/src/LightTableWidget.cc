#include <string>
#include <vector>

#include "LightTableWidget.h"
#include "TGTextView.h"
#include "TableManagers.h"

ClassImp(LightTableWidget)

const int LightTableWidget::m_cellHeight;
const int LightTableWidget::m_titleColor;

LightTableWidget::LightTableWidget (TGCompositeFrame *p, FWTableManager* tm, 
				    int w, int h)
     : TGTextView(p, w, h), 
       manager(tm)
{
     SetBackground(GetBlackPixel());
     SetForegroundColor(GetWhitePixel());
     SetSelectBack(GetWhitePixel());
     SetSelectFore(GetBlackPixel());
     TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY | 
					      kLHintsShrinkX | kLHintsShrinkY);
     p->AddFrame(this, hints);
}

LightTableWidget::~LightTableWidget ()
{
     delete textview;
}

void LightTableWidget::display (int rows)
{
     Clear();
     std::vector<std::string> text;
     col_widths.clear();
     manager->format(text, col_widths, rows);
     for (std::vector<std::string>::const_iterator i = text.begin(); 
	  i != text.end(); ++i) {
	  AddLineFast(i->c_str());
     }
     Update();
}

void LightTableWidget::selectRows (const std::set<int> &rows, Mask_t mask, 
				   Pixel_t hcolor)
{
     // this is easy: just redraw, DrawRegion() will ask the manager
     // for the list of selected rows
     DrawRegion(0, 0, GetWidth(), GetHeight());
}

Bool_t LightTableWidget::HandleButton(Event_t *event)
{
   // Handle mouse button event in text editor.

   if (event->fWindow != fCanvas->GetId()) {
      return kFALSE;
   }

   if (event->fCode == kButton1) {
      if (event->fType == kButtonPress) {
	   fMousePos.fY = ToObjYCoord(fVisible.fY + event->fY);
	   fMousePos.fX = ToObjXCoord(fVisible.fX + event->fX, fMousePos.fY);
// 	   printf("click on line %d, col %d\n", fMousePos.fY, fMousePos.fX);
	   if (fMousePos.fY >= 3 && fMousePos.fY <= manager->NumberOfRows() + 2) 
		manager->Selection(fMousePos.fY - 3, event->fState);
	   else if (fMousePos.fY == 1) {
		for (int col = 0, i = 0; i < col_widths.size(); ++i) {
		     col += col_widths[i] + 1;
		     if (col > fMousePos.fX) {
			  manager->sort(i);
			  display();
			  break;
		     }
		}
	   }
      } 
   } else if (event->fCode == kButton4) {
      // move three lines up
      if (fVisible.fY > 0) {
         SetVsbPosition(fVisible.fY / fScrollVal.fY - 3);
         //Mark(fMousePos.fX, fMarkedStart.fY - 3);
      }
   } else if (event->fCode == kButton5) {
      // move three lines down
      if ((Int_t)fCanvas->GetHeight() < ToScrYCoord(ReturnLineCount())) {
         TGLongPosition size;
         size.fY = ToObjYCoord(fVisible.fY + fCanvas->GetHeight()) - 1;
         SetVsbPosition(fVisible.fY / fScrollVal.fY + 3);
         //Mark(fMousePos.fX, size.fY + 3);
      }
   } 

//    if (event->fType == kButtonRelease) {
//       if (event->fCode == kButton1) {
//          if (fIsMarked) {
//             Copy();
//          }
//       }
//    }

   return kTRUE;
}

void LightTableWidget::DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw lines in exposed region.

     x = 0;
     y -= 1;
     if (y < 0)
	  y = 0;
     w = GetWidth();
     char *buffer;

   TGLongPosition pos;
   Long_t xoffset, len, len1, len2;
   Long_t line_count = fText->RowCount();
   Rectangle_t rect;
   rect.fX = x;
   rect.fY = y;
   pos.fY = ToObjYCoord(fVisible.fY + h);
   rect.fHeight = UShort_t(h + ToScrYCoord(pos.fY + 2) - ToScrYCoord(pos.fY));
   pos.fX = ToObjXCoord(fVisible.fX + w, pos.fY);
   rect.fWidth = UShort_t(w + ToScrXCoord(pos.fX + 1, pos.fY) - ToScrXCoord(pos.fX, pos.fY));
   Int_t yloc = rect.fY + (Int_t)fScrollVal.fY;
   pos.fY = ToObjYCoord(fVisible.fY + rect.fY);

   while (pos.fY < line_count &&
          yloc - fScrollVal.fY < (Int_t)fCanvas->GetHeight() &&
          yloc  < rect.fY + rect.fHeight) {

      pos.fX = ToObjXCoord(fVisible.fX + rect.fX, pos.fY);
      xoffset = ToScrXCoord(pos.fX, pos.fY);
      len = fText->GetLineLength(pos.fY) - pos.fX;

      gVirtualX->ClearArea(fCanvas->GetId(), x, Int_t(ToScrYCoord(pos.fY)),
                           rect.fWidth, UInt_t(ToScrYCoord(pos.fY+1)-ToScrYCoord(pos.fY)));


      if (len > 0) {
         if (len > ToObjXCoord(fVisible.fX + rect.fX + rect.fWidth, pos.fY) - pos.fX) {
            len = ToObjXCoord(fVisible.fX + rect.fX + rect.fWidth, pos.fY) - pos.fX + 1;
         }
         if (pos.fX == 0) {
            xoffset = -fVisible.fX;
         }
         if (pos.fY >= ToObjYCoord(fVisible.fY)) {
            buffer = fText->GetLine(pos, len);
            Int_t i = 0;
            while (buffer[i] != '\0') {
               if (buffer[i] == '\t') {
                  buffer[i] = ' ';
                  Int_t j = i+1;
                  while (buffer[j] == 16 && buffer[j] != '\0') {
                     buffer[j++] = ' ';
                  }
               }
               i++;
            }

	    if (manager->sel_indices.
		count(manager->table_row_to_index(pos.fY - 3)) == 0) {
                gVirtualX->DrawString(fCanvas->GetId(), fNormGC(), Int_t(xoffset),
                                      Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
                                      buffer, Int_t(len));
	    } else {
		 gVirtualX->FillRectangle(fCanvas->GetId(), fSelbackGC(),
					  Int_t(ToScrXCoord(pos.fX, pos.fY)),
					  Int_t(ToScrYCoord(pos.fY)),
					  UInt_t(ToScrXCoord(pos.fX+len, pos.fY)),
					  UInt_t(ToScrYCoord(pos.fY+1)-ToScrYCoord(pos.fY)));
		 gVirtualX->DrawString(fCanvas->GetId(), fSelGC(), Int_t(xoffset),
				       Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
				       buffer, Int_t(len));
            }
            delete [] buffer;
         }
      }
      pos.fY++;
      yloc += Int_t(ToScrYCoord(pos.fY) - ToScrYCoord(pos.fY-1));
   }
}
