#include <string>
#include <vector>
#include <assert.h>
#include <sstream>
#include <iostream>

#include "LightTableWidget.h"
#include "TColor.h"
#include "TGTextView.h"
#include "TGResourcePool.h"
#include "TGGC.h"
#include "TROOT.h"
#include "TableManagers.h"

ClassImp(LightTableWidget)

const int LightTableWidget::m_cellHeight;
const int LightTableWidget::m_titleColor;
const TGGC *LightTableWidget::fgShadowGC;

LightTableWidget::LightTableWidget (TGCompositeFrame *p, LightTableManager* tm, 
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

     // add "shadow" GC, which is 60% as bright as default
     if (fgShadowGC == 0) 
	  fgShadowGC = gClient->GetResourcePool()->GetFrameShadowGC();
     invisibleGC = *fgShadowGC;
     invisibleGC.SetFont(fNormGC.GetFont());
}

LightTableWidget::~LightTableWidget ()
{

}

void LightTableWidget::SetTextColor (Color_t col) 
{
     TColor *color = gROOT->GetColor(col);
     Pixel_t invisible = TColor::RGB2Pixel(Float_t(0.4 * color->GetRed()),
					   Float_t(0.4 * color->GetGreen()),
					   Float_t(0.4 * color->GetBlue()));
     invisibleGC.SetBackground(invisible);
     invisibleGC.SetForeground(invisible);
     TGTextView::SetForegroundColor(TColor::Number2Pixel(col));
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

	    if (!manager->rowIsSelected(pos.fY-3)/*manager->sel_indices.
		count(manager->table_row_to_index(pos.fY - 3)) == 0*/) {
		 if (pos.fY > 2 && !manager->rowIsVisible(pos.fY - 3)) {
		      // "invisible" items are greyed out
		      gVirtualX->DrawString(fCanvas->GetId(), invisibleGC(), Int_t(xoffset),
					    Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
					    buffer, Int_t(len));
		 } else {
		      gVirtualX->DrawString(fCanvas->GetId(), fNormGC(), Int_t(xoffset),
					    Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
					    buffer, Int_t(len));
		 }
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

void LightTableManager::format (std::vector<std::string> &ret, 
                                std::vector<int> &col_width,
                                int)
{
   ret.reserve(NumberOfRows() + 2); // col titles, horizontal line
   std::vector<std::string> titles = GetTitles(0);
   col_width.reserve(titles.size());
   for (std::vector<std::string>::const_iterator i = titles.begin();
        i != titles.end(); ++i) {
      col_width.push_back(i->length());
   }
   std::vector<std::string> row_content;
   for (int row = 0; row < NumberOfRows(); ++row) {
      FillCells(row, 0, row + 1, NumberOfCols(), row_content);
      for (std::vector<std::string>::const_iterator i = row_content.begin();
           i != row_content.end(); ++i) {
         const int length = i->length();
         if (col_width[i - row_content.begin()] < length)
            col_width[i - row_content.begin()] = length;
      }
   }
   int total_len = 0;
   for (unsigned int i = 0; i < col_width.size(); ++i) 
      total_len += col_width[i] + 1;
   //      ret.push_back(std::string(total_len, '=')); 
   //      sprintf(s, "%*s", (total_len + title().length()) / 2, title().c_str());
   //      ret.push_back(s);
   //      ret.push_back(std::string(total_len, '-')); 
   char *const s = new char[total_len+2];
   char * const sEnd = s+total_len+1;
   char *p = s;
   for (unsigned int i = 0; i < titles.size(); ++i) {
      p += snprintf(p, sEnd-p,"%*s", col_width[i] + 1, titles[i].c_str());
      assert(p<=sEnd);
   }
   ret.push_back(s);
   ret.push_back(std::string(total_len, '-')); 
   for (int row = 0; row < NumberOfRows(); ++row) {
      // 	  if (row == n_rows) {
      // 	       const char no_more[] = "more skipped";
      // 	       sprintf(s, "%*d %s", (total_len - sizeof(no_more)) / 2, 
      // 		       NumberOfRows() - row, no_more);
      // 	       ret.push_back(s);
      // 	       break;
      // 	  }
      FillCells(row, 0, row + 1, NumberOfCols(), row_content);
      char *p = s;
      for (unsigned int i = 0; i < row_content.size(); ++i) {
         p += snprintf(p, sEnd-p,"%*s", col_width[i] + 1, row_content[i].c_str());
         if(p>sEnd) {
            std::cout <<"exceeded row size of "<<total_len+1<<" with '"<<p
            <<"'\n while adding row "<<row<<" column "<<i<<" with value '"<<row_content[i]<<"'"<< std::endl;
         }
         assert(p<=sEnd);
      }
      ret.push_back(s);
   }
   delete [] s;
   //      ret.push_back(std::string(total_len, '-')); 
}

void LightTableManager::sort (int col, bool reset) 
{
   if (reset) {
      sort_asc_ = true;
      sort_col_ = col; 
   } else { 
      if (sort_col_ == col) {
         sort_asc_ = not sort_asc_;
      } else {
         sort_asc_ = true;
      }
      sort_col_ = col;
   }
   Sort(sort_col_, sort_asc_);
}

void LightTableManager::resort ()
{
     Sort(sort_col_, sort_asc_);
}
