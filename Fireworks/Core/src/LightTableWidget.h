// -*- C++ -*-
#ifndef Fireworks_Core_LightTableWidget_h
#define Fireworks_Core_LightTableWidget_h

#include <set>
#include <vector>

#include "GuiTypes.h"
#include "RQ_OBJECT.h"
#include "TGTextView.h"
#include "Fireworks/Core/interface/TableManager.h"

class TGCompositeFrame;
class TGTextView;
class TGGC;

class LightTableManager : public TableManager {
public:
   LightTableManager() : sort_asc_(true), sort_col_(0), m_print_index(false) {
   }

   void format (std::vector<std::string> &ret,
                std::vector<int> &col_widths,
                int n_rows);
   virtual void sort (int col, bool reset = false);
   virtual void resort ();
   virtual int preamble () {
      return 3 + (title().length() != 0);
   }
   virtual bool rowIsSelected(int row) const = 0;
   virtual bool rowIsVisible (int row) const {
      return true;
   }
   void setPrintIndex(bool iValue) {
      m_print_index = iValue;
   }
private:
   bool sort_asc_;
   int sort_col_;
   bool m_print_index;
};


class LightTableWidget : public TGTextView {
   RQ_OBJECT("LightTableWidget")

public:
   LightTableWidget (TGCompositeFrame *p, LightTableManager* tm,
                     int w = 0, int h = 0);
   virtual ~LightTableWidget();
   void display (int rows = 5);
   void Reinit (int tabRows = 5) {
      display(tabRows);
   }
   void selectRow (int, Mask_t, Pixel_t) {
   }
   void SelectRow (int row, Mask_t mask = 0, Pixel_t hcolor = 0)
   {
      selectRow(row, mask, hcolor);
   }
   void selectRows (const std::set<int> &row, Mask_t mask = 0,
                    Pixel_t hcolor = 0);
   void SelectRows (const std::set<int> &row, Mask_t mask = 0,
                    Pixel_t hcolor = 0)
   {
      selectRows(row, mask, hcolor);
   }
   void SetTextColor (Color_t col);

// GUI functions
public:
   virtual Bool_t HandleButton(Event_t *event);

protected:
   virtual void DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);


protected:
   LightTableManager  *manager;
   std::vector<int>   col_widths;
   static const TGGC  *fgShadowGC;
   TGGC invisibleGC;

// temporary hacks
public:
   static const int m_cellHeight      = 25;
   static const int m_titleColor      = 0xd8e6e6;

   ClassDef(LightTableWidget, 0)
};

#endif
