// -*- C++ -*-
#ifndef Fireworks_Core_LightTableWidget_h
#define Fireworks_Core_LightTableWidget_h

#include <set>
#include <vector>

#include "GuiTypes.h"
#include "RQ_OBJECT.h"
#include "TGTextView.h"
 
class FWTableManager;
class TGCompositeFrame;
class TGTextView;

class LightTableWidget : public TGTextView { 
     RQ_OBJECT("TableWidget")
     
public: 
     LightTableWidget (TGCompositeFrame *p, FWTableManager* tm,
		       int w = 0, int h = 0);
     virtual ~LightTableWidget(); 
     void display (int rows = 5);
     void Reinit (int tabRows = 5) { display(tabRows); }
     void selectRow (int, Mask_t, Pixel_t) { }
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
     
// GUI functions
public:
     virtual Bool_t HandleButton(Event_t *event);

protected:
     virtual void DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);


protected:
     TGTextView		*textview;
     FWTableManager	*manager;
     std::vector<int>	col_widths;

// temporary hacks
public:
     static const int m_cellHeight	= 25;
     static const int m_titleColor	= 0xd8e6e6;

     ClassDef(LightTableWidget, 0)
};             

#endif
