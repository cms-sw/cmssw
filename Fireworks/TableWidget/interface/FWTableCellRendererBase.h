#ifndef Fireworks_TableWidget_FWTableCellRendererBase_h
#define Fireworks_TableWidget_FWTableCellRendererBase_h
// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWTableCellRendererBase
// 
/**\class FWTableCellRendererBase FWTableCellRendererBase.h Fireworks/TableWidget/interface/FWTableCellRendererBase.h

 Description: base class for classes which handle drawing and interaction with cells in the table

 Usage:
    Renderers do the actual drawing of data into the cell (via the 'draw' method) and can allow user interaction with a cell
    (via the 'buttonEvent' method).  When drawing a cell in the table, FWTableWidget ask its held FWTableManagerBase for a 
    FWTableCellRendererBase for that particular cell and then calls the 'draw' method. Similarly, when a mouse button is pressed
    while the cursor is in a particular cell, the FWTableWidget asks its held FWTableManagerBase for a FWTableCellRendererBase for that
    particular cell and then calls the 'buttonEvent' method.  For efficiency reasons, we encourage the reuse of the same 
    FWTableCellRendererBase instance for multiple cells.  The FWTableWidget is written so that it always requests a renderer from the
    FWTableManagerBase thereby allowing this reuse to work even when the renderer holds state information particular for only one cell
     (e.g. color).
     
     Classes which inherit from FWTableManagerBase are responsible for creating the appropriate FWTableCellRendererBase for the data
     held in the table's cells.  When the FWTableWidget asks the FWTableManagerBase for a renderer for a particular cell, it is the 
     FWTableManagerBase's job to reset the renderer so that it can properly draw and/or interact with that cell.  To allow reuse of
     the same object, classes which inherit from FWTableCellRendererBase normally have 'setter' methods which are used to set the
     renderer so it can represent a particular cell.
    
    One inherits from FWTableCellRendererBase in order to provide custom ways to view data in the cell or to customize how users
    interact with the cell.  Inheritance can also be used to allow a renderer to work directly with data held by a FWTableManagerBase.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:40:18 EST 2009
// $Id: FWTableCellRendererBase.h,v 1.1 2009/02/03 20:33:03 chrjones Exp $
//

// system include files
#include "GuiTypes.h"

// user include files

// forward declarations

class FWTableCellRendererBase
{

   public:
      FWTableCellRendererBase();
      virtual ~FWTableCellRendererBase();

      // ---------- const member functions ---------------------
      ///returns the minimum width of the cell to which the renderer is representing
      virtual UInt_t width() const= 0;
      ///returns the minimum height of the cell to which the renderer is representing
      virtual UInt_t height() const = 0;
      
      /** Called to draw a particular cell: arguments
      iID: the id for the drawable in the window. Needed in order to do calls to gVirtualX or to TGFont
      iX: screen x position that the cell drawing area starts
      iY: screen y position that the cell drawing area starts
      iWidth: width (x dimension) of cell drawing area.  May be larger than value returned from width()
      iHeight: height (x dimension) of cell drawing area. May be larger than value returned from height()
      */
      virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)=0;

      // ---------- member functions ---------------------------
      /** Called when a mouse button event occurs when the cursor is over a particular cell: arguments
      iClickEvent: the ROOT GUI event caused by the mouse button
      iRelClickX: the x position of the cursor click relative to the start of the cell drawing area
      iRelClickY: the y position of the cursor click relative to the start of the cell drawing area
      */
      virtual void buttonEvent(Event_t* iClickEvent, int iRelClickX, int iRelClickY);

   private:
      FWTableCellRendererBase(const FWTableCellRendererBase&); // stop default

      const FWTableCellRendererBase& operator=(const FWTableCellRendererBase&); // stop default

      // ---------- member data --------------------------------

};


#endif
