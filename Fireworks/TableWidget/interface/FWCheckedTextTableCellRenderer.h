#ifndef Fireworks_TableWidget_FWCheckedTextTableCellRenderer_h
#define Fireworks_TableWidget_FWCheckedTextTableCellRenderer_h
// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWCheckedTextTableCellRenderer
// 
/**\class FWCheckedTextTableCellRenderer FWCheckedTextTableCellRenderer.h Fireworks/TableWidget/interface/FWCheckedTextTableCellRenderer.h

 Description: A Cell renderer which shows both a check box and text

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Feb  3 14:29:48 EST 2009
// $Id: FWCheckedTextTableCellRenderer.h,v 1.2 2009/03/04 15:31:20 chrjones Exp $
//

// system include files
#include "TQObject.h"

// user include files
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"

// forward declarations

class FWCheckedTextTableCellRenderer : public FWTextTableCellRenderer, public TQObject
{

   public:
      FWCheckedTextTableCellRenderer(const TGGC* iContext=&(getDefaultGC()));
      virtual ~FWCheckedTextTableCellRenderer();

      // ---------- const member functions ---------------------
      bool isChecked() const;

      virtual UInt_t width() const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setChecked( bool);

      virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight);

      virtual void buttonEvent(Event_t* iClickEvent, int iRelClickX, int iRelClickY);

      void checkBoxClicked(); //*SIGNAL*

      ClassDef(FWCheckedTextTableCellRenderer,0);

   private:
      //FWCheckedTextTableCellRenderer(const FWCheckedTextTableCellRenderer&); // stop default

      //const FWCheckedTextTableCellRenderer& operator=(const FWCheckedTextTableCellRenderer&); // stop default

      // ---------- member data --------------------------------
      static const UInt_t kGap = 2;
      bool m_isChecked;

};


#endif
