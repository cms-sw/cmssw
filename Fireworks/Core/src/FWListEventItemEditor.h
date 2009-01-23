#ifndef Fireworks_Core_FWListEventItemEditor_h
#define Fireworks_Core_FWListEventItemEditor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListEventItemEditor
//
/**\class FWListEventItemEditor FWListEventItemEditor.h Fireworks/Core/interface/FWListEventItemEditor.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Mar  3 09:35:59 EST 2008
// $Id: FWListEventItemEditor.h,v 1.3 2008/11/06 22:05:26 amraktad Exp $
//

// system include files
#include "TGedFrame.h"

// user include files

// forward declarations
class FWListEventItem;
class TGTextEntry;
class TGTextButton;

class FWListEventItemEditor : public TGedFrame
{

public:
   FWListEventItemEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                         UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~FWListEventItemEditor();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void SetModel(TObject* obj);

   void runFilter();
   void removeItem();
   ClassDef(FWListEventItemEditor, 0);
private:
   FWListEventItemEditor(const FWListEventItemEditor&);    // stop default

   const FWListEventItemEditor& operator=(const FWListEventItemEditor&);    // stop default

   // ---------- member data --------------------------------
   FWListEventItem* m_item;
   TGTextEntry* m_filterExpression;
   TGTextButton* m_filterRunExpressionButton;
};


#endif
