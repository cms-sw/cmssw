#ifndef Fireworks_Core_FWListModelEditor_h
#define Fireworks_Core_FWListModelEditor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListModelEditor
//
/**\class FWListModelEditor FWListModelEditor.h Fireworks/Core/interface/FWListModelEditor.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Mar  3 17:20:24 EST 2008
// $Id: FWListModelEditor.h,v 1.2 2008/11/06 22:05:26 amraktad Exp $
//

// system include files
#include "TGedFrame.h"

// user include files

// forward declarations
class FWListModel;
class TGTextButton;

class FWListModelEditor : public TGedFrame
{

public:
   FWListModelEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                     UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~FWListModelEditor();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void SetModel(TObject* obj);
   ClassDef(FWListModelEditor, 0);

   void openDetailView();
private:
   FWListModelEditor(const FWListModelEditor&);    // stop default

   const FWListModelEditor& operator=(const FWListModelEditor&);    // stop default

   // ---------- member data --------------------------------
   TGTextButton* m_showDetailViewButton;
   FWListModel* m_model;
};


#endif
