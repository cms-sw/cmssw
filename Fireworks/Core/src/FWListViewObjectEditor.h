#ifndef Fireworks_Core_FWListViewObjectEditor_h
#define Fireworks_Core_FWListViewObjectEditor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListViewObjectEditor
//
/**\class FWListViewObjectEditor FWListViewObjectEditor.h Fireworks/Core/interface/FWListViewObjectEditor.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 10 09:02:56 CDT 2008
// $Id: FWListViewObjectEditor.h,v 1.4 2008/11/06 22:05:26 amraktad Exp $
//

// system include files
#include <vector>
#include <boost/shared_ptr.hpp>
#include "TGedFrame.h"

// user include files
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

// forward declarations
class TGVerticalFrame;
class FWParameterSetterBase;

class FWListViewObjectEditor : public TGedFrame, public FWParameterSetterEditorBase
{

public:
   FWListViewObjectEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                          UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~FWListViewObjectEditor();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void SetModel(TObject* obj);
   virtual void updateEditor();
   //ClassDef(FWListViewObjectEditor, 0);

private:
   FWListViewObjectEditor(const FWListViewObjectEditor&);    // stop default

   const FWListViewObjectEditor& operator=(const FWListViewObjectEditor&);    // stop default

   // ---------- member data --------------------------------
   TGVerticalFrame* m_frame;
   //can't use boost::shared_ptr because CINT will see this
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;

};


#endif
