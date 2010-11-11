#ifndef Fireworks_Core_FWEveViewScaleEditor_h
#define Fireworks_Core_FWEveViewScaleEditor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveViewScaleEditor
// 
/**\class FWEveViewScaleEditor FWEveViewScaleEditor.h Fireworks/Core/interface/FWEveViewScaleEditor.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Sep 24 18:52:28 CEST 2010
// $Id: FWEveViewScaleEditor.h,v 1.3 2010/11/10 20:07:07 amraktad Exp $
//

// system include files

// user include files
#ifndef __CINT__
#include <boost/shared_ptr.hpp>
#endif
#include "TGFrame.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

class FWViewEnergyScale;
class FWParameterSetterBase;
class FWParameterBase;
class TGCheckButton;

// forward declarations

class FWEveViewScaleEditor : public TGVerticalFrame, public FWParameterSetterEditorBase
{
public:
   FWEveViewScaleEditor(TGCompositeFrame* w, FWViewEnergyScale* s=0);
   virtual ~FWEveViewScaleEditor();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void useGlobalScales();

   ClassDef(FWEveViewScaleEditor, 0);

private:
   FWEveViewScaleEditor(const FWEveViewScaleEditor&); // stop default

   const FWEveViewScaleEditor& operator=(const FWEveViewScaleEditor&); // stop default
   
   void addParam(const FWParameterBase*, const char* title = 0);   
   
   // ---------- member data --------------------------------
   FWViewEnergyScale* m_scale;
#ifndef __CINT__
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
#endif
   TGCheckButton* m_globalScalesBtn;
};


#endif
