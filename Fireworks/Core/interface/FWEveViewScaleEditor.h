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
// $Id$
//

// system include files

// user include files
#include <boost/shared_ptr.hpp>

#include "TGFrame.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

class FWEveView;
class FWParameterSetterBase;
class FWParameterBase;
class TGCheckButton;

// forward declarations

class FWEveViewScaleEditor : public TGVerticalFrame, public FWParameterSetterEditorBase
{
public:
   FWEveViewScaleEditor(TGCompositeFrame* w, FWEveView* v=0);
   virtual ~FWEveViewScaleEditor();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void addParam(const FWParameterBase*);
   void useGlobalScales();

private:
   FWEveViewScaleEditor(const FWEveViewScaleEditor&); // stop default

   const FWEveViewScaleEditor& operator=(const FWEveViewScaleEditor&); // stop default

   // ---------- member data --------------------------------
   FWEveView* m_view;
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
   TGCheckButton* m_globalScalesBtn;
};


#endif
