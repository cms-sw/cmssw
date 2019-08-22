#ifndef Fireworks_Core_FWViewEnergyScaleEditor_h
#define Fireworks_Core_FWViewEnergyScaleEditor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewEnergyScaleEditor
//
/**\class FWViewEnergyScaleEditor FWViewEnergyScaleEditor.h Fireworks/Core/interface/FWViewEnergyScaleEditor.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Sep 24 18:52:28 CEST 2010
//

// system include files

// user include files
#ifndef __CINT__
#include <memory>
#endif
#include "TGFrame.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

class FWViewEnergyScale;
class FWParameterSetterBase;
class FWParameterBase;
class TGCheckButton;

// forward declarations

class FWViewEnergyScaleEditor : public TGVerticalFrame, public FWParameterSetterEditorBase {
public:
  FWViewEnergyScaleEditor(FWViewEnergyScale* s, TGCompositeFrame* w, bool addAutoScaleControll = true);
  ~FWViewEnergyScaleEditor() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void setEnabled(bool);

  ClassDefOverride(FWViewEnergyScaleEditor, 0);

private:
  FWViewEnergyScaleEditor(const FWViewEnergyScaleEditor&);                   // stop default
  const FWViewEnergyScaleEditor& operator=(const FWViewEnergyScaleEditor&);  // stop default

  void addParam(FWParameterBase*, const char* title = nullptr);

  // ---------- member data --------------------------------

  FWViewEnergyScale* m_scale;
  bool m_enabled;

#ifndef __CINT__
  std::vector<std::shared_ptr<FWParameterSetterBase> > m_setters;
#endif
};

#endif
