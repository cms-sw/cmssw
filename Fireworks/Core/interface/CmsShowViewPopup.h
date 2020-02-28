#ifndef Fireworks_Core_CmsShowViewPopup_h
#define Fireworks_Core_CmsShowViewPopup_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowViewPopup
//
/**\class CmsShowViewPopup CmsShowViewPopup.h Fireworks/Core/interface/CmsShowViewPopup.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Wed Jun 25 15:15:12 EDT 2008
//

// system include files
#include <vector>
#ifndef __CINT__
#include <memory>
#include <sigc++/sigc++.h>
#endif
#include "TGFrame.h"

// user include files
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

// forward declarations
class TGLabel;
class TGTextButton;
class TGButton;
class TGFrame;
class TGTab;
class TGCompositeFrame;
class TEveWindow;

class FWParameterBase;
class FWParameterSetterBase;
class FWViewBase;
class FWColorManager;
class FWDialogBuilder;

/* Helper class to organise parameters in view controller */

class ViewerParameterGUI : public TGCompositeFrame, public FWParameterSetterEditorBase {
public:
  ViewerParameterGUI(const TGFrame*);
  ~ViewerParameterGUI() override {}

  ViewerParameterGUI& requestTab(const char*);
  ViewerParameterGUI& addParam(const FWParameterBase*);
  ViewerParameterGUI& separator();
  void reset();
  void populateComplete();

  TGCompositeFrame* getTabContainer();
  void addFrameToContainer(TGCompositeFrame*);

private:
  TGTab* m_tab;
  std::string m_selectedTabName;
#ifndef __CINT__
  std::vector<std::shared_ptr<FWParameterSetterBase> > m_setters;
#endif
};

//==============================================================================

class CmsShowViewPopup : public TGTransientFrame {
public:
  CmsShowViewPopup(const TGWindow* p = nullptr,
                   UInt_t w = 200,
                   UInt_t h = 200,
                   FWColorManager* cm = nullptr,
                   FWViewBase* wb = nullptr,
                   TEveWindow* ew = nullptr);
  ~CmsShowViewPopup() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  void CloseWindow() override;
  void MapWindow() override;
  void UnmapWindow() override;

  bool mapped() { return m_mapped; }

  void reset(FWViewBase*, TEveWindow* ew);

  void saveImage();
  void changeBackground();
  void backgroundColorWasChanged();
  TEveWindow* getEveWindow() const { return m_eveWindow; }
#ifndef __CINT__
  sigc::signal<void> closed_;
#endif

  ClassDefOverride(CmsShowViewPopup, 0);

private:
  CmsShowViewPopup(const CmsShowViewPopup&);                   // stop default
  const CmsShowViewPopup& operator=(const CmsShowViewPopup&);  // stop default

  // ---------- member data --------------------------------

  bool m_mapped;

  TGLabel* m_viewLabel;
  ViewerParameterGUI* m_paramGUI;
  TGTextButton* m_saveImageButton;
  TGTextButton* m_changeBackground;

  FWColorManager* m_colorManager;
  FWViewBase* m_viewBase;
  TEveWindow* m_eveWindow;
};

#endif
