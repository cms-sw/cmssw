#include <string>
#include <vector>
#include <iostream>
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWHLTValidator.h"

class TGVerticalFrame;
class TGFrame;
class TGTransientFrame;
class TGCompositeFrame;
class TGTextButton;

class FWGUIEventFilter: public TGTransientFrame{
  std::vector<FWEventSelector*>* m_main_sels;
  std::vector<FWEventSelector*>  m_sels;
  bool* m_main_globalOR;
  bool m_globalOR;
  TGCompositeFrame* m_mainFrame;
  std::vector<TGVerticalFrame*> m_columns;
  std::vector<std::vector<TGFrame*> > m_cells;
  static const int m_entryHeight = 20;
  static const int m_width = 500;
  static const int m_height = 500;
  bool m_haveNewEntry;
  FWEventSelector m_newEntry;
  TGTextButton* m_junctionWidget;
  FWHLTValidator m_validator;
  static const TGPicture* m_icon_enabled;
  static const TGPicture* m_icon_disabled;
  bool m_changed;
  TGTextButton* m_applyChanges;
  void changed(bool flag = true){
    m_changed = flag;
    m_applyChanges->SetEnabled(flag);
  }

public:
  FWGUIEventFilter(const TGWindow* parent,
		   std::vector<FWEventSelector*>& sels,
		   fwlite::Event&,
		   bool&);
  void addSelector(FWEventSelector* sel);
  void show();
  void update();
  void applyChanges();
  void dump(const char* text);
  void newEntry(const char* text=0);
  void junctionChanged();
  void junctionUpdate();
  
  void CloseWindow()
  {
    DoExit();
  }

  void DoExit()
  {
    UnmapWindow();
  }
};

