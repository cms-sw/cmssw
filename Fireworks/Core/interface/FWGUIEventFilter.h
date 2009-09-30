#include <string>
#include <vector>
#include <iostream>
#include "Fireworks/Core/interface/FWEventSelector.h";
class TGVerticalFrame;
class TGFrame;
class TGTransientFrame;
class TGCompositeFrame;
class TGTextButton;

class FWGUIEventFilter: public TGTransientFrame{
  std::vector<FWEventSelector>& m_sels;
  bool& m_globalOR;
  TGCompositeFrame* m_mainFrame;
  std::vector<TGVerticalFrame*> m_columns;
  std::vector<std::vector<TGFrame*> > m_cells;
  static const int m_entryHeight = 20;
  static const int m_width = 500;
  static const int m_height = 500;
  bool m_haveNewEntry;
  FWEventSelector m_newEntry;
  TGTextButton* m_junctionWidget;

public:
  FWGUIEventFilter(std::vector<FWEventSelector>& sels,
		   bool&);
  void addSelector(FWEventSelector& sel);
  void show();
  void update();
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
    // DeleteWindow();
  }
};

