#include <string>
#include <vector>
#include <iostream>
#include "Fireworks/Core/interface/FWEventSelector.h";
class TGVerticalFrame;
class TGFrame;
class TGTransientFrame;
class TGCompositeFrame;

class FWGUIEventFilter: public TGTransientFrame{
  std::vector<FWEventSelector>& m_sels;
  TGCompositeFrame* m_mainFrame;
  std::vector<TGVerticalFrame*> m_columns;
  std::vector<std::vector<TGFrame*> > m_cells;
  static const int m_entryHeight = 20;
  static const int m_width = 500;
  static const int m_height = 500;
  bool m_haveNewEntry;
  FWEventSelector m_newEntry;

public:
  FWGUIEventFilter(std::vector<FWEventSelector>& sels);
  void addSelector(FWEventSelector& sel);
  void show();
  void update();
  void dump(const char* text);
  void newEntry(const char* text=0);
  
  void CloseWindow()
  {
    DoExit();
  }

  void DoExit()
  {
    UnmapWindow();
    DeleteWindow();
  }
};

