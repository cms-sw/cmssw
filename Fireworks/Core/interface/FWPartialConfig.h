#ifndef Fireworks_Core_FWPartialConfig
#define Fireworks_Core_FWPartialConfig
#include <TGFrame.h>
#include <vector>
#include "Fireworks/Core/interface/FWConfiguration.h"

class TGCheckButton;
class FWConfigurationManager;
class FWEventItemsManager;

class FWPartialLoadGUI : public TGMainFrame
{
 public:
  FWPartialLoadGUI( const char* path, FWEventItemsManager*, FWConfigurationManager* );
  ~FWPartialLoadGUI();

  void Load();
  void Cancel();

 private:
  struct ConfEntry {
    TGCheckButton* btn;
    bool from_gui;
    ConfEntry(TGCheckButton* b,  bool g) : btn(b), from_gui(g){}
  };

  std::vector <ConfEntry> m_entries;
  FWConfiguration m_origConfig;
  FWEventItemsManager* m_eiMng;
  FWConfigurationManager* m_cfgMng;

  ClassDef(FWPartialLoadGUI, 0);
};

#endif
