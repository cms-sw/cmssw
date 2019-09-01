#ifndef Fireworks_Core_FWPartialConfig
#define Fireworks_Core_FWPartialConfig
#include <TGFrame.h>
#include <vector>
#include "Fireworks/Core/interface/FWConfiguration.h"

class TGCheckButton;
class FWConfigurationManager;
class FWEventItemsManager;

class FWPartialConfigGUI : public TGTransientFrame {
public:
  FWPartialConfigGUI(const char* path, FWConfigurationManager*);
  ~FWPartialConfigGUI() override {}
  void Cancel();

protected:
  std::vector<TGCheckButton*> m_entries;
  FWConfiguration m_origConfig;
  FWConfigurationManager* m_cfgMng;

  ClassDefOverride(FWPartialConfigGUI, 0);
};

//---------------------------------------------------------------------

class FWPartialConfigLoadGUI : public FWPartialConfigGUI {
public:
  FWPartialConfigLoadGUI(const char* path, FWConfigurationManager*, FWEventItemsManager*);
  ~FWPartialConfigLoadGUI() override;

  void Load();

private:
  FWEventItemsManager* m_eiMng;
  const char* m_oldConfigName;
  ClassDefOverride(FWPartialConfigLoadGUI, 0);
};

//---------------------------------------------------------------------

class FWPartialConfigSaveGUI : public FWPartialConfigGUI {
public:
  FWPartialConfigSaveGUI(const char* path_out, const char* path_in, FWConfigurationManager*);
  ~FWPartialConfigSaveGUI() override {}

  void WriteConfig();

private:
  std::string m_outFileName;
  std::string m_currFileName;

  ClassDefOverride(FWPartialConfigSaveGUI, 0);
};

#endif
