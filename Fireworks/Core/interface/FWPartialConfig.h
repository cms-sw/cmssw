#ifndef Fireworks_Core_FWPartialConfig
#define Fireworks_Core_FWPartialConfig
#include <TGFrame.h>
#include <vector>
#include "Fireworks/Core/interface/FWConfiguration.h"

class TGCheckButton;
class FWConfigurationManager;
class FWEventItemsManager;

class FWPartialConfigGUI : public TGMainFrame
{
 public:
  FWPartialConfigGUI(const char* path,  FWConfigurationManager*);
  ~FWPartialConfigGUI() {}
 protected:
  std::vector <TGCheckButton*> m_entries;
  FWConfiguration m_origConfig;
  FWConfigurationManager* m_cfgMng;
};

//---------------------------------------------------------------------

class FWPartialConfigLoadGUI : public FWPartialConfigGUI
{
 public:
  FWPartialConfigLoadGUI( const char* path,  FWConfigurationManager* ,FWEventItemsManager*);
  ~FWPartialConfigLoadGUI();

  void Load();
  void Cancel();

 private:
  FWEventItemsManager* m_eiMng;
  const char* m_oldConfigName;
  ClassDef(FWPartialConfigLoadGUI, 0);
};


//---------------------------------------------------------------------

class FWPartialConfigSaveGUI : public FWPartialConfigGUI
{
 public:
  FWPartialConfigSaveGUI( const char* path_out, FWConfigurationManager* );
  ~FWPartialConfigSaveGUI() {}

  void Write();
  void Cancel();

 private:
  std::string m_outFileName;

  ClassDef(FWPartialConfigSaveGUI, 0);
};


#endif
