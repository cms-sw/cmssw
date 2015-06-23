#include <iostream>
#include <fstream>
#include "Fireworks/Core/interface/FWPartialConfig.h"
#include "TGButton.h"
#include "TEveManager.h"
#include "TGString.h"
#include "TGLabel.h"
#include "TSystem.h"
#include "TROOT.h"

#include "Fireworks/Core/src/SimpleSAXParser.h"
#include "Fireworks/Core/interface/FWXMLConfigParser.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/interface/FWConfigurationManager.h"
#include "Fireworks/Core/interface/fwLog.h"  

namespace {
struct MyNameMap {
    std::vector< std::pair<std::string, std::string> > names = { { "GUI", "Windows" }, { "CommonPreferences", "Colors" }, { "EventNavigator" , "EventFilters" },   { "EventItems" , "Collections" } };

    std::string realName(std::string btnName)
    {
        for (std::vector< std::pair<std::string, std::string> >::iterator i = names.begin(); i!= names.end(); ++i) {
            if (i->second == btnName) 
                return i->first;
        }
        return btnName;
    }


    std::string btnName(const std::string& rlName)
    {

        for (std::vector< std::pair<std::string, std::string> >::iterator i = names.begin(); i!= names.end(); ++i) {
            if (i->first == rlName) 
                return i->second;
        }
        return rlName;
    }
};

    MyNameMap nmm;
}

FWPartialConfigGUI::FWPartialConfigGUI( const char* path, FWConfigurationManager* iCfg):
    TGTransientFrame(gClient->GetRoot(), FWGUIManager::getGUIManager()->getMainFrame(), 200, 140), m_cfgMng(iCfg)
{
    if (path) {
        std::ifstream g(path);
        FWXMLConfigParser parser(g);
        parser.parse();
        parser.config()->swap(m_origConfig);
    }
    else {
        FWConfiguration curr;
        m_cfgMng->to(curr);
        curr.swap(m_origConfig);
    }

    TGVerticalFrame* vf = new TGVerticalFrame(this);
    AddFrame(vf, new TGLayoutHints(kLHintsNormal, 2, 2, 4, 4));

    // can do a cast to non-const since we own the configuration
    FWConfiguration::KeyValues* kv = (FWConfiguration::KeyValues*)m_origConfig.keyValues();
    std::sort(kv->begin(), kv->end(),
              [](const std::pair<std::string, FWConfiguration>& lhs, const std::pair<std::string, FWConfiguration>& rhs) -> bool {
             return nmm.btnName(lhs.first) < nmm.btnName(rhs.first); } );

    for(FWConfiguration::KeyValues::const_iterator it = m_origConfig.keyValues()->begin(); it != m_origConfig.keyValues()->end(); ++it) {
        if ( it->second.keyValues()) {
            std::string nb =  nmm.btnName(it->first);
            TGCheckButton* cb = new TGCheckButton(vf, nb.c_str());
            vf->AddFrame(cb);

            m_entries.push_back(cb);
        }
    }
}

void FWPartialConfigGUI::Cancel()
{
    // AMT Is this eniugh. Should there be a destroy ?
    UnmapWindow();
}

//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------

FWPartialConfigLoadGUI::FWPartialConfigLoadGUI( const char* path, FWConfigurationManager* iCfg, FWEventItemsManager* iEiMng):
    FWPartialConfigGUI(path, iCfg), m_eiMng(iEiMng)
{
   TGHorizontalFrame* hf = new TGHorizontalFrame(this);
   AddFrame(hf, new TGLayoutHints( kLHintsRight| kLHintsBottom, 1, 1, 2, 4));
   TGTextButton* load = new TGTextButton(hf, " Load ");
   load->Connect("Clicked()", "FWPartialConfigLoadGUI", this, "Load()");
   hf->AddFrame(load, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
   TGTextButton* cancel = new TGTextButton(hf, " Cancel ");
   cancel->Connect("Clicked()", "FWPartialConfigGUI", this, "Cancel()");
   hf->AddFrame(cancel, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
   
   SetWindowName("Load Config");
   MapSubwindows();
   Layout();
   MapWindow();
}

FWPartialConfigLoadGUI::~FWPartialConfigLoadGUI() 
{
}

void FWPartialConfigLoadGUI::Load()
{   
    bool resetViews = true;
    bool resetEI = true;
    
    FWConfiguration::KeyValues* kv = (FWConfiguration::KeyValues*)m_origConfig.keyValues();
    
    for (auto i = m_entries.begin(); i != m_entries.end(); i++) {    
        if (!((*i)->IsOn())) {
            std::string key =  nmm.realName((*i)->GetText()->GetString());
            if (key == "EventItems") resetEI = false;
            if (key == "GUI") resetViews = false;

            for(FWConfiguration::KeyValues::iterator it = kv->begin(); it != kv->end(); ++it)
            {
                if ( key.compare(it->first) == 0) {
                    kv->erase(it);
                    break;
                }
            }
        } 
    }

    if (resetEI)
       m_eiMng->clearItems();

    if (resetViews)
        FWGUIManager::getGUIManager()->subviewDestroyAll();

    gEve->DisableRedraw();
    m_cfgMng ->setFrom(m_origConfig);
    gEve->EnableRedraw();

    UnmapWindow();
}

//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------

FWPartialConfigSaveGUI::FWPartialConfigSaveGUI( const char* path_out, const char* path_in, FWConfigurationManager* iCfg):
   FWPartialConfigGUI(0, iCfg), m_outFileName(path_out), m_currFileName(path_in)
{  
   TGHorizontalFrame* hf = new TGHorizontalFrame(this);
   AddFrame(hf, new TGLayoutHints( kLHintsRight| kLHintsBottom, 1, 1, 2, 4));
   
   TGTextButton* write = new TGTextButton(hf, " Write ");
   write->Connect("Clicked()", "FWPartialConfigSaveGUI", this, "Write()");
   hf->AddFrame(write, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));

   TGTextButton* cancel = new TGTextButton(hf, " Cancel ");
   cancel->Connect("Clicked()", "FWPartialConfigGUI", this, "Cancel()");
   hf->AddFrame(cancel, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));

   AddFrame(new TGLabel(this, Form("Output file: %s", gSystem->BaseName(path_out))),  new TGLayoutHints(kLHintsLeft, 8,2,3,3));
  
   SetWindowName("Save Config");

   MapSubwindows();
   Layout();
   MapWindow();
}


void
FWPartialConfigSaveGUI::Write()
{
    FWConfiguration destination;
    {
       m_currFileName = gSystem->Which(TROOT::GetMacroPath(), m_currFileName.c_str(), kReadPermission);
       // printf("going to parse m_currFileName %s\n", m_currFileName.c_str());
        std::ifstream g(m_currFileName.c_str());
        if (g.peek() == (int) '<') {
            FWXMLConfigParser parser(g);
            parser.parse();
            parser.config()->swap(destination);
            // printf("parsed %s ......... \n",m_currFileName.c_str() );
        }   
    }

    FWConfiguration curr;
    m_cfgMng->to(curr);

    FWConfiguration::KeyValues* cur_kv = (FWConfiguration::KeyValues*)m_origConfig.keyValues();
    FWConfiguration::KeyValues* old_kv = (FWConfiguration::KeyValues*)destination.keyValues();

    for (auto i = m_entries.begin(); i != m_entries.end(); i++) {    
        if ((*i)->IsOn()) {
            std::string key =  nmm.realName((*i)->GetText()->GetString());
            // printf("ON check key %s\n", key.c_str());
            for(FWConfiguration::KeyValues::iterator it = cur_kv->begin(); it != cur_kv->end(); ++it)
            {
                if ( key.compare(it->first) == 0) {
                    bool replace = false;
                    if (old_kv) {
                        for(FWConfiguration::KeyValues::iterator oldit = old_kv->begin(); oldit != old_kv->end(); ++oldit) {
                            if ( key.compare(oldit->first) == 0) {
                                replace = true;
                                oldit->second.swap(it->second);
                                break;
                            }
                        }
                    }

                    if (!replace)
                        destination.addKeyValue(it->first, it->second);

                    break;
                }
            }
        } 
    }

    // Dump content in the file
    std::ofstream file(m_outFileName.c_str());
    if(not file) {
       fwLog(fwlog::kError) << "FWPartialConfigSaveGUI::Write, can't open output file.!\n";
       return;
    }
    fwLog(fwlog::kInfo) << "Writing configuration to " << m_outFileName << std::endl;
    FWConfiguration::streamTo(file, destination, "top");
    UnmapWindow();
}
