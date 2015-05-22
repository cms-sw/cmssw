#include <iostream>
#include <fstream>
#include "Fireworks/Core/interface/FWPartialConfig.h"
#include "TGButton.h"
#include "TEveManager.h"
#include "TGString.h"

#include "Fireworks/Core/src/SimpleSAXParser.h"
#include "Fireworks/Core/interface/FWXMLConfigParser.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWConfigurationManager.h"

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


FWPartialLoadGUI::FWPartialLoadGUI( const char* path, FWEventItemsManager* iEiMng, FWConfigurationManager* iCfg):
    TGMainFrame(gClient->GetRoot(), 200, 140),m_eiMng(iEiMng), m_cfgMng(iCfg)
{
    std::ifstream g(path);
    FWXMLConfigParser parser(g);
    parser.parse();
    parser.config()->swap(m_origConfig);
    
    TGVerticalFrame* vf = new TGVerticalFrame(this);
    AddFrame(vf, new TGLayoutHints(kLHintsNormal, 2, 2, 4, 4));
    for(FWConfiguration::KeyValues::const_iterator it = m_origConfig.keyValues()->begin(); it != m_origConfig.keyValues()->end(); ++it) {
        // printf("itfirst %s kv = %p  sv = %p \n", it->first.c_str(), it->second.keyValues(), it->second.stringValues());
        if ( it->second.keyValues()) {
            std::string nb =  nmm.btnName(it->first);
            TGCheckButton* cb = new TGCheckButton(this, nb.c_str());
            m_entries.push_back(ConfEntry(cb, false));
            vf->AddFrame(cb);
        }
    }
    
   TGHorizontalFrame* hf = new TGHorizontalFrame(this);
   AddFrame(hf, new TGLayoutHints( kLHintsRight| kLHintsBottom, 1, 1, 2, 4));
   TGTextButton* load = new TGTextButton(hf, " Load ");
   load->Connect("Clicked()", "FWPartialLoadGUI", this, "Load()");
   hf->AddFrame(load, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
   TGTextButton* cancel = new TGTextButton(hf, " Cancel ");
   cancel->Connect("Clicked()", "FWPartialLoadGUI", this, "Cancel()");
   hf->AddFrame(cancel, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
   

   SetWindowName("Partial Load Config");
   MapSubwindows();
   Layout();
   MapWindow();
}

FWPartialLoadGUI::~FWPartialLoadGUI() 
{
}

void FWPartialLoadGUI::Load()
{   
    bool resetViews = true;
    bool resetEI = true;
    
    FWConfiguration::KeyValues* kvTop = (FWConfiguration::KeyValues*)m_origConfig.keyValues();
    FWConfiguration::KeyValues* kvGUI = 0;
    
    for (auto & el : m_entries) {    
        if (!el.btn->IsOn()) {
            std::string key =  nmm.realName(el.btn->GetText()->GetString());
            if (key == "EventItems") resetEI = false;
            if (key == "GUI") resetViews = false;
            FWConfiguration::KeyValues* kv = (el.from_gui) ? kvGUI : kvTop;

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


void FWPartialLoadGUI::Cancel()
{
    // AMT should not there be a destroy ??
    UnmapWindow();
}
