#ifndef ROOTSCHEMA_H
#define ROOTSCHEMA_H

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>

#include "RecoLuminosity/HLXReadOut/CoreUtils/include/ICTypeDefs.hh"
#include "RecoLuminosity/HLXReadOut/HLXCoreLibs/include/LumiStructures.hh"

class ROOTSchema{
        
public:
    
    // Additional information not included in LUMI_SECTION
   
    TTree *m_tree;
    TFile *m_file;
    
    HCAL_HLX::LUMI_SUMMARY *Summary;
    HCAL_HLX::LUMI_BUNCH_CROSSING *BX;
    HCAL_HLX::LUMI_THRESHOLD *Threshold;
    // JJ LUMI_SECTION
    HCAL_HLX::LUMI_SECTION* LumiSection;

    HCAL_HLX::LUMI_SECTION_HST *LumiSectionHist;
    HCAL_HLX::LEVEL1_HLT_TRIGGER *L1HLTrigger;
    HCAL_HLX::TRIGGER_DEADTIME *TriggerDeadtime;
    
    ROOTSchema(std::string, std::string, const int &);
    ~ROOTSchema();
    void FillTree(const HCAL_HLX::LUMI_SECTION&);
}; //~class LumiComp 

#endif
