#ifndef ROOTSCHEMA_H
#define ROOTSCHEMA_H

/*

Author: Adam Hunt - Princeton University
Date: 2007-10-05

*/



#include <string>
#include <sstream>
#include <iostream>

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>

#include "RecoLuminosity/HLXReadOut/CoreUtils/include/ICTypeDefs.hh"
#include "RecoLuminosity/HLXReadOut/HLXCoreLibs/include/LumiStructures.hh"

using std::cout;
using std::endl;

class ROOTSchema{
        
public:
    
    // Additional information not included in LUMI_SECTION
   
    TTree *m_tree;
    TFile *m_file;
    
    HCAL_HLX::LUMI_THRESHOLD     *Threshold;
    HCAL_HLX::LUMI_SECTION_HST   *LumiSectionHist;
    HCAL_HLX::LEVEL1_HLT_TRIGGER *L1HLTrigger;
    HCAL_HLX::TRIGGER_DEADTIME   *TriggerDeadtime;
    
    HCAL_HLX::LUMI_SECTION_HEADER *Header;
    HCAL_HLX::LUMI_SUMMARY        *Summary;
    HCAL_HLX::LUMI_BUNCH_CROSSING *BX;

    HCAL_HLX::ET_SUM_SECTION    *EtSum;
    HCAL_HLX::OCCUPANCY_SECTION *Occupancy;
    HCAL_HLX::LHC_SECTION       *LHC;

    ROOTSchema(std::string, std::string, const int &);
    ~ROOTSchema();
    void FillTree(const HCAL_HLX::LUMI_SECTION&);

    // MBCD = Make Branch, Copy Data
    void MBCD(const HCAL_HLX::ET_SUM_SECTION &in, HCAL_HLX::ET_SUM_SECTION **out,       int num, unsigned int compress);
    void MBCD(const HCAL_HLX::OCCUPANCY_SECTION &in, HCAL_HLX::OCCUPANCY_SECTION **out, int num, unsigned int compress);
    void MBCD(const HCAL_HLX::LHC_SECTION &in, HCAL_HLX::LHC_SECTION **out,             int num, unsigned int compress);
    // TO DO: make a template for these three functions
};

#endif
