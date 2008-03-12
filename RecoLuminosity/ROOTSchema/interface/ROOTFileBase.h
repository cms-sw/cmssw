/*

Adam Hunt - Princeton University
ahunt@princeton.edu

*/

#include <string>
#include <sstream>
#include <iostream>
#include <typeinfo>
#include <iomanip>

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>

#include "RecoLuminosity/HLXReadOut/CoreUtils/include/ICTypeDefs.hh"
#include "RecoLuminosity/HLXReadOut/HLXCoreLibs/include/LumiStructures.hh"

class ROOTFileBase{

 public:

  ROOTFileBase();
  ~ROOTFileBase();
  
  std::string GetFileName(){ return fileName; }
  std::string GetOutputDir(){ return outputDir; }
  void SetOutputDir(std::string dir){ outputDir = dir; }

 protected:
  
  TTree *m_tree;
  TFile *m_file;

  HCAL_HLX::LUMI_THRESHOLD      *Threshold;
  HCAL_HLX::LEVEL1_TRIGGER      *L1Trigger;
  HCAL_HLX::HLT                 *HLT;
  HCAL_HLX::TRIGGER_DEADTIME    *TriggerDeadtime;
  HCAL_HLX::LUMI_HF_RING_SET    *RingSet;

  HCAL_HLX::LUMI_SECTION_HEADER *Header;
  HCAL_HLX::LUMI_SUMMARY        *Summary;
  HCAL_HLX::LUMI_DETAIL         *Detail;
  
  HCAL_HLX::ET_SUM_SECTION      *EtSum;
  HCAL_HLX::OCCUPANCY_SECTION   *Occupancy;
  HCAL_HLX::LHC_SECTION         *LHC;

  HCAL_HLX::ET_SUM_SECTION      *EtSumPtr[HCAL_HLX_MAX_HLXS];
  HCAL_HLX::OCCUPANCY_SECTION   *OccupancyPtr[HCAL_HLX_MAX_HLXS];
  HCAL_HLX::LHC_SECTION         *LHCPtr[HCAL_HLX_MAX_HLXS];

  std::string fileName;
  std::string outputDir;
  
  void CreateFileName(const HCAL_HLX::LUMI_SECTION &localSection);
  void     CreateTree(const HCAL_HLX::LUMI_SECTION &localSection);
  void       FillTree(const HCAL_HLX::LUMI_SECTION &localSection);
  void CloseTree();
  void Concatenate(const HCAL_HLX::LUMI_SECTION &localSection);
  void InsertInformation();

  template< class T >
    void MakeBranch(const T &in, T **out, int HLXNum);
  
};
