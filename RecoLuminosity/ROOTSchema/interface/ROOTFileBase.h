#ifndef __ROOTFILEBASE_H__
#define __ROOTFILEBASE_H__

/*

Adam Hunt - Princeton University
ahunt@princeton.edu

*/
#include "RecoLuminosity/TCPReceiver/interface/ICTypeDefs.hh"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"
#include "RecoLuminosity/TCPReceiver/interface/TimeStamp.h"

#include <string>

#include "TFile.h"
#include "TTree.h"

namespace HCAL_HLX{

  class ROOTFileBase: public HCAL_HLX::TimeStamp{
    
  public:
    
    ROOTFileBase();
    ~ROOTFileBase();
    
    std::string GetFileName(){ return fileName_; }
    void SetFileName(const std::string& fileName);
    
    std::string GetOutputDir(){ return outputDir_; }
    void SetOutputDir(std::string dir){ outputDir_ = dir; }
    
    std::string CreateRunFileName(const unsigned int runNumber, const unsigned int firstSection);
    std::string CreateLSFileName(const unsigned int runNumber, const unsigned int sectionNumber);
    
  protected:

    unsigned int fileCounter_;
    
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
    
    std::string fileName_;
    std::string outputDir_;        
    std::string outputFilePrefix_;
      
    void CreateTree(const HCAL_HLX::LUMI_SECTION &localSection);
    void FillTree(const HCAL_HLX::LUMI_SECTION &localSection);
    void CloseTree();
    void InsertInformation();
    
    template< class T >
      void MakeBranch(const T &in, T **out, int HLXNum);
    
  };
}
#endif
