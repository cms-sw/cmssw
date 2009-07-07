#ifndef __ROOTFILEBASE_H__
#define __ROOTFILEBASE_H__

/*

Adam Hunt - Princeton University
ahunt@princeton.edu

*/
#include "RecoLuminosity/TCPReceiver/interface/ICTypeDefs.hh"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

#include <string>
#include <sstream>

namespace HCAL_HLX{

  class ROOTFileBase{
    
  public:
    
    ROOTFileBase();
    virtual ~ROOTFileBase();
    
    void SetDir( const std::string &dir );

    void SetFileType( const std::string &type );
    void SetDate( const std::string &date);

    void SetFileName(const HCAL_HLX::LUMI_SECTION &lumiSection);

    void SetFileName(unsigned int runNumber, 
		     unsigned int sectionNumber);

    std::string GetDir(){ return dirName_; }
    std::string GetFileName(){ return fileName_; }
    
    void SetEtSumOnly( bool bEtSumOnly );
    
  protected:

    void Init();
    void CleanUp();

    virtual void CreateTree() = 0;

    HCAL_HLX::LUMI_SECTION        *lumiSection_;

    HCAL_HLX::LUMI_SECTION_HEADER *Header_;
    HCAL_HLX::LUMI_SUMMARY        *Summary_;
    HCAL_HLX::LUMI_DETAIL         *Detail_;
    
    HCAL_HLX::ET_SUM_SECTION      *EtSumPtr_[36];
    HCAL_HLX::OCCUPANCY_SECTION   *OccupancyPtr_[36];
    HCAL_HLX::LHC_SECTION         *LHCPtr_[36];
    
    HCAL_HLX::LUMI_THRESHOLD      *Threshold_;
    HCAL_HLX::LEVEL1_TRIGGER      *L1Trigger_;
    HCAL_HLX::HLT                 *HLT_;
    HCAL_HLX::TRIGGER_DEADTIME    *TriggerDeadtime_;
    HCAL_HLX::LUMI_HF_RING_SET    *RingSet_;

    std::string filePrefix_;    
    std::string fileName_;
    std::string dirName_;        

    bool bEtSumOnly_;
    std::string date_;
    std::string fileType_;
  };
}
#endif
