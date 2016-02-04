#ifndef __ROOTFILEBASE_H__
#define __ROOTFILEBASE_H__

/*

Adam Hunt - Princeton University
ahunt@princeton.edu

*/

#include <string>
#include <sstream>

namespace HCAL_HLX{

  struct LUMI_SECTION;
  
  struct LUMI_SECTION_HEADER;
  struct LUMI_SUMMARY;
  struct LUMI_DETAIL;
  
  struct ET_SUM_SECTION;
  struct OCCUPANCY_SECTION;
  struct LHC_SECTION;

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
    
    std::string filePrefix_;    
    std::string fileName_;
    std::string dirName_;        

    bool bEtSumOnly_;
    std::string date_;
    std::string fileType_;
  };
}
#endif
