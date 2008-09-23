#ifndef ROOTSCHEMA_H
#define ROOTSCHEMA_H

/*

Author: Adam Hunt - Princeton University
Date: 2007-10-05

*/

#include "RecoLuminosity/TCPReceiver/interface/TimeStamp.h"
#include "RecoLuminosity/ROOTSchema/interface/FileToolKit.h"

#include <string>

namespace HCAL_HLX{

  class ROOTFileWriter;
  class ROOTFileMerger;
  class ROOTFileTransfer;
  class HTMLGenerator;

  struct LUMI_SECTION;

  class ROOTSchema: private TimeStamp, private FileToolKit{
    
  public:
    
    ROOTSchema();
    ~ROOTSchema();
    
    void SetLSDir(    const std::string &lsDir    );
    void SetMergeDir( const std::string &mergeDir );


    void SetMergeFiles(    const bool bMerge );

    void SetTransferFiles( const bool bTransfer );

    void SetFileType( const std::string &fileType );
    void SetHistoBins( const int NBins, const double XMin, const double XMax ); 
    bool ProcessSection(const HCAL_HLX::LUMI_SECTION &lumiSection);

    void SetWebDir(   const std::string &webDir   );
    void SetCreateWebPage( const bool bWBM );
    void EndRun();

  private:

    HCAL_HLX::ROOTFileMerger   *RFMerger_;
    HCAL_HLX::ROOTFileTransfer *RFTransfer_;
    HCAL_HLX::HTMLGenerator    *LumiHTML_;
    HCAL_HLX::ROOTFileWriter   *RFWriter_;
    
    unsigned int previousRun_;
    unsigned int firstSectionNumber_;
    unsigned int startTime_;

    bool bMerge_;
    bool bWBM_;
    bool bTransfer_;

    bool        bEtSumOnly_;
    std::string fileType_;

    std::string lsDir_;
    std::string mergeDir_;

    std::string dateDir_;
    std::string runDir_;

    // Setting a directory implies that you want to do something.
    
    // Making the file type "ET" implies true.
    void SetEtSumOnly(     const bool bEtSumOnly);
  };
}

#endif
