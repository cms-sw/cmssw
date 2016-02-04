#ifndef _HCAL_HLX_ROOTFILETRANSFER_H_
#define _HCAL_HLX_ROOTFILETRANSFER_H_

#include <string>

namespace HCAL_HLX{

  class ROOTFileTransfer{
  public:
    ROOTFileTransfer();
    ~ROOTFileTransfer();
     
    void SetFileName( const std::string &fileName ){ fileName_ = fileName; }
    void SetInputDir( const std::string &dirName ){ dirName_ = dirName; }
    void SetEtSumOnly( const bool &bEtSumOnly ){ bEtSumOnly_ = bEtSumOnly; }
    void SetFileType( const std::string &fileType );

    int TransferFile();
  private:
    std::string fileName_;
    std::string dirName_;
    std::string fileType_;

    bool bEtSumOnly_;
  };
}


#endif
