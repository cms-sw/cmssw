#ifndef _HCAL_HLX_ROOTFILETRANSFER_H_
#define _HCAL_HLX_ROOTFILETRANSFER_H_

#include <string>

namespace HCAL_HLX{

  class ROOTFileTransfer{
  public:
    ROOTFileTransfer();
    ~ROOTFileTransfer();
     
    void SetFileName( std::string fileName );
    int TransferFile();

  private:
    std::string fileName_;

  };
}


#endif
