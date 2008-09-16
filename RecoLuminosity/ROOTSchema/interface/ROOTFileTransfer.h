#ifndef _HCAL_HLX_ROOTFILETRANSFER_H_
#define _HCAL_HLX_ROOTFILETRANSFER_H_

#include <string>
#include "RecoLuminosity/TCPReceiver/interface/TimeStamp.h"

namespace HCAL_HLX{

  class ROOTFileTransfer: public TimeStamp{
  public:
    ROOTFileTransfer();
    ~ROOTFileTransfer();
     
    void SetFileName( std::string fileName );
    int TransferFile( );

  private:
    std::string fileName_;
    std::string dirName_;
  };
}


#endif
