#ifndef FUSHMCLIENT_H
#define FUSHMCLIENT_H


#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"

#include <vector>


namespace evf {
  

  class FUShmClient
  {
  public:
    //
    // construction/destruction
    //
    FUShmClient(FUShmBuffer* buffer);
    ~FUShmClient();
    
    
    //
    // member functions
    //
    unsigned int readNext(std::vector<std::vector<unsigned char> > &feds);

    double       crashPrb() const        { return crashPrb_; }
    void         setCrashPrb(double prb) { crashPrb_=prb;    }
    void         setSleep(int sec)       {  sleep_ = sec;    }
    
  private:
    //
    // member data
    //
    FUShmBuffer* buffer_;
    double       crashPrb_;
    int          sleep_;
  };
  
  
} // namespace evf


#endif
