#ifndef FUSHMSERVER_H
#define FUSHMSERVER_H 1


#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"


namespace evf {

  
  class FUShmServer
  {
  public:
    //
    // construction/destruction
    //
    FUShmServer(FUShmBuffer* buffer);
    ~FUShmServer();
    
    
    //
    // member functions
    //
    unsigned int writeNext(unsigned char *data,
			   unsigned int   nFed,
			   unsigned int  *fedSize);
    
    
  private:
    //
    // member data
    //
    FUShmBuffer* buffer_;
    
  };


} // namespace evf


#endif
