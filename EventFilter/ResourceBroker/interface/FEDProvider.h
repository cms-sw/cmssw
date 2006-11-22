#ifndef FEDPROVIDER_H
#define FEDPROVIDER_H 1


#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"


namespace evf {
  
  class FEDProvider
  {
  public:
    //
    // construction/destruction
    //
    FEDProvider();
    virtual ~FEDProvider();
    
    
    //
    // member functions
    //
    virtual FEDRawDataCollection* rqstEvent(unsigned int& evtNumber,
					    unsigned int& buResourceId)=0;
    
    virtual void sendDiscard(unsigned int buResourceId)=0;
    
    static  FEDProvider* instance();

  private:
    //
    // member data
    //
    static FEDProvider* instance_;
    
  };
  

} // namespace evf


//
// implementation of inline member functions
//

//______________________________________________________________________________
inline
evf::FEDProvider* evf::FEDProvider::instance()
{
  return instance_;
}


#endif
