#ifndef FURESOURCE_H
#define FURESOURCE_H 1


#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/ResourceBroker/interface/FUShmBufferCell.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "extern/log4cplus/linuxx86/include/log4cplus/logger.h"
#include "toolbox/include/toolbox/mem/Reference.h"

#include <vector>


namespace evf {
  
  class FUResource
  {
  public:
    //
    // construction/destruction
    //
    FUResource(unsigned int fuResourceId,log4cplus::Logger logger);
    FUResource(FUShmBufferCell* eventBuffer, log4cplus::Logger logger);
    virtual ~FUResource();
    
    
    //
    // member functions
    //
    void   allocate();
    void   release();   

    void   process(MemRef_t* bufRef);
    void   processDataBlock(MemRef_t* bufRef) throw (evf::Exception);
    void   checkDataBlockPayload(MemRef_t* bufRef) throw (evf::Exception);
    void   appendBlockToSuperFrag(MemRef_t* bufRef);
    
    void   superFragSize()       throw (evf::Exception);
    void   fillSuperFragBuffer() throw (evf::Exception);
    void   findFEDs()            throw (evf::Exception);
    void   fillFEDs();
    
    void   releaseSuperFrag();

    void   doCrcCheck(bool doCrcCheck) { doCrcCheck_=doCrcCheck; }

    bool   fatalError()   const { return fatalError_; }
    bool   isComplete()   const;
        
    UInt_t fuResourceId() const { return fuResourceId_; }
    UInt_t buResourceId() const { return buResourceId_; }
    UInt_t evtNumber()    const { return evtNumber_; }
    
    UInt_t nbErrors(bool reset=true);
    UInt_t nbCrcErrors(bool reset=true);
    UInt_t nbBytes(bool reset=true);

    FEDRawDataCollection* fedData() { return fedData_; }
    
    
  private:
    //
    // member data
    //
    log4cplus::Logger log_;
    
    bool      doCrcCheck_;
    bool      fatalError_;

    UInt_t    fuResourceId_;
    UInt_t    buResourceId_;
    UInt_t    evtNumber_;
    
    MemRef_t* superFragHead_;
    MemRef_t* superFragTail_;

    UInt_t    nFedMax_;
    UInt_t    nSuperFragMax_;
    UInt_t    eventSizeMax_;
    
    UInt_t    iBlock_;
    UInt_t    nBlock_;
    UInt_t    iSuperFrag_;
    UInt_t    nSuperFrag_;
        
    UInt_t    nbErrors_;
    UInt_t    nbCrcErrors_;
    UInt_t    nbBytes_;

    UInt_t    superFragSize_;
    UInt_t    eventSize_;
    
    evf::FUShmBufferCell* eventBuffer_;
    
    FEDRawDataCollection* fedData_;
    
  };
  
  //
  // typedefs
  //
  typedef std::vector<FUResource*> FUResourceVec_t;

  
} // namespace evf


//
// implementation of inline functions
//
inline
bool evf::FUResource::isComplete() const
{
  return (nBlock_&&nSuperFrag_&&(iSuperFrag_==nSuperFrag_)&&(iBlock_==nBlock_));
}


#endif
