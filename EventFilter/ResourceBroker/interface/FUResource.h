#ifndef FURESOURCE_H
#define FURESOURCE_H 1


#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/ShmBuffer/interface/FUShmRawCell.h"
#include "EventFilter/Utilities/interface/Exception.h"

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
    FUResource(UInt_t fuResourceId,log4cplus::Logger logger);
    virtual ~FUResource();
    
    
    //
    // member functions
    //
    void   allocate(FUShmRawCell* shmCell);
    void   release();   

    void   process(MemRef_t* bufRef);
    void   processDataBlock(MemRef_t* bufRef) throw (evf::Exception);
    void   checkDataBlockPayload(MemRef_t* bufRef) throw (evf::Exception);
    void   appendBlockToSuperFrag(MemRef_t* bufRef);
    
    void   superFragSize()        throw (evf::Exception);
    void   fillSuperFragPayload() throw (evf::Exception);
    void   findFEDs()             throw (evf::Exception);
    
    void   releaseSuperFrag();

    static
    void   doFedIdCheck(bool doFedIdCheck) { doFedIdCheck_=doFedIdCheck; }
    void   doCrcCheck(bool doCrcCheck)     { doCrcCheck_  =doCrcCheck; }
    
    bool   crcBeingChecked() { return doCrcCheck_; }

    bool   fatalError()   const { return fatalError_; }
    bool   isAllocated()  const { return 0!=shmCell_; }
    bool   isComplete()   const;
        
    UInt_t fuResourceId() const { return fuResourceId_; }
    UInt_t buResourceId() const { return buResourceId_; }
    UInt_t evtNumber()    const { return evtNumber_; }
    
    UInt_t nbErrors(bool reset=true);
    UInt_t nbCrcErrors(bool reset=true);
    UInt_t nbBytes(bool reset=true);

    evf::FUShmRawCell* shmCell() { return shmCell_; }
    
    
  private:
    //
    // member data
    //
    log4cplus::Logger log_;
    
    static
    bool      doFedIdCheck_;
    bool      doCrcCheck_;
    bool      fatalError_;

    UInt_t    fuResourceId_;
    UInt_t    buResourceId_;
    UInt_t    evtNumber_;
    
    MemRef_t* superFragHead_;
    MemRef_t* superFragTail_;

    UInt_t    eventPayloadSize_;
    UInt_t    nFedMax_;
    UInt_t    nSuperFragMax_;
    
    UInt_t    iBlock_;
    UInt_t    nBlock_;
    UInt_t    iSuperFrag_;
    UInt_t    nSuperFrag_;
        
    UInt_t    nbErrors_;
    UInt_t    nbCrcErrors_;
    UInt_t    nbBytes_;

    UInt_t    fedSize_[1024];
    UInt_t    superFragSize_;
    UInt_t    eventSize_;
    
    evf::FUShmRawCell* shmCell_;
    
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
