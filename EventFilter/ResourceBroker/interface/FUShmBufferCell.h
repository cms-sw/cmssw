#ifndef FUSHMBUFFERCELL_H
#define FUSHMBUFFERCELL_H 1


#include <iostream>


namespace evf {

  class FUShmBufferCell
  {
  public:
    //
    // construction/destruction
    //
    FUShmBufferCell(unsigned int index,
		    unsigned int bufferSize,
		    unsigned int nFed,
		    unsigned int nSuperFrag,
		    bool         ownsMemory=true);
    ~FUShmBufferCell();
    
    
    //
    // member functions
    //
    bool           ownsMemory()                   const { return ownsMemory_; }

    unsigned int   index()                        const { return fuResourceId_; }
    unsigned int   fuResourceId()                 const { return fuResourceId_; }
    unsigned int   buResourceId()                 const { return buResourceId_; }
    unsigned int   evtNumber()                    const { return evtNumber_; }
    unsigned int   nSkip()                        const { return nSkip_; }
    
    bool           isEmpty()                      const { return (state_==0); }
    bool           isWriting()                    const { return (state_==1); }
    bool           isWritten()                    const { return (state_==2); }
    bool           isProcessing()                 const { return (state_==3); }
    bool           isProcessed()                  const { return (state_==4); }
    bool           isDead()                       const { return (state_==5); }

    unsigned int   bufferSize()                   const { return bufferSize_; }
    unsigned char* bufferAddr()                   const;
    
    unsigned int   nFed()                         const { return nFed_; }
    unsigned int   fedSize(unsigned int i)        const;
    unsigned char* fedAddr(unsigned int i)        const;

    unsigned int   nSuperFrag()                   const { return nSuperFrag_; }
    unsigned int   superFragSize(unsigned int i)  const;
    unsigned char* superFragAddr(unsigned int i)  const;
    
    unsigned int   eventSize()                    const;

    void           setBuResourceId(unsigned int id) { buResourceId_=id; }
    void           setEvtNumber(unsigned int evt)   { evtNumber_   =evt; }

    void           skip()              { nSkip_++; }
    void           resetSkip()         { nSkip_=0; }
    
    void           setStateEmpty()     {state_=0;buResourceId_=evtNumber_=0xffffffff;}
    void           setStateWriting()   {state_=1;}
    void           setStateWritten()   {state_=2;}
    void           setStateProcessing(){state_=3;}
    void           setStateProcessed() {state_=4;}
    void           setStateDead()      {state_=5;}
    
    void           print_state();
    
    void           clear();
    void           print(int verbose=0) const;
    void           dump() const;
    
    unsigned int   readFed(unsigned int i,unsigned char*buffer) const;
    unsigned char* writeData(unsigned char*data,unsigned int dataSize);

    bool           markFed(unsigned int i,unsigned int size,unsigned char*addr);
    bool           markSuperFrag(unsigned int i,unsigned int size,unsigned char*addr);
    
    
    //
    // static member functions
    //
    static unsigned int size(unsigned int bufferSize,
			     unsigned int nFed,
			     unsigned int nSuperFrag);
    
    
  private:
    //
    // member data
    //
    bool         ownsMemory_;
    unsigned int fuResourceId_;
    unsigned int buResourceId_;
    unsigned int evtNumber_;
    unsigned int nSkip_;
    unsigned int state_;
    unsigned int bufferSize_;
    unsigned int nFed_;
    unsigned int nSuperFrag_;
    unsigned int fedSizeOffset_;
    unsigned int fedOffset_;
    unsigned int superFragSizeOffset_;
    unsigned int superFragOffset_;
    unsigned int bufferOffset_;
    unsigned int bufferPosition_;
    
  };

  
} // namespace evf


#endif
