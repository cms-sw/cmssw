#ifndef FUSHMRAWCELL_H
#define FUSHMRAWCELL_H 1


namespace evf {

  class FUShmRawCell
  {
  public:
    //
    // construction/destruction
    //
    FUShmRawCell(unsigned int payloadSize);
    ~FUShmRawCell();
    
    
    //
    // member functions
    //
    void           initialize(unsigned int index);
    
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

    unsigned int   payloadSize()                  const { return payloadSize_; }
    unsigned char* payloadAddr()                  const;
    
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
    
    void           printState();
    
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
    static unsigned int size(unsigned int payloadSize);
    
    
  private:
    //
    // member data
    //
    unsigned int fuResourceId_;
    unsigned int buResourceId_;
    unsigned int evtNumber_;
    unsigned int nSkip_;
    unsigned int state_;
    unsigned int payloadSize_;
    unsigned int nFed_;
    unsigned int nSuperFrag_;
    unsigned int fedSizeOffset_;
    unsigned int fedOffset_;
    unsigned int superFragSizeOffset_;
    unsigned int superFragOffset_;
    unsigned int payloadOffset_;
    unsigned int payloadPosition_;
    
  };

  
} // namespace evf


#endif
