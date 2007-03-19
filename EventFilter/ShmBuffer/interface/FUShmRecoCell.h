#ifndef FUSHMRECOCELL_H
#define FUSHMRECOCELL_H 1


namespace evf {

  class FUShmRecoCell
  {
  public:
    //
    // construction/destruction
    //
    FUShmRecoCell(unsigned int payloadSize);
    ~FUShmRecoCell();
    
    
    //
    // member functions
    //
    void           initialize(unsigned int index);
    
    unsigned int   index()       const { return index_; }
    unsigned int   evtNumber()   const { return evtNumber_; }
    
    bool           isEmpty()     const { return (state_==0); }
    bool           isWriting()   const { return (state_==1); }
    bool           isWritten()   const { return (state_==2); }
    bool           isSending()   const { return (state_==3); }
    bool           isSent()      const { return (state_==4); }

    unsigned int   payloadSize() const { return payloadSize_; }
    unsigned char* payloadAddr() const;
    unsigned int   eventSize()   const { return eventSize_; }

    void           writeData(unsigned char*data,unsigned int dataSize);
    
    void           setEvtNumber(unsigned int evt) { evtNumber_=evt; }
    
    void           setStateEmpty()   { state_=0;evtNumber_=0xffffffff; }
    void           setStateWriting() { state_=1; }
    void           setStateWritten() { state_=2; }
    void           setStateSending() { state_=3; }
    void           setStateSent()    { state_=4; }
    
    void           printState();
    void           clear();


    
    
    //
    // static member functions
    //
    static unsigned int size(unsigned int payloadSize);
    
    
  private:
    //
    // member data
    //
    unsigned int index_;
    unsigned int evtNumber_;
    unsigned int state_;
    unsigned int payloadSize_;
    unsigned int payloadOffset_;
    unsigned int eventSize_;
    
  };

  
} // namespace evf


#endif
