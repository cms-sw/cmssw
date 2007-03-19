#ifndef FUSHMDQMCELL_H
#define FUSHMDQMCELL_H 1


namespace evf {

  class FUShmDqmCell
  {
  public:
    //
    // construction/destruction
    //
    FUShmDqmCell(unsigned int payloadSize);
    ~FUShmDqmCell();
    
    
    //
    // member functions
    //
    void           initialize(unsigned int index);
    
    unsigned int   index()       const { return index_; }
    
    bool           isEmpty()     const { return (state_==0); }
    bool           isWriting()   const { return (state_==1); }
    bool           isWritten()   const { return (state_==2); }
    bool           isSending()   const { return (state_==3); }
    bool           isSent()      const { return (state_==4); }

    unsigned int   payloadSize() const { return payloadSize_; }
    unsigned char* payloadAddr() const;
    unsigned int   eventSize()   const { return eventSize_; }

    void           writeData(unsigned char*data,unsigned int dataSize);
    
    void           setStateEmpty()   { state_=0; }
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
    unsigned int state_;
    unsigned int payloadSize_;
    unsigned int payloadOffset_;
    unsigned int eventSize_;
    
  };

  
} // namespace evf


#endif
