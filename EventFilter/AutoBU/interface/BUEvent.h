#ifndef BUEVENT_H
#define BUEVENT_H 1


namespace evf
{

  class BUEvent
  {
  public:
    //
    // construction/destruction
    //
    BUEvent(unsigned int buResourceId,unsigned int bufferSize=0x400000);
    virtual ~BUEvent();
    

    //
    // member functions
    //
    void           initialize(unsigned int evtNumber);
    
    bool           writeFed(unsigned int id,unsigned char* data,unsigned int size);
    bool           writeFedHeader(unsigned int i);
    bool           writeFedTrailer(unsigned int i);
    
    unsigned int   buResourceId()          const { return buResourceId_; }
    unsigned int   evtNumber()             const { return evtNumber_; }
    unsigned int   evtSize()               const { return evtSize_; }
    unsigned int   bufferSize()            const { return bufferSize_; }
    unsigned int   nFed()                  const { return nFed_; }
    unsigned int   fedId(unsigned int i)   const { return fedId_[i]; }
    unsigned int   fedSize(unsigned int i) const { return fedSize_[i]; }
    unsigned char* fedAddr(unsigned int i) const;
    
    static bool    computeCrc() { return computeCrc_; }
    static void    setComputeCrc(bool computeCrc) { computeCrc_=computeCrc; }

    void           dump();
    
    
  private:
    //
    // member data
    //
    unsigned int   buResourceId_;
    unsigned int   evtNumber_;
    unsigned int   evtSize_;
    unsigned int   bufferSize_;
    unsigned int   nFed_;
    unsigned int  *fedId_;
    unsigned int  *fedPos_;
    unsigned int  *fedSize_;
    unsigned char *buffer_;

    static bool    computeCrc_;
    
  };
  
  
} // namespace evf


#endif
