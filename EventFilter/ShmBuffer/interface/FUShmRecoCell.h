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
    unsigned int   rawCellIndex()const { return rawCellIndex_; }
    unsigned int   runNumber()   const { return runNumber_; }
    unsigned int   evtNumber()   const { return evtNumber_; }
    unsigned int   outModId()    const { return outModId_; }
    unsigned int   type()        const { return type_; }
    
    unsigned int   payloadSize() const { return payloadSize_; }
    unsigned char* payloadAddr() const;
    unsigned int   eventSize()   const { return eventSize_; }
    
    void           writeInitMsg(unsigned char *data,
				unsigned int   dataSize);
    
    void           writeEventData(unsigned int   rawCellIndex,
				  unsigned int   runNumber,
				  unsigned int   evtNumber,
				  unsigned int   outModId,
				  unsigned char *data,
				  unsigned int   dataSize);
    
    void           writeErrorEvent(unsigned int   rawCellIndex,
				   unsigned int   runNumber,
				   unsigned int   evtNumber,
				   unsigned char *data,
				   unsigned int   dataSize);
    
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
    unsigned int rawCellIndex_;
    unsigned int runNumber_;
    unsigned int evtNumber_;
    unsigned int outModId_;
    unsigned int type_;
    unsigned int payloadSize_;
    unsigned int payloadOffset_;
    unsigned int eventSize_;
    
  };

  
} // namespace evf


#endif
