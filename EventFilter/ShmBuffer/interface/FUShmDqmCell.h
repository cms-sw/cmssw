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
    unsigned int   runNumber()   const { return runNumber_; }
    unsigned int   evtAtUpdate() const { return evtAtUpdate_; }
    unsigned int   folderId()    const { return folderId_; }
    unsigned int   fuProcessId() const { return fuProcessId_; }
    unsigned int   fuGuid()      const { return fuGuid_; }

    unsigned int   payloadSize() const { return payloadSize_; }
    unsigned char* payloadAddr() const;
    unsigned int   eventSize()   const { return eventSize_; }

    void           writeData(unsigned int   runNumber,
			     unsigned int   evtAtUpdate,
			     unsigned int   folderId,
			     unsigned int   fuProcessId,
			     unsigned int   fuGuid,
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
    unsigned int runNumber_;
    unsigned int evtAtUpdate_;
    unsigned int folderId_;
    unsigned int fuProcessId_;
    unsigned int fuGuid_;
    unsigned int payloadSize_;
    unsigned int payloadOffset_;
    unsigned int eventSize_;
    
  };

  
} // namespace evf


#endif
