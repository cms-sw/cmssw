#ifndef EventFilter_L1TRawToDigi_UnpackerCollections_h
#define EventFilter_L1TRawToDigi_UnpackerCollections_h

namespace edm {
  class Event;
}

namespace l1t {
  class UnpackerCollections {
  public:
    UnpackerCollections(edm::Event& e) : event_(e){};
    virtual ~UnpackerCollections(){};

  protected:
    edm::Event& event_;
  };
}  // namespace l1t

#endif
