#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDChannel.h"
#include <stdint.h>

namespace Phase2Tracker {

 class Phase2TrackerFEDZSChannelUnpacker
  {
  public:
    Phase2TrackerFEDZSChannelUnpacker(const Phase2TrackerFEDChannel& channel);
    virtual uint8_t clusterIndex() const = 0;
    bool hasData() const { return clustersLeft_; }
    virtual bool gluedToNextCluster() const = 0;
    Phase2TrackerFEDZSChannelUnpacker& operator ++ ();
    Phase2TrackerFEDZSChannelUnpacker& operator ++ (int);
  protected:
    virtual uint8_t rawIndex() const = 0;
    virtual uint8_t rawSize() const = 0;
    virtual uint8_t chipId() const = 0;
    void computePosAndSize(); 
    const uint8_t* data_;
    uint16_t currentOffset_; // caution : this is in bits, not bytes
    uint16_t clustersLeft_;
    uint8_t clusterdatasize_;
    uint8_t clusterindex_;
    uint8_t clustersize_;
  };

  // unpacker for ZS CBC data
  inline Phase2TrackerFEDZSChannelUnpacker::Phase2TrackerFEDZSChannelUnpacker(const Phase2TrackerFEDChannel& channel)
    : data_(channel.data())
  {
      currentOffset_ = channel.offset() * 8 + channel.bitoffset();
  }

  void Phase2TrackerFEDZSChannelUnpacker::computePosAndSize()
  {
    clusterindex_ = rawIndex();
    clustersize_  = rawSize();
    while(gluedToNextCluster())
    {
      std::cout << "Cluster " << clustersLeft_ << " " << (uint16_t)clusterindex_ << " " << (uint16_t)clustersize_ << " on chip " << (uint16_t)chipId() << " has to be merged with next one : ";
      currentOffset_ += clusterdatasize_;
      clustersLeft_--;
      clustersize_ += rawSize();
      std::cout << (uint16_t)rawIndex() << " " << (uint16_t)rawSize() << " on chip " << (uint16_t)chipId() << std::endl;
    }
  }

  inline Phase2TrackerFEDZSChannelUnpacker& Phase2TrackerFEDZSChannelUnpacker::operator ++ ()
  {
    currentOffset_ += clusterdatasize_; 
    clustersLeft_--;
    computePosAndSize();
    return (*this);
  }
/*
  bool Phase2TrackerFEDZSChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ == 0 or size < 8) { return false; }
    uint8_t nextpos = (uint8_t)read_n_at_m(data_,8,3+currentOffset_+clusterdatasize_);
    return (nextpos == rawIndex() + size);
  }
*/  
  inline Phase2TrackerFEDZSChannelUnpacker& Phase2TrackerFEDZSChannelUnpacker::operator ++ (int)
  {
    ++(*this); return *this;
  }

  class Phase2TrackerFEDZSSon2SChannelUnpacker : public Phase2TrackerFEDZSChannelUnpacker
  {
      public:
          Phase2TrackerFEDZSSon2SChannelUnpacker(const Phase2TrackerFEDChannel& channel) : Phase2TrackerFEDZSChannelUnpacker(channel)
          {
              clusterdatasize_ = S_CLUSTER_SIZE_BITS;
              clustersLeft_ = channel.length()*8/clusterdatasize_;
              computePosAndSize();
          }
          inline uint8_t rawIndex()     const { return (uint8_t)read_n_at_m(data_,8,3+currentOffset_); }
          inline uint8_t rawSize()      const { return (uint8_t)read_n_at_m(data_,3,currentOffset_)+1; }
          inline uint8_t chipId()       const { return (uint8_t)read_n_at_m(data_,4,11+currentOffset_); }
          inline uint8_t clusterIndex() const { return clusterindex_; }
          inline uint8_t clusterSize()  const { return clustersize_; }
          bool gluedToNextCluster() const;
  };

  bool Phase2TrackerFEDZSSon2SChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ <= 1 or size < 8) { return false; }
    uint8_t nextpos = (uint8_t)read_n_at_m(data_,8,3+currentOffset_+clusterdatasize_);
    return (nextpos == rawIndex() + size*2);
  }

  class Phase2TrackerFEDZSSonPSChannelUnpacker : public Phase2TrackerFEDZSChannelUnpacker
  {
      public:
          Phase2TrackerFEDZSSonPSChannelUnpacker(const Phase2TrackerFEDChannel& channel) : Phase2TrackerFEDZSChannelUnpacker(channel)
          {
              clusterdatasize_ = S_CLUSTER_SIZE_BITS;  
              clustersLeft_ = channel.length()*8/clusterdatasize_;
              computePosAndSize();
          }
          inline uint8_t rawIndex()     const { return (uint8_t)read_n_at_m(data_,8,3+currentOffset_); }
          inline uint8_t rawSize()      const { return (uint8_t)read_n_at_m(data_,3,currentOffset_)+1; }
          inline uint8_t chipId()       const { return (uint8_t)read_n_at_m(data_,4,11+currentOffset_); }
          inline uint8_t clusterIndex() const { return clusterindex_; }
          inline uint8_t clusterSize()  const { return clustersize_; }
          bool gluedToNextCluster() const;
  };

  bool Phase2TrackerFEDZSSonPSChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ <= 1 or size < 8) { return false; }
    uint8_t nextpos = (uint8_t)read_n_at_m(data_,8,3+currentOffset_+clusterdatasize_);
    return (nextpos == rawIndex() + size);
  }

  class Phase2TrackerFEDZSPonPSChannelUnpacker : public Phase2TrackerFEDZSChannelUnpacker
  {
      public:
          Phase2TrackerFEDZSPonPSChannelUnpacker(const Phase2TrackerFEDChannel& channel) : Phase2TrackerFEDZSChannelUnpacker(channel)
          {
              clusterdatasize_ = P_CLUSTER_SIZE_BITS;  
              clustersLeft_ = channel.length()*8/clusterdatasize_;
              computePosAndSize();
          }
          inline uint8_t rawIndex()     const { return (uint8_t)read_n_at_m(data_,7,7+currentOffset_);  }
          inline uint8_t rawSize()      const { return (uint8_t)read_n_at_m(data_,3,currentOffset_)+1; }
          inline uint8_t chipId()       const { return (uint8_t)read_n_at_m(data_,4,14+currentOffset_); }
          inline uint8_t clusterIndex() const { return clusterindex_; }
          inline uint8_t clusterSize()  const { return clustersize_; }
          inline uint8_t clusterZpos()  const { return (uint8_t)read_n_at_m(data_,4,3+currentOffset_); }
          bool gluedToNextCluster() const;
  };

  bool Phase2TrackerFEDZSPonPSChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ <= 1 or size < 8 or clusterZpos() != (uint8_t)read_n_at_m(data_,4,3+currentOffset_+clusterdatasize_)) { return false; }
    uint8_t nextpos = (uint8_t)read_n_at_m(data_,7,7+currentOffset_+clusterdatasize_);
    return (nextpos == rawIndex() + size);
  }

} // end of Phase2Tracker namespace

#endif // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H

