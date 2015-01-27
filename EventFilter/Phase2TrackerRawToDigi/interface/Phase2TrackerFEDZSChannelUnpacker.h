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
    bool hasData() const { return clustersLeft_; }
    Phase2TrackerFEDZSChannelUnpacker& operator ++ ();
    Phase2TrackerFEDZSChannelUnpacker& operator ++ (int);
    virtual int clusterX() const = 0; 
    virtual int clusterY() const = 0;
    void Merge(); 
    int mergedX(); 
    int mergedY();
    int mergedSize();
  protected:
    virtual uint8_t rawX() const = 0;
    virtual uint8_t rawSize() const = 0;
    virtual uint8_t chipId() const = 0;
    virtual bool gluedToNextCluster() const = 0;
    const uint8_t* data_;
    uint16_t currentOffset_; // caution : this is in bits, not bytes
    uint16_t clustersLeft_;
    uint8_t clusterdatasize_;
    int mergedx_;
    int mergedy_;
    int mergedsize_;
    bool merged_ = false;
  };

  inline Phase2TrackerFEDZSChannelUnpacker::Phase2TrackerFEDZSChannelUnpacker(const Phase2TrackerFEDChannel& channel)
    : data_(channel.data())
  {
      currentOffset_ = channel.offset() * 8 + channel.bitoffset();
  }

  void Phase2TrackerFEDZSChannelUnpacker::Merge()
  {
    mergedx_ = clusterX();
    mergedsize_  = rawSize();
    while(gluedToNextCluster())
    {
      std::cout << "Cluster " << (uint16_t)clusterX() << " " << (uint16_t)rawSize() << " on chip " << (uint16_t)chipId() << " has to be merged with next one : ";
      ++(*this);
      mergedsize_ += rawSize();
      std::cout << (uint16_t)clusterX() << " " << (uint16_t)rawSize() << " on chip " << (uint16_t)chipId() << std::endl;
    }
    merged_ = true;
  }

  int Phase2TrackerFEDZSChannelUnpacker::mergedX()
  {
    if (!merged_) Merge();
    return mergedx_;
  }

  int Phase2TrackerFEDZSChannelUnpacker::mergedY()
  {
    if (!merged_) Merge();
    return clusterY();
  }

  int Phase2TrackerFEDZSChannelUnpacker::mergedSize()
  {
    if (!merged_) Merge();
    return mergedsize_;
  }

  inline Phase2TrackerFEDZSChannelUnpacker& Phase2TrackerFEDZSChannelUnpacker::operator ++ ()
  {
    currentOffset_ += clusterdatasize_; 
    clustersLeft_--;
    merged_ = false;
    return (*this);
  }

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
          }
          inline uint8_t rawX()     const { return (uint8_t)read_n_at_m(data_,8,3+currentOffset_); }
          inline uint8_t rawSize()  const { return (uint8_t)read_n_at_m(data_,3,currentOffset_)+1; }
          inline uint8_t chipId()   const { return (uint8_t)read_n_at_m(data_,4,11+currentOffset_); }
          int clusterX() const; 
          int clusterY() const;
          bool gluedToNextCluster() const;
          Phase2TrackerFEDZSSon2SChannelUnpacker next() const;
  };

  int Phase2TrackerFEDZSSon2SChannelUnpacker::clusterX() const
  {
    uint8_t id = chipId();
    if(id>7) id -= 8;
    return (STRIPS_PER_CBC*id + rawX())/2;
  }

  int Phase2TrackerFEDZSSon2SChannelUnpacker::clusterY() const
  {
    return (chipId() > 7)?1:0;
  }

  bool Phase2TrackerFEDZSSon2SChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ <= 1 or this->next().clusterY() != clusterY()) return false; 
    if (this->next().clusterX() == clusterX() + size) 
    {
      if(size == 8 or (clusterX() + size)%(STRIPS_PER_CBC/2) == 0) return true;
    }
    return false;
  }

  Phase2TrackerFEDZSSon2SChannelUnpacker Phase2TrackerFEDZSSon2SChannelUnpacker::next() const
  {
    Phase2TrackerFEDZSSon2SChannelUnpacker next(*this);
    next++;
    return next;
  }

  class Phase2TrackerFEDZSSonPSChannelUnpacker : public Phase2TrackerFEDZSChannelUnpacker
  {
      public:
          Phase2TrackerFEDZSSonPSChannelUnpacker(const Phase2TrackerFEDChannel& channel) : Phase2TrackerFEDZSChannelUnpacker(channel)
          {
              clusterdatasize_ = S_CLUSTER_SIZE_BITS;  
              clustersLeft_ = channel.length()*8/clusterdatasize_;
          }
          inline uint8_t rawX()     const { return (uint8_t)read_n_at_m(data_,8,3+currentOffset_); }
          inline uint8_t rawSize()  const { return (uint8_t)read_n_at_m(data_,3,currentOffset_)+1; }
          inline uint8_t chipId()   const { return (uint8_t)read_n_at_m(data_,4,11+currentOffset_); }
          int clusterX() const;
          int clusterY() const;
          bool gluedToNextCluster() const;
          Phase2TrackerFEDZSSonPSChannelUnpacker next() const;
  };
  
  int Phase2TrackerFEDZSSonPSChannelUnpacker::clusterX() const
  {
    uint8_t id = chipId();
    if(id>7) id -= 8;
    return PS_ROWS*id + rawX();
  }

  int Phase2TrackerFEDZSSonPSChannelUnpacker::clusterY() const
  {
    return (chipId() > 7)?1:0;
  }

  Phase2TrackerFEDZSSonPSChannelUnpacker Phase2TrackerFEDZSSonPSChannelUnpacker::next() const
  {
    Phase2TrackerFEDZSSonPSChannelUnpacker next(*this);
    next++;
    return next;
  }

  bool Phase2TrackerFEDZSSonPSChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ <= 1 or this->next().clusterY() != clusterY()) { return false; }
    if (this->next().clusterX() == clusterX() + size) 
    {
      if(size == 8 or (clusterX() + size)%PS_ROWS == 0) return true;
    }
    return false; 
  }

  class Phase2TrackerFEDZSPonPSChannelUnpacker : public Phase2TrackerFEDZSChannelUnpacker
  {
      public:
          Phase2TrackerFEDZSPonPSChannelUnpacker(const Phase2TrackerFEDChannel& channel) : Phase2TrackerFEDZSChannelUnpacker(channel)
          {
              clusterdatasize_ = P_CLUSTER_SIZE_BITS;  
              clustersLeft_ = channel.length()*8/clusterdatasize_;
          }
          inline uint8_t rawX()        const { return (uint8_t)read_n_at_m(data_,7,7+currentOffset_);  }
          inline uint8_t rawSize()     const { return (uint8_t)read_n_at_m(data_,3,currentOffset_)+1; }
          inline uint8_t chipId()      const { return (uint8_t)read_n_at_m(data_,4,14+currentOffset_); }
          inline uint8_t rawY()        const { return (uint8_t)read_n_at_m(data_,4,3+currentOffset_); }
          int clusterX() const; 
          int clusterY() const; 
          bool gluedToNextCluster() const;
          Phase2TrackerFEDZSPonPSChannelUnpacker next() const;
  };

  int Phase2TrackerFEDZSPonPSChannelUnpacker::clusterX() const
  {
    uint8_t id = chipId();
    if(id>7) id -= 8;
    return PS_ROWS*id + rawX();
  }

  int Phase2TrackerFEDZSPonPSChannelUnpacker::clusterY() const
  {
    return (chipId() > 7)?(rawY()+PS_COLS/2):rawY();
  }

  Phase2TrackerFEDZSPonPSChannelUnpacker Phase2TrackerFEDZSPonPSChannelUnpacker::next() const
  {
    Phase2TrackerFEDZSPonPSChannelUnpacker next(*this);
    next++;
    return next;
  }

  bool Phase2TrackerFEDZSPonPSChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ <= 1 or this->next().clusterY() != clusterY()) { return false; }
    if (this->next().clusterX() == clusterX() + size) 
    {
      if(size == 8 or (clusterX() + size)%PS_ROWS == 0) return true;
    }
    return false; 
  }

} // end of Phase2Tracker namespace

#endif // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H

