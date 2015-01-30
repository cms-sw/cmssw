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
    // go to next clusters, merge adjacent clusters (default method)
    Phase2TrackerFEDZSChannelUnpacker& operator ++ ();
    Phase2TrackerFEDZSChannelUnpacker& operator ++ (int);
    inline int clusterX()    const { return clusterx_; } 
    inline int clusterY()    const { return clustery_; }
    inline int clusterSize() const { return clustersize_; }
    virtual int unMergedX()    const = 0; 
    virtual int unMergedY()    const = 0;
    virtual int unMergedSize() const = 0;
  protected:
    // merge next clusters if adjacent
    void Merge(); 
    // go to next cluster without merging adjacent clusters
    Phase2TrackerFEDZSChannelUnpacker& advance();
    // raw methods (bitwise operations)
    virtual uint8_t rawX()    const = 0;
    virtual uint8_t rawSize() const = 0;
    virtual uint8_t chipId()  const = 0;
    virtual bool gluedToNextCluster() const = 0;
    const uint8_t* data_;
    uint16_t currentOffset_; // caution : this is in bits, not bytes
    uint16_t clustersLeft_;
    uint8_t clusterdatasize_;
    int clusterx_;
    int clustery_;
    int clustersize_;
  };

  inline Phase2TrackerFEDZSChannelUnpacker::Phase2TrackerFEDZSChannelUnpacker(const Phase2TrackerFEDChannel& channel)
    : data_(channel.data())
  {
      currentOffset_ = channel.offset() * 8 + channel.bitoffset();
  }

  void Phase2TrackerFEDZSChannelUnpacker::Merge()
  {
    clusterx_     = unMergedX();
    clustery_     = unMergedY();
    clustersize_  = unMergedSize();
    while(gluedToNextCluster())
    {
      std::cout << "Cluster " << (uint16_t)unMergedX() << " " << (uint16_t)unMergedSize() << " on chip " << (uint16_t)chipId() << " has to be merged with next one : ";
      this->advance();
      clustersize_ += unMergedSize();
      std::cout << (uint16_t)unMergedX() << " " << (uint16_t)unMergedSize() << " on chip " << (uint16_t)chipId() << std::endl;
    }
  }

  inline Phase2TrackerFEDZSChannelUnpacker& Phase2TrackerFEDZSChannelUnpacker::operator ++ ()
  {
    currentOffset_ += clusterdatasize_; 
    clustersLeft_--;
    Merge();
    return (*this);
  }

  Phase2TrackerFEDZSChannelUnpacker& Phase2TrackerFEDZSChannelUnpacker::advance()
  {
    currentOffset_ += clusterdatasize_;
    clustersLeft_--;
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
              Merge();
          }
          inline uint8_t rawX()     const { return (uint8_t)read_n_at_m(data_,8,3+currentOffset_); }
          inline int Plane()        const { return rawX()%2; }
          inline uint8_t rawSize()  const { return (uint8_t)read_n_at_m(data_,3,currentOffset_)+1; }
          inline uint8_t chipId()   const { return (uint8_t)read_n_at_m(data_,4,11+currentOffset_); }
          int unMergedX() const; 
          int unMergedY() const;
          inline int unMergedSize() const { return rawSize(); } 
          bool gluedToNextCluster() const;
          Phase2TrackerFEDZSSon2SChannelUnpacker next() const;
  };

  int Phase2TrackerFEDZSSon2SChannelUnpacker::unMergedX() const
  {
    uint8_t id = chipId();
    if(id>7) id -= 8;
    return (STRIPS_PER_CBC*id + rawX())/2;
  }

  int Phase2TrackerFEDZSSon2SChannelUnpacker::unMergedY() const
  {
    return (chipId() > 7)?1:0;
  }

  bool Phase2TrackerFEDZSSon2SChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ <= 1 or this->next().unMergedY() != unMergedY() or this->next().Plane() != Plane()) return false; 
    if (this->next().unMergedX() == unMergedX() + size) 
    {
      if(size == 8 or (unMergedX() + size)%(STRIPS_PER_CBC/2) == 0) return true;
    }
    return false;
  }

  Phase2TrackerFEDZSSon2SChannelUnpacker Phase2TrackerFEDZSSon2SChannelUnpacker::next() const
  {
    Phase2TrackerFEDZSSon2SChannelUnpacker next(*this);
    next.advance();
    return next;
  }

  class Phase2TrackerFEDZSSonPSChannelUnpacker : public Phase2TrackerFEDZSChannelUnpacker
  {
      public:
          Phase2TrackerFEDZSSonPSChannelUnpacker(const Phase2TrackerFEDChannel& channel) : Phase2TrackerFEDZSChannelUnpacker(channel)
          {
              clusterdatasize_ = S_CLUSTER_SIZE_BITS;  
              clustersLeft_ = channel.length()*8/clusterdatasize_;
              Merge();
          }
          inline uint8_t rawX()     const { return (uint8_t)read_n_at_m(data_,8,3+currentOffset_); }
          inline uint8_t rawSize()  const { return (uint8_t)read_n_at_m(data_,3,currentOffset_)+1; }
          inline uint8_t chipId()   const { return (uint8_t)read_n_at_m(data_,4,11+currentOffset_); }
          int unMergedX() const;
          int unMergedY() const;
          inline int unMergedSize() const { return rawSize(); } 
          bool gluedToNextCluster() const;
          Phase2TrackerFEDZSSonPSChannelUnpacker next() const;
  };
  
  int Phase2TrackerFEDZSSonPSChannelUnpacker::unMergedX() const
  {
    uint8_t id = chipId();
    if(id>7) id -= 8;
    return PS_ROWS*id + rawX();
  }

  int Phase2TrackerFEDZSSonPSChannelUnpacker::unMergedY() const
  {
    return (chipId() > 7)?1:0;
  }

  Phase2TrackerFEDZSSonPSChannelUnpacker Phase2TrackerFEDZSSonPSChannelUnpacker::next() const
  {
    Phase2TrackerFEDZSSonPSChannelUnpacker next(*this);
    next.advance();
    return next;
  }

  bool Phase2TrackerFEDZSSonPSChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ <= 1 or this->next().unMergedY() != unMergedY()) { return false; }
    if (this->next().unMergedX() == unMergedX() + size) 
    {
      if(size == 8 or (unMergedX() + size)%PS_ROWS == 0) return true;
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
              Merge();
          }
          inline uint8_t rawX()        const { return (uint8_t)read_n_at_m(data_,7,7+currentOffset_);  }
          inline uint8_t rawY()        const { return (uint8_t)read_n_at_m(data_,4,3+currentOffset_); }
          inline uint8_t rawSize()     const { return (uint8_t)read_n_at_m(data_,3,currentOffset_)+1; }
          inline uint8_t chipId()      const { return (uint8_t)read_n_at_m(data_,4,14+currentOffset_); }
          int unMergedX() const; 
          int unMergedY() const; 
          inline int unMergedSize() const { return rawSize(); } 
          bool gluedToNextCluster() const;
          Phase2TrackerFEDZSPonPSChannelUnpacker next() const;
  };

  int Phase2TrackerFEDZSPonPSChannelUnpacker::unMergedX() const
  {
    uint8_t id = chipId();
    if(id>7) id -= 8;
    return PS_ROWS*id + rawX();
  }

  int Phase2TrackerFEDZSPonPSChannelUnpacker::unMergedY() const
  {
    return (chipId() > 7)?(rawY()+PS_COLS/2):rawY();
  }

  Phase2TrackerFEDZSPonPSChannelUnpacker Phase2TrackerFEDZSPonPSChannelUnpacker::next() const
  {
    Phase2TrackerFEDZSPonPSChannelUnpacker next(*this);
    next.advance();
    return next;
  }

  bool Phase2TrackerFEDZSPonPSChannelUnpacker::gluedToNextCluster() const
  {
    uint8_t size = rawSize();
    if (clustersLeft_ <= 1 or this->next().unMergedY() != unMergedY()) { return false; }
    if (this->next().unMergedX() == unMergedX() + size) 
    {
      if(size == 8 or (unMergedX() + size)%PS_ROWS == 0) return true;
    }
    return false; 
  }

} // end of Phase2Tracker namespace

#endif // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H

