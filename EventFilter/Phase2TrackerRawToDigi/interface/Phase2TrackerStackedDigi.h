#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"

namespace Phase2Tracker
{
  class stackedDigi {
    public:
      stackedDigi() {}
      stackedDigi(const SiPixelCluster *, STACK_LAYER, int);
      ~stackedDigi() {}
      bool operator<(stackedDigi) const ;
      inline const SiPixelCluster * getDigi() const { return digi_; }
      inline STACK_LAYER getLayer() const { return layer_; }
      inline int getModuleType() const { return moduletype_; }
      inline int getRawX() const { return rawx_; }
      inline int getRawY() const { return rawy_; }
      inline int getChipId() const { return chipid_; }
    private:
      void calcchipid();
      const SiPixelCluster * digi_;
      STACK_LAYER layer_;
      int moduletype_;
      int rawx_;
      int rawy_;
      int chipid_;
  };

  stackedDigi::stackedDigi(const SiPixelCluster * digi, STACK_LAYER layer, int moduletype) : digi_(digi), layer_(layer), moduletype_(moduletype) 
  {
    calcchipid();
  }

  void stackedDigi::calcchipid() 
  {
    int x = digi_->minPixelRow();
    int y = digi_->minPixelCol();
    if (moduletype_ == 1)
    {
      if(layer_ == LAYER_INNER)
      {
        // PonPS
        chipid_ = x/PS_ROWS;
        // which side ?
        if (y >= PS_COLS/2)
        {
          chipid_ += 8;
          rawy_ = y - PS_COLS/2;
        }
        rawx_ = x%PS_ROWS;
      }
      else
      {
        // SonPS
        chipid_ = x/PS_ROWS;
        if (y > 0) { chipid_ += 8; }
        rawx_ = x%PS_ROWS;
        rawy_ = y;
      }
    }
    else
    {
      x *= 2;
      if(layer_ == LAYER_OUTER) { x += 1; }
      chipid_ = x/STRIPS_PER_CBC;
      // which side ? 
      if (y > 0) { chipid_ += 8; }
      rawx_ = x%STRIPS_PER_CBC;
      rawy_ = y;
    }
  }

  bool stackedDigi::operator<(const stackedDigi d2) const
  {
    // in PS modules, P should always come before S, independantly of the chipid
    if(moduletype_ == 1)
    {
      if(layer_ < d2.getLayer()) { return true;  }
      if(layer_ > d2.getLayer()) { return false; }
    }
    if (chipid_ <  d2.getChipId()) { return true; }
    if (chipid_ == d2.getChipId() and rawx_ < d2.getRawX() ) { return true; }
    return false;
  }
}
