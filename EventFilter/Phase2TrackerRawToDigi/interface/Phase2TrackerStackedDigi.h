#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"

namespace Phase2Tracker
{
  class stackedDigi {
    public:
      stackedDigi() {}
      stackedDigi(const SiPixelCluster *, STACK_LAYER, int);
      stackedDigi(int, int, int, STACK_LAYER, int);
      ~stackedDigi() {}
      bool operator<(stackedDigi) const ;
      inline STACK_LAYER getLayer() const { return layer_; }
      inline int getModuleType() const { return moduletype_; }
      inline int getRawX()   const { return rawx_; }
      inline int getRawY()   const { return rawy_; }
      inline int getDigiX()  const { return digix_; }
      inline int getDigiY()  const { return digiy_; }
      inline int getSizeX()  const { return sizex_; }
      inline int getSide()   const { return side_; }
      // get side and type, to map to concentrators (0 = S-left, 1 = S-right, 2 = P-left, 3 = P-right)
      inline int getSideType()   const { return side_ + 2*(1-moduletype_) + 2*layer_*moduletype_; }
      inline int getChipId() const { return chipid_; }
      inline int getStripsX() const { return (moduletype_ == 1)?PS_ROWS:(STRIPS_PER_CBC/2); }
      void setPosSizeX(int, int);
      bool shouldSplit() const;
      std::vector<stackedDigi> splitDigi() const;
    private:
      void calcchipid();
      int digix_, digiy_, sizex_;
      STACK_LAYER layer_;
      int moduletype_;
      int rawx_, rawy_;
      int side_;
      int chipid_;
  };

  stackedDigi::stackedDigi(const SiPixelCluster * digi, STACK_LAYER layer, int moduletype) : 
      digix_(digi->minPixelRow()), 
      digiy_(digi->minPixelCol()), 
      sizex_(digi->sizeX()),
      layer_(layer), 
      moduletype_(moduletype) 
  {
    calcchipid();
  }

  stackedDigi::stackedDigi(int digix, int digiy, int sizex, STACK_LAYER layer, int moduletype) :
    digix_(digix),
    digiy_(digiy),
    sizex_(sizex),
    layer_(layer),
     moduletype_(moduletype)
  {
    calcchipid();
  }

  bool stackedDigi::shouldSplit() const
  {
    if (getSizeX() > 8 or getDigiX()%getStripsX() + getSizeX() > getStripsX()) return true;
    return false;
  }

  std::vector<stackedDigi> stackedDigi::splitDigi() const
  {
    std::vector<stackedDigi> parts;
    if(shouldSplit())
    {
      int pos = getDigiX();
      int end = pos + getSizeX();
      int isize;
      while (pos < end)
      {
        int nextchip = (pos/getStripsX() + 1)*getStripsX();
        isize = std::min(std::min(8,end-pos),nextchip-pos);
        stackedDigi ndig(*this);
        ndig.setPosSizeX(pos,isize);
        parts.push_back(ndig);
        pos += isize;
      }
      // debug couts
      /*
      std::cout << "--- Split digi at " << getDigiX() << ", raw: " << getRawX() << ", length: " << getSizeX() << std::endl;
      for (auto id = parts.begin(); id < parts.end(); id++)
      {
        std::cout << " -- " << id->getDigiX() << ", raw:  " << id->getRawX() << " " << id->getSizeX() << std::endl;
      }
      std::cout << "--- End of split ---" << std::endl;
      */
      // end of debug
    }
    else
    {
      parts.push_back(*this);
    }
    return parts;
  }


  void stackedDigi::setPosSizeX(int pos, int size)
  {
    digix_ = pos;
    sizex_ = size;
    calcchipid();
  }

  void stackedDigi::calcchipid() 
  {
    int x = digix_;
    int y = digiy_;
    side_ = 0;
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
          side_ = 1;
          rawy_ = y - PS_COLS/2;
        }
        else
        {  
          rawy_ = y;
        }
        rawx_ = x%PS_ROWS;
      }
      else
      {
        // SonPS
        chipid_ = x/PS_ROWS;
        if (y > 0) 
        { 
          chipid_ += 8; 
          side_ = 1;
        }
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
      if (y > 0) 
      { 
        chipid_ += 8; 
        side_ = 1;
      }
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
