/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors
 *  Leszek Grzanka (braciszek@gmail.com)
 ****************************************************************************/

#ifndef DataFormats_L1TotemRP_TotemRPCCBits
#define DataFormats_L1TotemRP_TotemRPCCBits

#include <iosfwd>
#include <iostream>
#include <cstdlib>
#include <bitset>

/**
 * TODO: describe
 **/
class TotemRPCCBits
{
  public:
    /// Construct from a packed id. It is required that the Detector part of
    /// id is Totem and the SubDet part is RP, otherwise an exception is thrown.
    explicit TotemRPCCBits(uint32_t id, std::bitset<16> bs) : id_(id)
    {
      setBS(bs);
    }

    TotemRPCCBits() : id_(0)
    { 
      reset();
    }

    void reset()
    {
      std::bitset<16> nullBitset;
      nullBitset.reset();
      setBS(nullBitset);
    }

    void setId(uint32_t id)
    {
      id_ = id;
    }

    inline uint32_t getId() const
    {
      return id_;
    }

    inline std::bitset<16> getBS() const
    {
      return bs_;
    };
  
    void setBS(const std::bitset<16> &bs)
    {
        bs_ = bs;
    };

  private:
    uint32_t id_;
    std::bitset<16> bs_;
};


inline bool operator<( const TotemRPCCBits& one, const TotemRPCCBits& other)
{
  if(one.getId() < other.getId())
    return true;
  else if(one.getId() == other.getId())
    return one.getBS().to_ulong() < other.getBS().to_ulong();
  else
    return false;
}

#endif
