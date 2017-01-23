#ifndef __DataFormats_L1THGCal_HGCFETriggerDigi_h__
#define __DataFormats_L1THGCal_HGCFETriggerDigi_h__

#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <vector>

/*******
 *
 * Class: l1t::HGCFETriggerDigi
 * Author: L. Gray (FNAL)
 * Date: 26 July, 2015
 *
 * An abstract representation of an HGC Front-End Trigger data payload.
 * The user of the class (some trigger algorithm) provides a codec in 
 * order to interpret the data payload held within class. This implementation
 * is chosen since final form of the HGC trigger primitives was not known
 * in July 2015.
 *
 * The CODEC class used below *must* implement the following interfaces:
 * 
 * CODEC::getCodecType() const -- returns an unsigned char indexing the codec
 *
 * CODEC::encode(const DATA&) const 
 *     -- encodes the data payload described by DATA into std::vector<bool>
 *
 * DATA CODEC::decode(const std::vector<bool>& data) const 
 *     -- decodes a std::vector<bool> into DATA
 * 
 * DATA must implement the following interfaces:
 * DATA::operator<<(std::ostream& out) const 
 *     -- prints the contents of the formatted data
 *
 *******/

#include "DataFormats/DetId/interface/DetId.h"

namespace l1t {  
  constexpr unsigned char hgcal_bad_codec(0xff);
  class HGCFETriggerDigi {
  public:
    
    typedef std::vector<bool> data_payload;
    typedef uint32_t key_type; 

    HGCFETriggerDigi() : codec_((unsigned char)0xffff) {}
    ~HGCFETriggerDigi() {}
    
    //detector id information
    uint32_t id() const { return detid_; } // for edm::SortedCollection
    template<typename IDTYPE>
    IDTYPE getDetId() const { return IDTYPE(detid_); }
    template<typename IDTYPE>
    void   setDetId(const IDTYPE& id) { detid_ = id.rawId(); }

    // encoding and decoding
    unsigned char getWhichCodec() const { return codec_; }
    
    template<typename CODEC,typename DATA>
    void encode(const CODEC& codec, const DATA& data) {
      if( codec_ != hgcal_bad_codec ) {
         throw cms::Exception("HGCTriggerAlreadyEncoded")
           << "HGC Codec and data already set with codec: " 
           << std::hex << codec_ << std::dec;
      }
      codec_ = codec.getCodecType();
      data_ = codec.encode(data);
    }

    template<typename CODEC, typename DATA>
    void decode(const CODEC& codec, DATA& data) const {
      if( codec_ != codec.getCodecType() ){
        throw cms::Exception("HGCTriggerWrongCodec")
          << "Wrong HGC codec: " << std::hex << codec.getCodecType() 
          << " given to data encoded with HGC codec type: " 
          << codec_ << std::dec;
      }
      data = codec.decode(data_, detid_);
    }
 
    void print(std::ostream& out) const;
    template<typename CODEC>
    void print(const CODEC& codec, std::ostream& out) const;

  private:
    uint32_t detid_;      // save in the abstract form
    unsigned char codec_; // 0xff is special and means no encoder
    data_payload  data_;    
  };  

  template<typename CODEC>
  void HGCFETriggerDigi::print(const CODEC& codec, std::ostream& out) const {
    if( codec_ != codec.getCodecType() ){
      throw cms::Exception("HGCTriggerWrongCodec")
        << "Wrong HGC codec: " << codec.getCodecType() 
        << " given to data encoded with HGC codec type: " 
        << codec_;
    }
    out << codec.decode(data_, detid_);
    out << std::endl << " decoded from: " << std::endl;
    this->print(out);
  }
}

#endif
