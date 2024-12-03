#ifndef EventFilter_Phase2TrackerRawToDigi_ChannelsOffset_H
#define EventFilter_Phase2TrackerRawToDigi_ChannelsOffset_H

// Class to store the payload offsets of the various channels in the FedRawData collection
// as from the current outer tracker data format

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"

using namespace Phase2TrackerSpecifications;
using namespace Phase2DAQFormatSpecification;


class ChannelsOffset {
public:
    std::vector<uint32_t> values_;
    std::vector<uint16_t> offsetMap_{std::vector<uint16_t>(CICs_PER_SLINK,0)};

    void setValue(std::vector<uint32_t>& newValues) {
      values_ = newValues;
      fillOffsetMap();
    }
    
    void printValues() const {
      for (size_t i = 0; i < values_.size(); ++i) {
        std::cout << "ChannelsOffset[" << i << "]: " << values_[i] << "   " << std::bitset<N_BITS_PER_WORD>(values_[i]) <<  std::endl;
      }
    }   
    void printValue(size_t i) const {
      std::cout << "ChannelsOffset[" << i << "]: " << values_[i] << "   " << std::bitset<N_BITS_PER_WORD>(values_[i]) <<  std::endl;
    }   
    
    void fillOffsetMap(){
      for (size_t i = 0; i < CICs_PER_SLINK/2; ++i) {
        // extract the lower 16 bits by masking with 0xFFFF
       offsetMap_[i*2] = static_cast<uint16_t>(values_[i] & 0xFFFF);
        // extract the upper 16 bits by shifting right by 16
        offsetMap_[i*2+1] =  static_cast<uint16_t>(values_[i] >> 16) ; 
      }
    }

    uint16_t getOffsetForChannel(unsigned int iChannel){
      if (iChannel >= CICs_PER_SLINK) {
        throw cms::Exception("ChannelsOffset") << " iChannel " << iChannel << " too high";
      }
      return offsetMap_[iChannel];
    }

    void printMap() const {
      for (size_t i = 0; i < offsetMap_.size(); ++i) {
        std::cout << "offsetMap[" << i << "]: " << offsetMap_[i] << std::endl;
     }
    }   
};

#endif