#ifndef EventFilter_Phase2TrackerRawToDigi_ChannelsOffset_H
#define EventFilter_Phase2TrackerRawToDigi_ChannelsOffset_H

// Class to store the payload offsets of the various channels in the FedRawData collection
// as from the current outer tracker data format

constexpr int N_CHANNELS = 36;
constexpr int LINE_LENGTH = 32;

class ChannelsOffset {
public:
    std::vector<uint32_t> values_;
    std::vector<uint16_t> offsetMap_{std::vector<uint16_t>(N_CHANNELS,0)};

    void setValue(std::vector<uint32_t>& newValues) {
      values_ = newValues;
      fillOffsetMap();
    }
    
    void printValues() const {
      for (size_t i = 0; i < values_.size(); ++i) {
        std::cout << "ChannelsOffset[" << i << "]: " << values_[i] << "   " << std::bitset<LINE_LENGTH>(values_[i]) <<  std::endl;
      }
    }   
    void printValue(size_t i) const {
      std::cout << "ChannelsOffset[" << i << "]: " << values_[i] << "   " << std::bitset<LINE_LENGTH>(values_[i]) <<  std::endl;
    }   
    
    void fillOffsetMap(){
      for (size_t i = 0; i < N_CHANNELS/2; ++i) {
        // extract the lower 16 bits by masking with 0xFFFF
       offsetMap_[i*2] = static_cast<uint16_t>(values_[i] & 0xFFFF);
        // extract the upper 16 bits by shifting right by 16
        offsetMap_[i*2+1] =  static_cast<uint16_t>(values_[i] >> 16) ; 
      }
    }

    uint16_t getOffsetForChannel(unsigned int iChannel){
      if (iChannel >= N_CHANNELS) {
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