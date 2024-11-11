#ifndef EventFilter_Phase2TrackerRawToDigi_ChannelsOffset_H
#define EventFilter_Phase2TrackerRawToDigi_ChannelsOffset_H


class ChannelsOffset {
public:
    std::vector<uint32_t> values_;
    std::vector<uint16_t> offsetMap_{std::vector<uint16_t>(36,0)};

    void setValue(std::vector<uint32_t>& newValues) {
      values_ = newValues;
      fillOffsetMap();
    }
    
    void printValues() const {
      for (size_t i = 0; i < values_.size(); ++i) {
        std::cout << "ChannelsOffset[" << i << "]: " << values_[i] << "   " << std::bitset<32>(values_[i]) <<  std::endl;
      }
    }   
    void printValue(size_t i) const {
      std::cout << "ChannelsOffset[" << i << "]: " << values_[i] << "   " << std::bitset<32>(values_[i]) <<  std::endl;
    }   
    
    void fillOffsetMap(){
      for (size_t i = 0; i < 18; ++i) {
// //       for (size_t i = 0; i < values_.size(); ++i) {
        // extract the lower 16 bits by masking with 0xFFFF
       offsetMap_[i*2] = static_cast<uint16_t>(values_[i] & 0xFFFF);
        // extract the upper 16 bits by shifting right by 16
        offsetMap_[i*2+1] =  static_cast<uint16_t>(values_[i] >> 16) ; 
      }
    }

    uint16_t getOffsetForChannel(unsigned int iChannel){
      if (iChannel > 35) {
        throw cms::Exception("Phase2TClusterProducer") << " iChannel " << iChannel << " too high";
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