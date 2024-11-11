#ifndef EventFilter_Phase2TrackerRawToDigi_TrackerHeader_H
#define EventFilter_Phase2TrackerRawToDigi_TrackerHeader_H


class TrackerHeader {
    public:
    
        std::vector<uint32_t> values_{std::vector<uint32_t>(4,0)};
    
        void setValue(std::vector<uint32_t>& newValues) {
          values_ = newValues;
        }
    
        void printValues() const {
          for (size_t i = 0; i < values_.size(); ++i) {
            std::cout << "TrackerHeader[" << i << "]: " << values_[i] << "   " << std::bitset<32>(values_[i]) <<  std::endl;
          }
        }   
        void printValue(size_t i) const {
          std::cout << "TrackerHeader[" << i << "]: " << values_[i] << "   " << std::bitset<32>(values_[i]) <<  std::endl;
        }   
};

#endif