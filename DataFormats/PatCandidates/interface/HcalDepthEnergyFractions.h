#ifndef DataFormats_PatCandidates_HcalDepthEnergyFractions_h
#define DataFormats_PatCandidates_HcalDepthEnergyFractions_h

namespace pat{

  //
  // Hcal depth energy fracrtion struct
  //
  class HcalDepthEnergyFractions {

  private:
    //do not store
    std::vector<float> fractions_;
    //store
    std::vector<uint8_t> fractionsI_;
  public:
    explicit HcalDepthEnergyFractions(std::vector<float> v):
                              fractions_(v),fractionsI_() { init(); }
    HcalDepthEnergyFractions():fractions_(),fractionsI_() { }
    
    // produce vector of uint8 vector from vector of float
    void init() {
      for (auto frac : fractions_) fractionsI_.push_back((uint8_t)(frac*200.));
    }

    // reset vector
    void reset(std::vector<float> v) {
      fractions_ = v;
      init();
    }

    // provide a full vector for each depth
    std::vector<float> fractions() {
      fractions_.clear();
      for (auto frac : fractionsI_) fractions_.push_back(float(frac)/200.);
      return fractions_;
    }

    // provide info for individual depth
    float fraction(unsigned int i) const {
      if (i<fractionsI_.size()) return float(fractionsI_[i])/200.;
      else return -1.;
    }
    
  };

}
 
#endif

