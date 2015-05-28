#ifndef DIGIHGCAL_HGCDATAFRAME_H
#define DIGIHGCAL_HGCDATAFRAME_H

#include <vector>
#include <ostream>

/**
   @class HGCDataFrame
   @short Readout digi for HGC
*/

template <class D, class S>
class HGCDataFrame {
public:

  /**
     @short key to sort the collection
   */
  typedef D key_type; 

  /**
     @short CTOR
  */
  HGCDataFrame() : id_(0), maxSampleSize_(6)                      { data_.resize(maxSampleSize_); }
  explicit HGCDataFrame(const D& id) : id_(id), maxSampleSize_(6) { data_.resize(maxSampleSize_); }

  /**
    @short det id
  */
  const D& id() const { return id_; }
    
  /** 
    @short total number of samples in the digi 
  */
  int size() const { return data_.size() & 0xf; }

  /**
     @short allow to set size
   */
  void resize(size_t s) { data_.resize(s); }

  /**
     @short assess/set specific samples
  */
  const S& operator[](int i) const { return data_[i]; }
  const S& sample(int i)     const { return data_[i]; }
  void setSample(int i, const S &sample) { if(i<(int)data_.size()) data_[i]=sample; }


private:

  //collection of samples
  std::vector<S> data_;
  
  // det id for this data frame
  D id_;

  //number of samples and maximum available
  size_t maxSampleSize_;

};

#endif
