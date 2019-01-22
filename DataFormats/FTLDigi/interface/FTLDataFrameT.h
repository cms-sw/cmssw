#ifndef DIGIFTL_FTLDATAFRAMET_H
#define DIGIFTL_FTLDATAFRAMET_H

#include <vector>
#include <ostream>
#include <iostream>

/**
   @class FTLDataFrameT
   @short Readout digi for HGC
*/

template <class D, class S, class DECODE>
  class FTLDataFrameT {
 public:
  
  /**
     @short key to sort the collection
  */
  typedef D key_type; 
  
  /**
     @short CTOR
  */
  FTLDataFrameT() : id_(0), maxSampleSize_(15)             { data_.resize(maxSampleSize_); }
  FTLDataFrameT(const D& id) : id_(id), maxSampleSize_(15) { data_.resize(maxSampleSize_); }
  FTLDataFrameT(const FTLDataFrameT& o) : data_(o.data_), id_(o.id_), maxSampleSize_(o.maxSampleSize_) { }
  
  /**
    @short det id
  */
  const D& id() const { return id_; }

  /**
   @short row
   */
   const int row() const { return DECODE::row(id_,data_); }

  /**
   @short column
   */
   const int column() const { return DECODE::col(id_,data_); }
    
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
  void print(std::ostream &out=std::cout)
  {
    for(size_t i=0; i<data_.size(); i++)
      {
	out << "[" << i << "] ";
	data_[i].print(out); 
      }
  }


private:

  //collection of samples
  std::vector<S> data_;
  
  // det id for this data frame
  D id_;

  //number of samples and maximum available
  size_t maxSampleSize_;

};

#endif
