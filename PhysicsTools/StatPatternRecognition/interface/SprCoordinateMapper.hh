// File and Version Information:
//      $Id: SprCoordinateMapper.hh,v 1.2 2007/10/30 18:56:12 narsky Exp $
//
// Description:
//      Class SprCoordinateMapper
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005,2007         California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprCoordinateMapper_HH
#define _SprCoordinateMapper_HH

#include <vector>
#include <string>

class SprPoint;


class SprCoordinateMapper
{
public:
  static SprCoordinateMapper* createMapper(const std::vector<unsigned>& 
					   mapper) {
    if( mapper.empty() ) return 0;
    return new SprCoordinateMapper(mapper);
  }

  static SprCoordinateMapper* 
  createMapper(const std::vector<std::string>& from,
	       const std::vector<std::string>& to);

  virtual ~SprCoordinateMapper() { this->clear(); }

  SprCoordinateMapper* clone() const {
    return new SprCoordinateMapper(*this);
  }

  /*
    output() method requires clear() to be called before deletion of
    the mapper. map() method does not require the cleaner.
  */
  const SprPoint* output(const SprPoint* input);
  void map(const std::vector<double>& in, std::vector<double>& out) const;

  // clean up
  void clear();

  // return mapper
  void mapper(std::vector<unsigned>& mapper) const {
    mapper = mapper_;
  }

  // return mapped index
  int mappedIndex(int d) const {
    if( d<0 || d>=static_cast<int>(mapper_.size()) ) return -1;
    return mapper_[d];
  }

private:
  SprCoordinateMapper(const std::vector<unsigned>& mapper)
    : mapper_(mapper), toDelete_() {}

  SprCoordinateMapper(const SprCoordinateMapper& other)
    : mapper_(other.mapper_), toDelete_() {}

  std::vector<unsigned> mapper_;
  std::vector<const SprPoint*> toDelete_;
};





#endif
