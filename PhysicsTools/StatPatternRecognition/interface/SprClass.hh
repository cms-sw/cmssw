//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprClass.hh,v 1.3 2007/10/30 18:56:12 narsky Exp $
//
// Description:
//      Class SprClass :
//          Keeps info about point's class.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2006              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprClass_HH
#define _SprClass_HH

#include <iostream>
#include <vector>
#include <string>
#include <cassert>


class SprClass
{
public:
  ~SprClass() {}

  SprClass()
    : classes_(1,0), negate_(false) {}

  SprClass(int cls, bool negate=false)
    : classes_(1,cls), negate_(negate) {}

  SprClass(const std::vector<int>& classes, bool negate=false)
    : classes_(classes), negate_(negate) 
  {
    bool status = this->checkClasses();
    assert( status );
  }

  SprClass(const SprClass& other)
    :
    classes_(other.classes_),
    negate_(other.negate_)
  {}

  // Check for overlap.
  // Return 0 if no overlap, 1 if overlap , and -1 if unknown (negation=true).
  int overlap(const SprClass& other) const;

  // operators
  bool operator==(int cls) const;
  bool operator==(const SprClass& other) const;

  bool operator!=(int cls) const { return !this->operator==(cls); }
  bool operator!=(const SprClass& other) const {
    return !this->operator==(other);
  }

  inline SprClass& operator=(int cls);
  inline SprClass& operator=(const SprClass& other);

  // equality to integer
  bool int_eq(int cls) const {
    return this->operator==(cls);
  }

  // accessor
  bool value(std::vector<int>& classes) const { 
    classes = classes_;
    return negate_;
  }

  // express class value as a string
  std::string toString() const;

private:
  bool checkClasses() const;// checks classes for absence of repetitions

  std::vector<int> classes_;
  bool negate_;
};


inline SprClass& SprClass::operator=(int cls) 
{
  classes_.clear();
  classes_.resize(1,cls);
  negate_ = false;
  return *this;
}


inline SprClass& SprClass::operator=(const SprClass& other) 
{
  classes_ = other.classes_;
  negate_ = other.negate_;
  return *this;
}


inline bool operator==(int cls1, const SprClass& cls2) {
  return (cls2==cls1);
}


inline bool operator!=(int cls1, const SprClass& cls2) {
  return !(cls2==cls1);
}


inline std::ostream& operator<<(std::ostream& os, const SprClass& cls) {
  std::vector<int> classes;
  bool negate = cls.value(classes);
  if( !classes.empty() ) {
    for( unsigned int i=0;i<classes.size()-1;i++ )
      os << classes[i] << ",";
    os << classes[classes.size()-1];
  }
  os << "(" << ( negate ? -1 : 1 ) << ")";
  return os;
}

#endif
