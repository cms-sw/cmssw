#ifndef INDEXED_H
#define INDEXED_H

template<class T> 
class Indexed : public T {
 private: 
  unsigned index_;
 public: 
  Indexed() : T(), index_(0) {}
  Indexed(T t, unsigned i) : T(t), index_(i) {}
  unsigned index() const {return index_;}
  Indexed operator+(const Indexed& rhs) const {return Indexed(this->T::operator+(rhs),0);}
};
  
#endif
