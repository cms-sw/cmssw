/*
 * $Id: Majority.h,v 1.1 2009/02/25 17:34:05 pgras Exp $
 */

#ifndef MAJORITY_H
#define MAJORITY_H

#include <map>
  
/**
 */
template<class T>
class Majority {
  //constructor(s) and destructor(s)
public:
  /** Constructs a Majority
   */
  Majority(): n_(0){}

  /**Destructor
   */
  virtual ~Majority(){}
  
  void
  add(const T& value){
    votes_[value] += 1.;
    n_ += 1.;
  }
  
  T result(double* proba) const{
    std::pair<T, double> m(T(), -1.);
    for(typename std::map<T, double>::const_iterator it = votes_.begin();
	it != votes_.end();
	++it){
      if(it->second > m.second){
	m = *it;
      }
    }
    if(proba) *proba = n_>0?m.second/n_:-1;
    return m.first;
  }

  //method(s)
public:
private:

  //attribute(s)
protected:
private:
  std::map<T, double> votes_;
  double n_;
};

#endif //MAJORITY_H not defined
