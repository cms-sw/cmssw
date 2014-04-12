#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

//#define LOCAL_DEBUG

void MuonBaseNumber::addBase(LevelBaseNumber num){
  basenumber_type::iterator cur=sortedBaseNumber.begin();
  basenumber_type::iterator end=sortedBaseNumber.end();

  // do a small check if level is already occupied

  while (cur!=end) {
    if (num.level()==(*cur).level()) {
#ifdef LOCAL_DEBUG
      std::cout << "MuonBaseNumber::addBase was asked to add "
		<<num.level()<<" "
		<<num.super()<<" "
		<<num.base()
		<<" to existing level "
		<<(*cur).level()<<" "
		<<(*cur).super()<<" "
		<<(*cur).base() << " but refused.";
#endif
      return; // don't overwrite current volume stored
    }
    cur++;
  }

  cur=sortedBaseNumber.begin();
  while (cur!=end) {
    if (num.level()<(*cur).level()) break;
    cur++;
  }
  sortedBaseNumber.insert(cur,num);

#ifdef LOCAL_DEBUG
  cur=sortedBaseNumber.begin();
  end=sortedBaseNumber.end();
  std::cout << "MuonBaseNumber::AddBase ";
  while (cur!=end) {
    std::cout<<(*cur).level()<<" ";
    std::cout<<(*cur).super()<<" ";
    std::cout<<(*cur).base();
    std::cout<<",";
    cur++;
  }
  std::cout <<std::endl;
#endif

}

void MuonBaseNumber::addBase(const int level,const int super,const int base){
  LevelBaseNumber num(level,super,base);
  addBase(num);
}

int MuonBaseNumber::getLevels() const {
  return sortedBaseNumber.size();
}

int MuonBaseNumber::getSuperNo(int level) const {
  basenumber_type::const_iterator cur=sortedBaseNumber.begin();
  basenumber_type::const_iterator end=sortedBaseNumber.end();
  while (cur!=end) {
    if ((*cur).level()==level) {
      return (*cur).super();
    }
    cur++;
  }
  return 0;
}

int MuonBaseNumber::getBaseNo(int level) const {
  basenumber_type::const_iterator cur=sortedBaseNumber.begin();
  basenumber_type::const_iterator end=sortedBaseNumber.end();
  while (cur!=end) {
    if ((*cur).level()==level) {
      return (*cur).base();
    }
    cur++;
  }
  return 0;
}





