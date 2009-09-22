#include "CommonTools/Utils/interface/strbitset.h"



namespace std {
strbitset operator&(const strbitset& l, const strbitset& r)
{
  strbitset ret; ret.set(false);
  strbitset ret1 ( r );
  ret1.set(false);
  if ( r.map_.size() != l.map_.size() ) {
    std::cout << "strbitset operator& : rhs and lhs do not have same size" << std::endl;
    return ret;
  }
  for ( strbitset::str_index_map::const_iterator irbegin = r.map_.begin(),
	  irend = r.map_.end(), ir = irbegin;
	ir != irend; ++ir ) {
    std::string key = ir->first;
    strbitset::str_index_map::const_iterator il = l.map_.find( key );
    if ( il == l.map_.end() ) {
      std::cout << "strbitset operator& : rhs and lhs do not share same elements" << std::endl;
      return ret;
    }
    ret1[key] = l[key] && r[key];
  }
  return ret1;
}


strbitset operator|(const strbitset& l, const strbitset& r)
{
  strbitset ret; ret.set(false);
  strbitset ret1 ( r );
  ret1.set(false);
  if ( r.map_.size() != l.map_.size() ) {
    std::cout << "strbitset operator& : rhs and lhs do not have same size" << std::endl;
    return ret;
  }
  for ( strbitset::str_index_map::const_iterator irbegin = r.map_.begin(),
	  irend = r.map_.end(), ir = irbegin;
	ir != irend; ++ir ) {
    std::string key = ir->first;
    strbitset::str_index_map::const_iterator il = l.map_.find( key );
    if ( il == l.map_.end() ) {
      std::cout << "strbitset operator& : rhs and lhs do not share same elements" << std::endl;
      return ret;
    }
    ret1[key] = l[key] || r[key];
  }
  return ret1;
}


strbitset operator^(const strbitset& l, const strbitset& r)
{
  strbitset ret; ret.set(false);
  strbitset ret1 ( r );
  ret1.set(false);
  if ( r.map_.size() != l.map_.size() ) {
    std::cout << "strbitset operator& : rhs and lhs do not have same size" << std::endl;
    return ret;
  }
  for ( strbitset::str_index_map::const_iterator irbegin = r.map_.begin(),
	  irend = r.map_.end(), ir = irbegin;
	ir != irend; ++ir ) {
    std::string key = ir->first;
    strbitset::str_index_map::const_iterator il = l.map_.find( key );
    if ( il == l.map_.end() ) {
      std::cout << "strbitset operator& : rhs and lhs do not share same elements" << std::endl;
      return ret;
    }
    ret1[key] = (l[key] || r[key]) && !(l[key] && r[key]);
  }
  return ret1;
}

}
