#ifndef x_TagName_h
#define x_TagName_h

#include <utility>
#include <string>
#include <iostream>
#include <map>

using std::pair;
using std::string;
using std::cout;
using std::endl;
using std::map;
using std::pair;

class TagName 
{
public:
  
  TagName() {}
  
  explicit TagName(const string & name) 
    : name_(regName(name)), id_(count()) 
    { };
  
  const std::string & str() const { return name_->first; }
  
  std::string operator()() const { return name_->first; }
  
  bool sameName(const TagName & tn) const {
    return (name_ == tn.name_);
  }
  
  bool operator<(const TagName & n) const {
    return (id_ < n.id_);
  }
    
private:
  typedef std::map<std::string,unsigned int> Registry;
  typedef unsigned int count_type;
  
  Registry::iterator name_;
  count_type id_; // identification for equality checks
  
  static Registry::iterator regName(const string & s) {
    static Registry reg;
    Registry::size_type sz = reg.size();
    Registry::value_type val(s, sz);
    /*
    pair<Registry::iterator, bool> insert = reg.insert(val);
    if (!insert.second) {  
      sz = insert.first->second;
    }  
    return insert.first;
    */
    return reg.insert(val).first;
  }
  
  static count_type count()  {
    static count_type i=0;
    ++i;
    return i;
   } 
};

#endif
