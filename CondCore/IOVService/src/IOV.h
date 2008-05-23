#ifndef IOVService_IOV_h
#define IOVService_IOV_h
#include "CondCore/DBCommon/interface/Time.h"
#include <vector>
#include <string>
#include <algorithm>
#include <boost/bind.hpp>

namespace cond {

  class IOV {
  public:
    typedef std::pair<cond::Time_t, std::string> Item;
    typedef std::vector<Item> Container;
    typedef Container::iterator iterator;
    typedef Container::const_iterator const_iterator;

    IOV(){}
    IOV(int type, cond::Time_t since) :
      timetype(type), firstsince(since){}
    
    virtual ~IOV(){}


    size_t add(cond::Time_t time, std::string const & token) {
      iov.push_back(Item(time, token));
      return iov.size()-1;
    }

    iterator find(cond::Time_t time) {
      return std::lower_bound(iov.begin(),iov.end(),Item(time,""),
			      boost::bind(std::less<cond::Time_t>(),
					  boost::bind(&Item::first,_1),
					  boost::bind(&Item::first,_2)
					  )
			      );
    }

    const_iterator find(cond::Time_t time) const {
      return std::lower_bound(iov.begin(),iov.end(),Item(time,""),
			      boost::bind(std::less<cond::Time_t>(),
					  boost::bind(&Item::first,_1),
					  boost::bind(&Item::first,_2)
					  )
			      );
    }


    cond::TimeType timeType() const { return cond::timeTypeSpecs[timetype].type;}

    //std::map<unsigned long long,std::string> iov;
    Container iov;
    int timetype;
    cond::Time_t firstsince;
  };

}//ns cond
#endif
