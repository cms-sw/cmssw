#ifndef IOVService_IOV_h
#define IOVService_IOV_h
#include "CondCore/DBCommon/interface/Time.h"
#include <vector>
#include <string>
#include <algoritm>
#include <boost/bind.hpp>

namespace cond{
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

    iterator find(cond::Time_t time) {
      return std::lower_bound(iov.begin(),iov.end(),
			      boost::bind(std::less<float>(),
					  boost::bind(&Item::first,_1),
					  boost::bind(&Item::first,_2)
					  )
			      );
    }

    const_iterator find(cond::Time_t time) const {
      return std::lower_bound(iov.begin(),iov.end(),
			      boost::bind(std::less<float>(),
					  boost::bind(&Item::first,_1),
					  boost::bind(&Item::first,_2)
					  )
			      );
    }

    //std::map<unsigned long long,std::string> iov;
    Container iov;
    int timetype;
    cond::Time_t firstsince;
  };
}//ns cond
#endif
