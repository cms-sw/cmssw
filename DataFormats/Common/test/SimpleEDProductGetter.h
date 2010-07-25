#ifndef DATAFORMAT_COMMON_TEST_SIMPLEEDPRODUCTGETTER_H
#define DATAFORMAT_COMMON_TEST_SIMPLEEDPRODUCTGETTER_H

#include <map>
#include <memory>

#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/Wrapper.h"


class SimpleEDProductGetter : public edm::EDProductGetter
{
 public:

  typedef std::map<edm::ProductID, boost::shared_ptr<edm::EDProduct> > map_t;
  template <class T>
  void 
  addProduct(edm::ProductID const& id, std::auto_ptr<T> p)
  {
    typedef edm::Wrapper<T> wrapper_t;

    boost::shared_ptr<wrapper_t> product(new wrapper_t(p));
    database[id] = product;    
  }

  size_t size() const 
  { return database.size(); }

  virtual edm::EDProduct const* getIt(edm::ProductID const& id) const
  { 
    map_t::const_iterator i = database.find(id);
    if (i == database.end()) {
      edm::Exception e(edm::errors::ProductNotFound, "InvalidID");
      e << "No product with ProductID " 
        << id 
        << " is available from this EDProductGetter\n";
      e.raise();
    }
    return i->second.get();
  }

 private:
  map_t database;
};

#endif
