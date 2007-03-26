#ifndef DATAFORMAT_COMMON_TEST_SIMPLEEDPRODUCTGETTER_H
#define DATAFORMAT_COMMON_TEST_SIMPLEEDPRODUCTGETTER_H

#include <map>
#include <memory>

#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/Wrapper.h"


class SimpleEDProductGetter : public edm::EDProductGetter
{
 public:
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
    boost::shared_ptr<edm::EDProduct> pprod = database[id];
    return &*pprod;
  }

 private:

  mutable std::map<edm::ProductID, boost::shared_ptr<edm::EDProduct> > database;
};

#endif
