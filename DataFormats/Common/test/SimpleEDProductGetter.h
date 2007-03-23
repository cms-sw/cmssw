#ifndef DATAFORMAT_COMMON_TEST_SIMPLEEDPRODUCTGETTER_H
#define DATAFORMAT_COMMON_TEST_SIMPLEEDPRODUCTGETTER_H

#include "DataFormats/Common/interface/EDProductGetter.h"

#include <map>
#include "boost/shared_ptr.hpp"
#include "DataFormats/Common/interface/EDProduct.h"

class SimpleEDProductGetter : public edm::EDProductGetter
{
 public:
  virtual edm::EDProduct const* getIt(edm::ProductID const& id) const { return 0; }

 private:

  std::map<edm::ProductID, boost::shared_ptr<edm::EDProduct> > database;
};

#endif
