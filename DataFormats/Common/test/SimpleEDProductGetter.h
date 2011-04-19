#ifndef DataFormats_Common_SimpleEDProductGetter_h
#define DataFormats_Common_SimpleEDProductGetter_h

#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/WrapperInterfaceBase.h"

#include <map>
#include <memory>

class SimpleEDProductGetter : public edm::EDProductGetter {
public:

  typedef std::map<edm::ProductID, edm::WrapperHolder> map_t;

  template<typename T>
  void
  addProduct(edm::ProductID const& id, std::auto_ptr<T> p) {
    typedef edm::Wrapper<T> wrapper_t;
    edm::WrapperInterfaceBase const* interface = wrapper_t::getInterface();
    boost::shared_ptr<void const> wrapper(new wrapper_t(p), edm::WrapperHolder::EDProductDeleter(interface));
    database[id] = edm::WrapperHolder(wrapper, interface);
  }

  size_t size() const {
    return database.size();
  }

  virtual edm::WrapperHolder getIt(edm::ProductID const& id) const {
    map_t::const_iterator i = database.find(id);
    if (i == database.end()) {
      edm::Exception e(edm::errors::ProductNotFound, "InvalidID");
      e << "No product with ProductID "
        << id
        << " is available from this EDProductGetter\n";
      e.raise();
    }
    return i->second;
  }

private:
  map_t database;
};
#endif
