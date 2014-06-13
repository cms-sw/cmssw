#ifndef DataFormats_Common_SimpleEDProductGetter_h
#define DataFormats_Common_SimpleEDProductGetter_h

#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/WrapperInterfaceBase.h"

#include <map>
#include <memory>

class SimpleEDProductGetter : public edm::EDProductGetter {
public:

  typedef std::map<edm::ProductID, edm::WrapperOwningHolder> map_t;

  template<typename T>
  void
  addProduct(edm::ProductID const& id, std::auto_ptr<T> p) {
    database[id] = edm::WrapperOwningHolder(new edm::Wrapper<T>(p), edm::Wrapper<T>::getInterface());
  }

  size_t size() const {
    return database.size();
  }

  virtual edm::WrapperHolder getIt(edm::ProductID const& id) const override {
    map_t::const_iterator i = database.find(id);
    if (i == database.end()) {
      edm::Exception e(edm::errors::ProductNotFound, "InvalidID");
      e << "No product with ProductID "
        << id
        << " is available from this EDProductGetter\n";
      throw e;
    }
    return edm::WrapperHolder(i->second.wrapper(), i->second.interface());
  }

private:
  virtual unsigned int transitionIndex_() const override {
    return 0U;
  }

  map_t database;
};
#endif
