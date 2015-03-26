#ifndef DataFormats_Common_SimpleEDProductGetter_h
#define DataFormats_Common_SimpleEDProductGetter_h

#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <map>
#include <memory>

class SimpleEDProductGetter : public edm::EDProductGetter {
public:

  typedef std::map<edm::ProductID, std::shared_ptr<edm::WrapperBase> > map_t;

  template<typename T>
  void
  addProduct(edm::ProductID const& id, std::unique_ptr<T> p) {
    typedef edm::Wrapper<T> wrapper_t;
    std::shared_ptr<wrapper_t> product = std::make_shared<wrapper_t>(std::move(p));
    database[id] = product;
  }

  size_t size() const {
    return database.size();
  }

  virtual edm::WrapperBase const* getIt(edm::ProductID const& id) const override {
    map_t::const_iterator i = database.find(id);
    if (i == database.end()) {
      edm::Exception e(edm::errors::ProductNotFound, "InvalidID");
      e << "No product with ProductID "
        << id
        << " is available from this EDProductGetter\n";
      throw e;
    }
    return i->second.get();
  }

  virtual edm::WrapperBase const*
  getThinnedProduct(edm::ProductID const&, unsigned int&) const override {return nullptr;}

  virtual void
  getThinnedProducts(edm::ProductID const& pid,
                     std::vector<edm::WrapperBase const*>& wrappers,
                     std::vector<unsigned int>& keys) const override { }


private:
  virtual unsigned int transitionIndex_() const override {
    return 0U;
  }

  map_t database;
};
#endif
