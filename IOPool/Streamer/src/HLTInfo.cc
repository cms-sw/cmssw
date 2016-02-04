#include "IOPool/Streamer/interface/HLTInfo.h"

#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "IOPool/Streamer/interface/BufferArea.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

namespace {
  // 250KB/fragment, allocate room for 20 fragments
  int const fragment_count = 20;
  int const frag_size = sizeof(stor::FragEntry);
  int const ptr_size = sizeof(void*);
}

namespace stor {

  HLTInfo::HLTInfo():
    cmd_q_(edm::getEventBuffer(ptr_size, 50)),
    evtbuf_q_(edm::getEventBuffer(ptr_size, 100)),
    frag_q_(edm::getEventBuffer(frag_size, 200)) {
  }

  HLTInfo::HLTInfo(edm::ParameterSet const&):
    cmd_q_(edm::getEventBuffer(ptr_size, 50)),
    evtbuf_q_(edm::getEventBuffer(ptr_size, 100)),
    frag_q_(edm::getEventBuffer(frag_size, 200)) {
  }

  HLTInfo::HLTInfo(edm::ProductRegistry const& pr):
    prods_(pr),
    cmd_q_(edm::getEventBuffer(ptr_size, 50)),
    evtbuf_q_(edm::getEventBuffer(ptr_size, 100)),
    frag_q_(edm::getEventBuffer(frag_size, 200)) {
     prods_.setFrozen();
  }

  HLTInfo::~HLTInfo() {
  }

  HLTInfo::HLTInfo(HLTInfo const&) { }
  HLTInfo const& HLTInfo::operator=(HLTInfo const&) { return *this; }

  void HLTInfo::mergeRegistry(edm::ProductRegistry& pr) {
    typedef edm::ProductRegistry::ProductList ProdList;
    ProdList plist(prods_.productList());
    ProdList::iterator pi(plist.begin()), pe(plist.end());

    for(; pi != pe; ++pi) {
      pr.copyProduct(pi->second);
    }
  }

  void HLTInfo::declareStreamers(edm::ProductRegistry const& reg) {
    typedef edm::ProductRegistry::ProductList ProdList;
    ProdList plist(reg.productList());
    ProdList::const_iterator pi(plist.begin()), pe(plist.end());

    for(; pi != pe; ++pi) {
      //pi->second.init();
      std::string real_name = edm::wrappedClassName(pi->second.className());
      //FDEBUG(6) << "declare: " << real_name << std::endl;
      edm::loadCap(real_name);
    }
  }

  void HLTInfo::buildClassCache(edm::ProductRegistry const& reg) {
    typedef edm::ProductRegistry::ProductList ProdList;
    ProdList plist(reg.productList());
    ProdList::const_iterator pi(plist.begin()), pe(plist.end());

    for(; pi != pe; ++pi) {
      //pi->second.init();
      std::string real_name = edm::wrappedClassName(pi->second.className());
      //FDEBUG(6) << "BuildReadData: " << real_name << std::endl;
      edm::doBuildRealData(real_name);
    }
  }

  boost::mutex HLTInfo::lock_;
}

