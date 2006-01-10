
#include "IOPool/Streamer/interface/BufferArea.h"
#include "IOPool/Streamer/interface/HLTInfo.h"

namespace 
{
  // 250KB/fragment, allocate room for 20 fragments
  const int fragment_count = 20;
  const int frag_size = sizeof(stor::FragEntry);
  const int ptr_size = sizeof(void*);
}

namespace stor 
{

  HLTInfo::HLTInfo():
    cmd_q_(edm::getEventBuffer(ptr_size,50)),
    evtbuf_q_(edm::getEventBuffer(ptr_size,100)),
    frag_q_(edm::getEventBuffer(frag_size,200))
  {
  }

  HLTInfo::HLTInfo(const edm::ParameterSet& ps):
    cmd_q_(edm::getEventBuffer(ptr_size,50)),
    evtbuf_q_(edm::getEventBuffer(ptr_size,100)),
    frag_q_(edm::getEventBuffer(frag_size,200))
  {
  }

  HLTInfo::HLTInfo(const edm::ProductRegistry& pr):
    prods_(pr),
    cmd_q_(edm::getEventBuffer(ptr_size,50)),
    evtbuf_q_(edm::getEventBuffer(ptr_size,100)),
    frag_q_(edm::getEventBuffer(frag_size,200))
  {
  }

  HLTInfo::~HLTInfo()
  {
  }

  HLTInfo::HLTInfo(const HLTInfo&) { }
  const HLTInfo& HLTInfo::operator=(const HLTInfo&) { return *this; }

  void HLTInfo::mergeRegistry(edm::ProductRegistry& pr)
  {
    typedef edm::ProductRegistry::ProductList ProdList; 
    ProdList plist(prods_.productList());
    ProdList::iterator pi(plist.begin()),pe(plist.end());
    
    for(;pi!=pe;++pi)
      {
	pr.copyProduct(pi->second);
      }
  }

  boost::mutex HLTInfo::lock_;
}


