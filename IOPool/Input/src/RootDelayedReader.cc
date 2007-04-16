/*----------------------------------------------------------------------
$Id: RootDelayedReader.cc,v 1.7 2007/04/08 23:13:48 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/Input/interface/RootDelayedReader.h"
#include "IOPool/Common/interface/RefStreamer.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "Reflex/Type.h"
#include "Reflex/Object.h"

namespace edm {

  RootDelayedReader::RootDelayedReader(EntryNumber const& entry,
 boost::shared_ptr<BranchMap const> bMap,
 boost::shared_ptr<TFile const> filePtr)
 : entryNumber_(entry), branches_(bMap), filePtr_(filePtr) {}

  RootDelayedReader::~RootDelayedReader() {}

  std::auto_ptr<EDProduct>
  RootDelayedReader::get(BranchKey const& k, EDProductGetter const* ep) const {
    SetRefStreamer(ep);
    ROOT::Reflex::Object object = branches().find(k)->second.first.Construct();
    std::auto_ptr<EDProduct> p(static_cast<EDProduct *>(object.Address()));
    TBranch *br = branches().find(k)->second.second;
    EDProduct *pp = p.get();
    br->SetAddress(&pp);
    br->GetEntry(entryNumber_);
    return p;
  }
}
