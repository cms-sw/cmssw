#ifndef FWCore_Framework_DelayedReader_h
#define FWCore_Framework_DelayedReader_h

/*----------------------------------------------------------------------
  
DelayedReader: The abstract interface through which the EventPrincipal
uses input sources to retrieve EDProducts from external storage.

$Id: DelayedReader.h,v 1.6 2007/05/29 19:28:15 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include "DataFormats/Provenance/interface/ProvenanceDelayedReader.h"
#include "boost/shared_ptr.hpp"

class TFile;

namespace edm {
  class BranchKey;
  class EDProduct;
  class EDProductGetter;
  class DelayedReader : public ProvenanceDelayedReader {
  public:
    virtual ~DelayedReader();
    boost::shared_ptr<TFile const> filePtr() const {return filePtrImpl();}
    virtual std::auto_ptr<EDProduct> getProduct(BranchKey const& k, EDProductGetter const* ep) const = 0;

  private:
    virtual boost::shared_ptr<TFile const> filePtrImpl() const {return boost::shared_ptr<TFile const>();}
  };
}

#endif
