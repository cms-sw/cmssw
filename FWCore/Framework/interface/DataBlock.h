#ifndef Framework_DataBlock_h
#define Framework_DataBlock_h

/*----------------------------------------------------------------------
  
DataBlock: These are the classes responsible for management of
EDProducts. It is not seen by reconstruction code;

The template parameter, AUX, contains the information for the Event,
Luminosity Block, or Run, that is not in an EDProduct.

$Id: DataBlock.h,v 1.2 2006/10/31 23:54:01 wmtan Exp $

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Framework/interface/DataBlockImpl.h"
#include "FWCore/Framework/interface/EPEventProvenanceFiller.h"
#include "FWCore/Framework/interface/UnscheduledHandler.h"

namespace edm {
    
  class DelayedReader;
  class NoDelayedReader;

  template <typename AUX>
  class DataBlock : private DataBlockImpl {
    typedef typename AUX::IDValue IDValue;
    typedef typename AUX::TimeValue TimeValue;
    typedef typename AUX::LumiValue LumiValue;
  public:
    typedef DataBlockImpl::const_iterator const_iterator;
    typedef DataBlockImpl::SharedConstGroupPtr SharedConstGroupPtr;

    DataBlock(IDValue const& id,
	TimeValue const& time,
	ProductRegistry const& reg,
	ProcessHistoryID const& hist = ProcessHistoryID(),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader)) :
      DataBlockImpl(reg, hist, rtrv),
      aux_(id, time),
      unscheduledHandler_(),
      provenanceFiller_() {}
    DataBlock(IDValue const& id,
	TimeValue const& time,
	ProductRegistry const& reg,
	LumiValue const& lumi,
	ProcessHistoryID const& hist = ProcessHistoryID(),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader)) :
      DataBlockImpl(reg, hist, rtrv),
      aux_(id, time, lumi),
      unscheduledHandler_(),
      provenanceFiller_() {}
    ~DataBlock() {}

    IDValue id() const {return aux_.id();}
    TimeValue time() const {return aux_.time();}
    AUX const& aux() const {return aux_;}
    DataBlockImpl & impl() {return *this;}

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler);

    using DataBlockImpl::addGroup;
    using DataBlockImpl::addToProcessHistory;
    using DataBlockImpl::begin;
    using DataBlockImpl::beginProcess;
    using DataBlockImpl::end;
    using DataBlockImpl::endProcess;
    using DataBlockImpl::getAllProvenance;
    using DataBlockImpl::getByLabel;
    using DataBlockImpl::get;
    using DataBlockImpl::getBySelector;
    using DataBlockImpl::getByType;
    using DataBlockImpl::getGroup;
    using DataBlockImpl::getIt;
    using DataBlockImpl::getMany;
    using DataBlockImpl::getManyByType;
    using DataBlockImpl::getProvenance;
    using DataBlockImpl::groupGetter;
    using DataBlockImpl::numEDProducts;
    using DataBlockImpl::processHistory;
    using DataBlockImpl::processHistoryID;
    using DataBlockImpl::prodGetter;
    using DataBlockImpl::productRegistry;
    using DataBlockImpl::put;
    using DataBlockImpl::store;
  private:
    virtual bool unscheduledFill(Group const& group) const;
    virtual bool fillAndMatchSelector(Provenance& prov, SelectorBase const& selector) const;

    AUX aux_;
    // Handler for unscheduled modules
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler_;
    // Provenance filler for unscheduled modules
    boost::shared_ptr<EPEventProvenanceFiller> provenanceFiller_;
  };
  
  template <typename AUX>
  void
  DataBlock<AUX>::setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler) {
    unscheduledHandler_ = iHandler;
    provenanceFiller_ = boost::shared_ptr<EPEventProvenanceFiller>(
	new EPEventProvenanceFiller(unscheduledHandler_, const_cast<DataBlock<AUX>*>(this)));
  }

  template <typename AUX>
  bool
  DataBlock<AUX>::fillAndMatchSelector(Provenance& prov, SelectorBase const& selector) const {
    ProvenanceAccess provAccess(&prov, provenanceFiller_.get());
    return (selector.match(provAccess));
  }

  template <typename AUX>
  bool
  DataBlock<AUX>::unscheduledFill(Group const& group) const {
    if (unscheduledHandler_ &&
	unscheduledHandler_->tryToFill(group.provenance(), *const_cast<DataBlock<AUX>*>(this))) {
      //see if product actually retrieved.
      if(!group.product()) {
        throw edm::Exception(errors::ProductNotFound, "InaccessibleProduct")
          <<"product not accessible\n" << group.provenance();
      }
      return true;
    }
    return false;
  }
}
#endif

