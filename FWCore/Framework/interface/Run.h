#ifndef Framework_Run_h
#define Framework_Run_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     Run
// 
/**\class Run Run.h FWCore/Framework/interface/Run.h

Description: This is the primary interface for accessing per run EDProducts and inserting new derived products.

For its usage, see "FWCore/Framework/interface/DataViewImpl.h"

*/
/*----------------------------------------------------------------------

$Id: Run.h,v 1.13 2008/05/12 18:14:07 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class Run : private DataViewImpl<RunLumiEntryInfo>
  {
  public:
    Run(RunPrincipal& rp, const ModuleDescription& md);
    ~Run(){}

    typedef DataViewImpl<RunLumiEntryInfo> Base;
    // AUX functions.
    RunID const& id() const {return aux_.id();}
    RunNumber_t run() const {return aux_.run();}
    Timestamp const& beginTime() const {return aux_.beginTime();}
    Timestamp const& endTime() const {return aux_.endTime();}

    using Base::get;
    using Base::getByLabel;
    using Base::getByType;
    using Base::getMany;
    using Base::getManyByType;
    using Base::me;
    using Base::processHistory;

    ///Put a new product.
    template <typename PROD>
    void
    put(std::auto_ptr<PROD> product) {put<PROD>(product, std::string());}

    ///Put a new product with a 'product instance name'
    template <typename PROD>
    void
    put(std::auto_ptr<PROD> product, std::string const& productInstanceName);

    Provenance
    getProvenance(BranchID const& theID) const;

    void
    getAllProvenance(std::vector<Provenance const*> &provenances) const;

    // Return true if this Run has been subjected to a process with
    // the given processName, and false otherwise.
    // If true is returned, then ps is filled with the ParameterSets
    // (possibly more than one) used to configure the identified
    // process(es). Equivalent ParameterSets are compressed out of the
    // result.
    bool
    getProcessParameterSet(std::string const& processName,
			   std::vector<ParameterSet>& ps) const;

  private:
    RunPrincipal const&
    runPrincipal() const;

    RunPrincipal &
    runPrincipal();

    // commit_() is called to complete the transaction represented by
    // this DataViewImpl. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class DaqSource;
    friend class InputSource;
    friend class RawInputSource;
    friend class EDFilter;
    friend class EDProducer;

    void commit_();

    RunAuxiliary const& aux_;
  };

  template <typename PROD>
  void
  Run::put(std::auto_ptr<PROD> product, std::string const& productInstanceName)
  {
    if (product.get() == 0) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      throw edm::Exception(edm::errors::NullPointerError)
        << "Run::put: A null auto_ptr was passed to 'put'.\n"
	<< "The pointer is of type " << typeID << ".\n"
	<< "The specified productInstanceName was '" << productInstanceName << "'.\n";
    }

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    typename boost::mpl::if_c<detail::has_postinsert<PROD>::value, 
      DoPostInsert<PROD>, 
      DoNotPostInsert<PROD> >::type maybe_inserter;
    maybe_inserter(product.get());

    ConstBranchDescription const& desc =
      getBranchDescription(TypeID(*product), productInstanceName);

    Wrapper<PROD> *wp(new Wrapper<PROD>(product));

    putProducts().push_back(std::make_pair(wp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.
  }

}
#endif
