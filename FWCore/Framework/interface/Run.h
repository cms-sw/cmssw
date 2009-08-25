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

$Id: Run.h,v 1.15 2008/08/22 01:44:37 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class Run
  {
  public:
    Run(RunPrincipal& rp, const ModuleDescription& md);
    ~Run(){}

    typedef DataViewImpl Base;
    // AUX functions.
    RunID const& id() const {return aux_.id();}
    RunNumber_t run() const {return aux_.run();}
    Timestamp const& beginTime() const {return aux_.beginTime();}
    Timestamp const& endTime() const {return aux_.endTime();}

    template <typename PROD>
    bool 
    get(SelectorBase const&, Handle<PROD>& result) const;
    
    template <typename PROD>
    bool 
    getByLabel(std::string const& label, Handle<PROD>& result) const;
    
    template <typename PROD>
    bool 
    getByLabel(std::string const& label,
	       std::string const& productInstanceName, 
	       Handle<PROD>& result) const;
    
    /// same as above, but using the InputTag class 	 
    template <typename PROD> 	 
    bool 	 
    getByLabel(InputTag const& tag, Handle<PROD>& result) const; 	 
    
    template <typename PROD>
    void 
    getMany(SelectorBase const&, std::vector<Handle<PROD> >& results) const;
    
    template <typename PROD>
    bool
    getByType(Handle<PROD>& result) const;
    
    template <typename PROD>
    void 
    getManyByType(std::vector<Handle<PROD> >& results) const;
    
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

    ProcessHistory const&
    processHistory() const;

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

    DataViewImpl provRecorder_;
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
      provRecorder_.getBranchDescription(TypeID(*product), productInstanceName);

    Wrapper<PROD> *wp(new Wrapper<PROD>(product));

    provRecorder_.putProducts().push_back(std::make_pair(wp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.
  }

  template <typename PROD>
  bool 
  Run::get(SelectorBase const& sel, Handle<PROD>& result) const {
    return provRecorder_.get(sel,result);
  }
  
  template <typename PROD>
  bool 
  Run::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return provRecorder_.getByLabel(label,result);
  }
  
  template <typename PROD>
  bool 
  Run::getByLabel(std::string const& label,
                  std::string const& productInstanceName, 
                  Handle<PROD>& result) const {
    return provRecorder_.getByLabel(label,productInstanceName,result);
  }
  
  /// same as above, but using the InputTag class 	 
  template <typename PROD> 	 
  bool 	 
  Run::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    return provRecorder_.getByLabel(tag,result);
  }
  
  template <typename PROD>
  void 
  Run::getMany(SelectorBase const& sel, std::vector<Handle<PROD> >& results) const {
    return provRecorder_.getMany(sel,results);
  }
  
  template <typename PROD>
  bool
  Run::getByType(Handle<PROD>& result) const {
    return provRecorder_.getByType(result);
  }
  
  template <typename PROD>
  void 
  Run::getManyByType(std::vector<Handle<PROD> >& results) const {
    return provRecorder_.getManyByType(results);
  }
  
}
#endif
