#ifndef DataFormats_PatCandidates_TriggerPath_h
#define DataFormats_PatCandidates_TriggerPath_h

// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerPath
//
//
/*
  \class    pat::TriggerPath TriggerPath.h "DataFormats/PatCandidates/interface/TriggerPath.h"
  \brief    Analysis-level HLTrigger path class

   TriggerPath implements a container for trigger paths' information within the 'pat' namespace.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerPath

  \author   Volker Adler
*/

#include <string>
#include <vector>
#include <type_traits>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"

namespace pat {

  /// Pair to store decision and name of L1 seeds
  typedef std::pair<bool, std::string> L1Seed;
  /// Collection of L1Seed
  typedef std::vector<L1Seed> L1SeedCollection;

  class TriggerPath {
    /// Data Members

    /// Path name
    std::string name_;
    /// Path index in trigger table
    unsigned index_;
    /// Pre-scale
    double prescale_;
    /// Was path run?
    bool run_;
    /// Did path succeed?
    bool accept_;
    /// Was path in error?
    bool error_;
    /// List of all module labels in the path
    /// filled in correct order by PATTriggerProducer;
    /// modules not necessarily in filter collection;
    /// consumes disc space
    std::vector<std::string> modules_;
    /// Indeces of trigger filters in pat::TriggerFilterCollection in event
    /// as produced together with the pat::TriggerPathCollection;
    /// also filled in correct order by PATTriggerProducer;
    /// indices of active filters in filter collection
    std::vector<unsigned> filterIndices_;
    /// Index of the last active filter in the list of modules
    unsigned lastActiveFilterSlot_;
    /// Number of modules identified as L3 filters by the 'saveTags' parameter
    /// available starting from CMSSW_4_2_3
    unsigned l3Filters_;
    /// List of L1 seeds and their decisions
    L1SeedCollection l1Seeds_;

  public:
    /// Constructors and Desctructor

    /// Default constructor
    TriggerPath();
    /// Constructor from path name only
    TriggerPath(const std::string& name);
    /// Constructor from values
    TriggerPath(const std::string& name,
                unsigned index,
                double prescale,
                bool run,
                bool accept,
                bool error,
                unsigned lastActiveFilterSlot,
                unsigned l3Filters = 0);

    /// Destructor
    virtual ~TriggerPath() = default;

    /// Methods

    /// Set the path name
    void setName(const std::string& name) { name_ = name; };
    /// Set the path index
    void setIndex(unsigned index) { index_ = index; };
    /// Set the path pre-scale
    void setPrescale(double prescale) { prescale_ = prescale; };
    /// Set the run flag
    void setRun(bool run) { run_ = run; };
    /// Set the success flag
    void setAccept(bool accept) { accept_ = accept; };
    /// Set the error flag
    void setError(bool error) { error_ = error; };
    /// Set the index of the last active filter
    void setLastActiveFilterSlot(unsigned lastActiveFilterSlot) { lastActiveFilterSlot_ = lastActiveFilterSlot; };
    /// Set the number of modules identified as L3 filter
    void setL3Filters(unsigned l3Filters) { l3Filters_ = l3Filters; };
    /// Add a new module label
    void addModule(const std::string& name) { modules_.push_back(name); };
    /// Add a new trigger fillter collection index
    void addFilterIndex(const unsigned index) { filterIndices_.push_back(index); };
    /// Add a new L1 seed
    void addL1Seed(const L1Seed& seed) { l1Seeds_.push_back(seed); };
    void addL1Seed(bool decision, const std::string& expression) { l1Seeds_.push_back(L1Seed(decision, expression)); };
    /// Get the path name
    const std::string& name() const { return name_; };
    /// Get the path index
    unsigned index() const { return index_; };
    /// Get the path pre-scale
    template <typename T = unsigned int>
    T prescale() const {
      static_assert(std::is_same_v<T, double>,
                    "\n\tPlease use prescale<double>"
                    "\n\t(other types for prescales are not supported anymore by pat::TriggerPath");
      return prescale_;
    };
    /// Get the run flag
    bool wasRun() const { return run_; };
    /// Get the success flag
    bool wasAccept() const { return accept_; };
    /// Get the error flag
    bool wasError() const { return error_; };
    /// Get the index of the last active filter
    unsigned lastActiveFilterSlot() const { return lastActiveFilterSlot_; };
    /// Get the number of modules identified as L3 filter
    /// available starting from CMSSW_4_2_3
    unsigned l3Filters() const { return l3Filters_; };
    /// Determines, if the path is a x-trigger, based on the number of modules identified as L3 filter
    /// available starting from CMSSW_4_2_3
    bool xTrigger() const { return (l3Filters_ > 2); };
    /// Get all module labels
    const std::vector<std::string>& modules() const { return modules_; };
    /// Get all trigger fillter collection indeces
    const std::vector<unsigned>& filterIndices() const { return filterIndices_; };
    /// Get the index of a certain module;
    /// returns size of 'modules_' ( modules().size() ) if name is unknown
    /// and -1 if list of modules is not filled
    int indexModule(const std::string& name) const;
    /// Get all L1 seeds
    const L1SeedCollection& l1Seeds() const { return l1Seeds_; };
    /// Get names of all L1 seeds with a certain decision
    std::vector<std::string> l1Seeds(const bool decision) const;
    /// Get names of all succeeding L1 seeds
    std::vector<std::string> acceptedL1Seeds() const { return l1Seeds(true); };
    /// Get names of all failing L1 seeds
    std::vector<std::string> failedL1Seeds() const { return l1Seeds(false); };
  };

  /// Collection of TriggerPath
  typedef std::vector<TriggerPath> TriggerPathCollection;
  /// Persistent reference to an item in a TriggerPathCollection
  typedef edm::Ref<TriggerPathCollection> TriggerPathRef;
  /// Persistent reference to a TriggerPathCollection product
  typedef edm::RefProd<TriggerPathCollection> TriggerPathRefProd;
  /// Vector of persistent references to items in the same TriggerPathCollection
  typedef edm::RefVector<TriggerPathCollection> TriggerPathRefVector;
  /// Const iterator over vector of persistent references to items in the same TriggerPathCollection
  typedef edm::RefVectorIterator<TriggerPathCollection> TriggerPathRefVectorIterator;

}  // namespace pat

#endif
