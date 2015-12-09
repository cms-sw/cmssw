#ifndef DataFormats_Provenance_ProcessHistory_h
#define DataFormats_Provenance_ProcessHistory_h

#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace edm {
  class ProcessHistory {
  public:
    typedef ProcessConfiguration value_type;
    typedef std::vector<value_type> collection_type;

    typedef collection_type::iterator       iterator;
    typedef collection_type::const_iterator const_iterator;

    typedef collection_type::reverse_iterator       reverse_iterator;
    typedef collection_type::const_reverse_iterator const_reverse_iterator;

    typedef collection_type::reference reference;
    typedef collection_type::const_reference const_reference;

    typedef collection_type::size_type size_type;

    ProcessHistory() : data_(), transient_() {}
    explicit ProcessHistory(size_type n) : data_(n), transient_() {}
    explicit ProcessHistory(collection_type const& vec) : data_(vec), transient_() {}

    template<typename... Args>
    void emplace_back(Args&&... args) {data_.emplace_back(std::forward<Args>(args)...); phid() = ProcessHistoryID();}

    void push_front(const_reference t) {data_.insert(data_.begin(), t); phid() = ProcessHistoryID();}
    void push_back(const_reference t) {data_.push_back(t); phid() = ProcessHistoryID();}
    void swap(ProcessHistory& other) {data_.swap(other.data_); phid().swap(other.phid());}
    bool empty() const {return data_.empty();}
    size_type size() const {return data_.size();}
    size_type capacity() const {return data_.capacity();}
    void reserve(size_type n) {data_.reserve(n);}

    reference operator[](size_type i) {return data_[i];}
    const_reference operator[](size_type i) const {return data_[i];}

    reference at(size_type i) {return data_.at(i);}
    const_reference at(size_type i) const {return data_.at(i);}

    const_iterator begin() const {return data_.begin();}
    const_iterator end() const {return data_.end();}

    const_reverse_iterator rbegin() const {return data_.rbegin();}
    const_reverse_iterator rend() const {return data_.rend();}

//     iterator begin() {return data_.begin();}
//     iterator end() {return data_.end();}

//     reverse_iterator rbegin() {return data_.rbegin();}
//     reverse_iterator rend() {return data_.rend();}

    collection_type const& data() const {return data_;}
    ProcessHistoryID id() const;
    ProcessHistoryID setProcessHistoryID();

    // Return true, and fill in config appropriately, if the a process
    // with the given name is recorded in this ProcessHistory. Return
    // false, and do not modify config, if process with the given name
    // is not found.
    bool getConfigurationForProcess(std::string const& name, ProcessConfiguration& config) const;

    void clear() {
      data_.clear();
      phid() = ProcessHistoryID();
    }

    ProcessHistory& reduce();

    void initializeTransients() {transient_.reset();}

    struct Transients {
      Transients() : phid_() {}
      void reset() {phid_.reset();}
      ProcessHistoryID phid_;
    };

  private:
    ProcessHistoryID& phid() {return transient_.phid_;}
    collection_type data_;
    Transients transient_;
  };

  // Free swap function
  inline
  void
  swap(ProcessHistory& a, ProcessHistory& b) {
    a.swap(b);
  }

  inline
  bool
  operator==(ProcessHistory const& a, ProcessHistory const& b) {
    return a.data() == b.data();
  }

  inline
  bool
  operator!=(ProcessHistory const& a, ProcessHistory const& b) {
    return !(a==b);
  }

  bool
  isAncestor(ProcessHistory const& a, ProcessHistory const& b);

  inline
  bool
  isDescendant(ProcessHistory const& a, ProcessHistory const& b) {
    return isAncestor(b, a);
  }

  std::ostream& operator<<(std::ostream& ost, ProcessHistory const& ph);
}

#endif
