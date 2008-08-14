#ifndef DataFormats_Provenance_ProcessHistory_h
#define DataFormats_Provenance_ProcessHistory_h

#include <iosfwd>
#include <string>
#include <vector>

#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

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

    ProcessHistory() : data_(), id_() {}
    explicit ProcessHistory(size_type n) : data_(n), id_() {}
    explicit ProcessHistory(collection_type const& vec) : data_(vec), id_() {}

    void push_back(const_reference t) {data_.push_back(t); id_=ProcessHistoryID();}
    void swap(ProcessHistory& other) {data_.swap(other.data_); id_.swap(other.id_);}
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

    // Return true, and fill in config appropriately, if the a process
    // with the given name is recorded in this ProcessHistory. Return
    // false, and do not modify config, if process with the given name
    // is found.
    bool getConfigurationForProcess(std::string const& name, ProcessConfiguration& config) const;

  private:
    collection_type data_;
    mutable ProcessHistoryID id_;
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
  isDescendent(ProcessHistory const& a, ProcessHistory const& b) {
    return isAncestor(b, a);
  }

  std::ostream& operator<<(std::ostream& ost, ProcessHistory const& ph);
}

#endif
