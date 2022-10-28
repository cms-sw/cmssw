#ifndef DataFormats_L1Trigger_RegionalOutput_h
#define DataFormats_L1Trigger_RegionalOutput_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include <vector>

namespace l1t {
  template <typename T>
  class RegionalOutput {
  public:
    typedef typename T::value_type value_type;
    typedef edm::Ref<T> ref;
    typedef edm::RefProd<T> refprod;

    class iterator {
    public:
      typedef typename T::value_type value_type;
      typedef ptrdiff_t difference_type;
      iterator(const RegionalOutput<T>& src, unsigned int idx) : src_(&src), idx_(idx) {}
      iterator(iterator const& it) : src_(it.src_), idx_(it.idx_) {}
      iterator() : src_(nullptr), idx_(0) {}
      iterator& operator++() {
        ++idx_;
        return *this;
      }
      iterator operator++(int) {
        iterator ci = *this;
        ++idx_;
        return ci;
      }
      iterator& operator--() {
        --idx_;
        return *this;
      }
      iterator operator--(int) {
        iterator ci = *this;
        --idx_;
        return ci;
      }
      difference_type operator-(iterator const& o) const { return idx_ - o.idx_; }
      iterator operator+(difference_type n) const { return iterator(src_, idx_ + n); }
      iterator operator-(difference_type n) const { return iterator(src_, idx_ - n); }
      bool operator<(iterator const& o) const { return idx_ < o.idx_; }
      bool operator==(iterator const& ci) const { return idx_ == ci.idx_; }
      bool operator!=(iterator const& ci) const { return idx_ != ci.idx_; }
      value_type const& operator*() const { return src_->objAt(idx_); }
      value_type const* operator->() const { return &src_->objAt(idx_); }
      iterator& operator+=(difference_type d) {
        idx_ += d;
        return *this;
      }
      iterator& operator-=(difference_type d) {
        idx_ -= d;
        return *this;
      }
      value_type const& operator[](difference_type d) const { return src_->objAt(idx_ + d); }
      // interface to get EDM refs & related stuff
      edm::Ref<T> ref() const { return src_->refAt(idx_); }
      edm::ProductID id() const { return src_->id(); }
      unsigned int idx() const { return idx_; }
      unsigned int key() const { return idx_; }

    private:
      const RegionalOutput<T>* src_;
      unsigned int idx_;
    };
    typedef iterator const_iterator;

    class Region {
    public:
      typedef typename T::value_type value_type;
      typedef typename RegionalOutput<T>::iterator iterator;
      typedef typename RegionalOutput<T>::const_iterator const_iterator;

      const value_type& operator[](unsigned int idx) const { return src_->objAt(ibegin_ + idx); }
      const value_type& front() const { return src_->objAt(ibegin_); }
      const value_type& back() const { return src_->objAt(iend_ - 1); }
      iterator begin() const { return iterator(*src_, ibegin_); }
      iterator end() const { return iterator(*src_, iend_); }
      unsigned int size() const { return iend_ - ibegin_; }
      bool empty() const { return (iend_ == ibegin_); }
      // interface to get EDM refs & related stuff
      ref refAt(unsigned int idx) const { return src_->refAt(ibegin_ + idx); }
      edm::ProductID id() const { return src_->id(); }

    private:
      const RegionalOutput<T>* src_;
      unsigned int ibegin_, iend_;
      friend class RegionalOutput<T>;
      Region(const RegionalOutput<T>* src, unsigned int ibegin, unsigned int iend)
          : src_(src), ibegin_(ibegin), iend_(iend) {}
    };

    RegionalOutput() : refprod_(), values_(), regions_(), etas_(), phis_() {}
    RegionalOutput(const edm::RefProd<T>& prod) : refprod_(prod), values_(), regions_(), etas_(), phis_() {}

    void addRegion(const std::vector<int>& indices, const float eta, const float phi) {
      regions_.emplace_back((regions_.empty() ? 0 : regions_.back()) + indices.size());
      values_.insert(values_.end(), indices.begin(), indices.end());
      etas_.push_back(eta);
      phis_.push_back(phi);
    }

    edm::ProductID id() const { return refprod_.id(); }
    unsigned int size() const { return values_.size(); }
    unsigned int nRegions() const { return regions_.size(); }
    bool empty() const { return values_.empty(); }
    void clear() {
      values_.clear();
      regions_.clear();
      etas_.clear();
      phis_.clear();
    }
    void shrink_to_fit() {
      values_.shrink_to_fit();
      regions_.shrink_to_fit();
      etas_.shrink_to_fit();
      phis_.shrink_to_fit();
    }

    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, values_.size()); }

    Region region(unsigned int ireg) const {
      if (ireg >= regions_.size())
        throw cms::Exception("Region index out of bounds");
      return Region(this, ireg == 0 ? 0 : regions_[ireg - 1], regions_[ireg]);
    }

    const float eta(unsigned int ireg) const { return etas_[ireg]; }
    const float phi(unsigned int ireg) const { return phis_[ireg]; }

    ref refAt(unsigned int idx) const { return ref(refprod_, values_[idx]); }
    const value_type& objAt(unsigned int idx) const { return (*refprod_)[values_[idx]]; }

    //Used by ROOT storage
    CMS_CLASS_VERSION(3)

  protected:
    refprod refprod_;
    std::vector<unsigned int> values_;   // list of indices to objects in each region, flattened.
    std::vector<unsigned int> regions_;  // for each region, store the index of one-past the last object in values
    std::vector<float> etas_;            // floatEtaCenter of each PFregion
    std::vector<float> phis_;            // floatPhiCenter of each PFregion
  };
}  // namespace l1t
#endif
