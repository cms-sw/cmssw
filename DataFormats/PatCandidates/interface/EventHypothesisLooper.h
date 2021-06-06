#ifndef DataFormats_EventHypothesis_interface_EventHypothesisLooper_h
#define DataFormats_EventHypothesis_interface_EventHypothesisLooper_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include <algorithm>

namespace pat {
  namespace eventhypothesis {
    template <typename T>
    class DynCastCandPtr {
    public:
      const T *get(const reco::Candidate *ptr);
      void clearCache() { isPtrCached_ = false; }
      bool typeOk(const reco::Candidate *ptr) {
        doPtr(ptr);
        return cachePtr_ != 0;
      }

    private:
      void doPtr(const reco::Candidate *ptr);
      bool isPtrCached_;
      const T *cachePtr_;
    };
    template <typename T>
    void DynCastCandPtr<T>::doPtr(const reco::Candidate *ptr) {
      if (!isPtrCached_) {
        cachePtr_ = dynamic_cast<const T *>(ptr);
        isPtrCached_ = true;
      }
    }
    template <typename T>
    const T *DynCastCandPtr<T>::get(const reco::Candidate *ptr) {
      doPtr(ptr);
      if ((ptr != nullptr) && (cachePtr_ == nullptr))
        throw cms::Exception("Type Checking")
            << "You can't convert a " << typeid(*ptr).name() << " to a " << typeid(T).name() << "\n"
            << "note: you can use c++filt command to convert the above in human readable types.\n";
      return cachePtr_;
    }

    template <>
    struct DynCastCandPtr<reco::Candidate> {
      const reco::Candidate *get(const reco::Candidate *ptr) { return ptr; }
      void clearCache() {}
      bool typeOk(const reco::Candidate *ptr) { return true; }
    };

    template <typename T>
    class Looper {
    public:
      /// Looper from EventHypothesis and an external, not owned, ParticleFilter.
      /// That is: MyFilter flt; Looper(eh, flt);
      Looper(const EventHypothesis &eh, const ParticleFilter &filter);

      /// Looper from EventHypothesis and an internal, owned, ParticleFilter
      /// That is: Looper(eh, new MyFilter());
      Looper(const EventHypothesis &eh, const ParticleFilter *filter);
      /// Looper from EventHypothesis and a shared ParticleFilter
      /// That is: Looper(eh, ParticleFilterPtr(new MyFilter()));
      Looper(const EventHypothesis &eh, const ParticleFilterPtr &filter);
      ~Looper() {}

      /// Accessor as if it was a const_iterator on a list of T
      const T &operator*() const { return ptr_.get(iter_->second.get()); }
      /// Accessor as if it was a const_iterator on a list of T
      const T *operator->() const { return ptr_.get(iter_->second.get()); }
      /// Accessor as if it was a smart pointer to const T *
      const T *get() const { return ptr_.get(iter_->second.get()); }

      /// test if the type is correct
      bool isTypeOk() const { return ptr_.typeOk(iter_->second.get()); }

      /// Role of pointed item
      const std::string &role() const { return iter_->first; }
      /// EDM Ref to pointed particle
      const CandRefType &ref() const { return iter_->second; }
      /// C++ reference to pointed particle
      const reco::Candidate &cand() const { return *iter_->second; }

      /// Index of this item in the full EventHypothesis
      size_t globalIndex() { return iter_ - eh_.begin(); }
      /// Index of this item among those in the loop
      size_t index() const { return num_; }
      /// Number of particles in the loop
      size_t size() const {
        if (total_ < 0)
          realSize();
        return total_;
      }

      /// iteration
      Looper &operator++();
      /// iteration
      Looper &operator--();
      /// skip (might be slow)
      Looper &skip(int delta);
      /// Reset to the start or to any other specific item; negatives count from the end.
      /// might be slow, especially with negative items
      Looper &reset(int item = 0);

      /// Returns true if you have not run out of the boundaries.
      /// It does NOT check if typeOk()
      operator bool() const;

      /// returns true if loopers point to the same record
      template <typename T2>
      bool operator==(const Looper<T2> &other) const {
        return iter_ == other.iter_;
      }
      template <typename T2>
      bool operator!=(const Looper<T2> &other) const {
        return iter_ != other.iter_;
      }
      template <typename T2>
      bool operator<=(const Looper<T2> &other) const {
        return iter_ <= other.iter_;
      }
      template <typename T2>
      bool operator>=(const Looper<T2> &other) const {
        return iter_ >= other.iter_;
      }
      template <typename T2>
      bool operator<(const Looper<T2> &other) const {
        return iter_ < other.iter_;
      }
      template <typename T2>
      bool operator>(const Looper<T2> &other) const {
        return iter_ > other.iter_;
      }

    private:
      struct null_deleter {
        void operator()(void const *) const {}
      };
      typedef typename EventHypothesis::const_iterator const_iterator;

      void first();
      void realSize() const;
      bool assertOk() const;

      const EventHypothesis &eh_;
      const ParticleFilterPtr filter_;
      const_iterator iter_;
      int num_;
      mutable int total_;  // mutable as it is not computed unless needed
      mutable DynCastCandPtr<T> ptr_;
    };
    typedef Looper<reco::Candidate> CandLooper;

    template <typename T>
    Looper<T>::Looper(const EventHypothesis &eh, const ParticleFilter &filter)
        : eh_(eh), filter_(ParticleFilterPtr(&filter, typename Looper<T>::null_deleter())), total_(-1) {
      first();
    }

    template <typename T>
    Looper<T>::Looper(const EventHypothesis &eh, const ParticleFilter *filter) : eh_(eh), filter_(filter), total_(-1) {
      first();
    }

    template <typename T>
    Looper<T>::Looper(const EventHypothesis &eh, const ParticleFilterPtr &filter)
        : eh_(eh), filter_(filter), total_(-1) {
      first();
    }

    template <typename T>
    bool Looper<T>::assertOk() const {
      assert(iter_ <= eh_.end());
      assert((iter_ + 1) >= eh_.begin());
      assert((iter_ < eh_.begin()) || (iter_ == eh_.end()) || ((*filter_)(*iter_)));
      return true;
    }

    template <typename T>
    Looper<T> &Looper<T>::operator++() {
      ptr_.clearCache();
      assert(assertOk());
      if (iter_ == eh_.end())
        return *this;
      do {
        ++iter_;
        if (iter_ == eh_.end())
          break;
        if ((*filter_)(*iter_)) {
          assert(assertOk());
          ++num_;
          return *this;
        }
      } while (true);
      assert(assertOk());
      return *this;
    }
    template <typename T>
    Looper<T> &Looper<T>::operator--() {
      ptr_.clearCache();
      assert(assertOk());
      if (num_ < 0)
        return *this;
      do {
        --iter_;
        if (iter_ < eh_.begin()) {
          num_ = -1;
          break;
        }
        if ((*filter_)(*iter_)) {
          assert(assertOk());
          --num_;
          return *this;
        }
      } while (true);
      assert(assertOk());
      return *this;
    }

    template <typename T>
    Looper<T> &Looper<T>::skip(int delta) {
      assert(assertOk());
      std::advance(this, delta);
      assert(assertOk());
      return *this;
    }

    template <typename T>
    Looper<T> &Looper<T>::reset(int item) {
      assert(assertOk());
      if (item >= 0) {
        first();
        std::advance(this, item);
      } else {
        num_ = item + 1;
        iter_ = eh_.end();
        std::advance(this, item);
      }
      assert(assertOk());
      return *this;
    }

    template <typename T>
    void Looper<T>::first() {
      num_ = 0;
      iter_ = eh_.begin();
      ptr_.clearCache();
      for (; iter_ != eh_.end(); ++iter_) {
        if ((*filter_)(*iter_))
          break;
      }
      assert(assertOk());
    }

    template <typename T>
    Looper<T>::operator bool() const {
      return (iter_ < eh_.end()) && (iter_ >= eh_.begin());
    }

    template <typename T>
    void Looper<T>::realSize() const {
      EventHypothesis::const_iterator it = iter_;
      if (it < eh_.begin()) {
        it = eh_.begin();
        total_ = 0;
      } else {
        total_ = num_;
      }
      for (; it != eh_.end(); ++it) {
        if ((*filter_)(*it))
          ++total_;
      }
    }
  }  // namespace eventhypothesis
}  // namespace pat

#endif
