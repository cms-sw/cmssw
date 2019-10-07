#ifndef PhysicsTools_Utilities_FunctClone_h
#define PhysicsTools_Utilities_FunctClone_h

#include <vector>
#include <algorithm>
#include <memory>
#include <cassert>

namespace funct {

  template <typename F>
  struct Master {
    Master(const F& f) : f_(new F(f)), toBeUpdated_(1, true) {}
    double operator()() const { return get(0); }
    double operator()(double x) const { return get(0, x); }
    void add() const { toBeUpdated_.resize(size() + 1, true); }
    size_t size() const { return toBeUpdated_.size(); }
    double get(size_t i) const {
      if (toBeUpdated_[i])
        update();
      toBeUpdated_[i] = true;
      return value_;
    }
    double get(size_t i, double x) const {
      if (toBeUpdated_[i])
        update(x);
      toBeUpdated_[i] = true;
      return value_;
    }

  private:
    void reset() const { std::fill(toBeUpdated_.begin(), toBeUpdated_.end(), true); }
    void clear() const { std::fill(toBeUpdated_.begin(), toBeUpdated_.end(), false); }
    void update() const {
      clear();
      value_ = (*f_)();
    }
    void update(double x) const {
      clear();
      value_ = (*f_)(x);
    }
    const std::shared_ptr<F> f_;
    mutable double value_;
    mutable std::vector<bool> toBeUpdated_;
  };

  template <typename F>
  struct Slave {
    Slave(const Master<F>& master) : master_(master), id_(master.size()) {
      assert(id_ > 0);
      master_.add();
    }
    double operator()() const { return master_.get(id_); }
    double operator()(double x) const { return master_.get(id_, x); }
    void setId(size_t i) { id_ = i; }

  private:
    const Master<F>& master_;
    size_t id_;
  };

  template <typename F>
  Master<F> master(const F& f) {
    return Master<F>(f);
  }

  template <typename F>
  Slave<F> slave(const Master<F>& m) {
    return Slave<F>(m);
  }
}  // namespace funct

#endif
