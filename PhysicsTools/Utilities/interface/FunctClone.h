#ifndef PhysicsTools_Utilities_FunctClone_h
#define PhysicsTools_Utilities_FunctClone_h
#include <boost/shared_ptr.hpp>

namespace funct {

  template<typename F>
  struct Master {
    Master(const F& f) : f_(f), cached_(false) { }
    double operator()() const {
      return cache();
    }
    double operator()(double x) const {
      return cache(x);
    }
    double cacheValue() const {
      if(cached_) return value_;
      else return (*this)();
    }
    double cacheValue(double x) const {
      if(cached_) return value_;
      else return (*this)(x);
    }
  private:
    double cache() const {
      cached_ = true;
      return value_ = f_();
    }
    double cache(double x) const {
      cached_ = true;
      return value_ = f_(x);
    }
    const F & f_;
    mutable double value_;
    mutable bool cached_;
  };

  template<typename F>
  struct Slave {
    Slave(const Master<F>& master) : master_(master) { }
    double operator()() const { return master_.cacheValue(); }
    double operator()(double x) const { return master_.cacheValue(x); }
  private:
    const Master<F> & master_;
  };

  template<typename F>
  Master<F> master(const F& f) { return Master<F>(f); }

  template<typename F>
  Slave<F> slave(const Master<F>& m) { return Slave<F>(m); }
}

#endif
