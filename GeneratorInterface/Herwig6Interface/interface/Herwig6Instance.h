#ifndef gen_Herwig6Instance_h
#define gen_Herwig6Instance_h

#include "GeneratorInterface/Core/interface/FortranInstance.h"
#include <memory>

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  // the callbacks from Herwig which are passed on to the Herwig6Instance
  extern "C" {
  double hwrgen_(int *);
  void cms_hwwarn_(char fn[6], int *, int *);
  }

  //Forward declare here to avoid system dependency
  struct TimeoutHolder;

  class Herwig6Instance : public FortranInstance {
  public:
    Herwig6Instance();
    ~Herwig6Instance() override;

    // passes a configuration parameter
    bool give(const std::string &line);

    bool callWithTimeout(unsigned int secs, void (*fn)()) {
      InstanceWrapper wrapper(this);
      return timeout(secs, fn);
    }

    // method to open External Particle Spectra Files
    void openParticleSpecFile(const std::string fileName);

    void setHerwigRandomEngine(CLHEP::HepRandomEngine *v) { randomEngine = v; }

  protected:
    // intercept HERWIG warnings and errors (default: no)
    virtual bool hwwarn(const std::string &fn, int code);

  private:
    // list all the Fortran callbacks here
    friend double gen::hwrgen_(int *);
    friend void gen::cms_hwwarn_(char fn[6], int *, int *);

    // call with timeout
    bool timeout(unsigned int secs, void (*fn)());

    // used from timeout
    static void _timeout_sighandler(int signr);

    // the random engine
    // in order to make sure no one else uses this engine
    // the access is only possible if pyr_ is called after
    // the current instance has been selected using enter()
    // if Herwig is called without an active instance and does
    // either a callback or requests a random number an
    // an exception will be thrown
    CLHEP::HepRandomEngine *randomEngine;

    // for timeout facility
    std::unique_ptr<TimeoutHolder> timeoutPrivate;
  };

}  // namespace gen

#endif  // gen_Herwig6Instance_h
