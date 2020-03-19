#ifndef gen_FortranInstance_h
#define gen_FortranInstance_h

#include <string>

namespace gen {

  // the callbacks from Pythia6/Herwig6 which are passed to the FortranInstance
  extern "C" {
  void upinit_();
  void upevnt_();
  void upveto_(int *);
  }

  class FortranInstance {
  public:
    FortranInstance() : instanceNesting(0) {}
    virtual ~FortranInstance() noexcept(false);

    void call(void (&fn)()) {
      InstanceWrapper wrapper(this);
      fn();
    }
    template <typename T>
    T call(T (&fn)()) {
      InstanceWrapper wrapper(this);
      return fn();
    }
    template <typename A>
    void call(void (&fn)(A), A a) {
      InstanceWrapper wrapper(this);
      fn(a);
    }
    template <typename T, typename A>
    T call(T (&fn)(A), A a) {
      InstanceWrapper wrapper(this);
      return fn(a);
    }
    template <typename A1, typename A2>
    void call(void (&fn)(A1, A2), A1 a1, A2 a2) {
      InstanceWrapper wrapper(this);
      fn(a1, a2);
    }
    template <typename T, typename A1, typename A2>
    T call(T (&fn)(A1, A2), A1 a1, A2 a2) {
      InstanceWrapper wrapper(this);
      return fn(a1, a2);
    }

    // if a member is instantiated at the beginning of a method,
    // it makes sure this FortranInstance instance is kept as
    // current instance during the execution of the method
    // This wrapper makes the enter()..leave() cycle exception-safe
    struct InstanceWrapper {
      InstanceWrapper(FortranInstance *instance) {
        this->instance = instance;
        instance->enter();
      }

      ~InstanceWrapper() { instance->leave(); }

      FortranInstance *instance;
    };

    // set this instance to be the current one
    // will throw exception when trying to reenter Herwig twice
    virtual void enter();

    // leave instance
    // will throw if the currently running instance does not match
    virtual void leave();

    // get the currently running instance (from enterInstance)
    // intended for callbacks from Fortran
    template <typename T>
    static T *getInstance() {
      T *instance = dynamic_cast<T *>(currentInstance);
      if (!instance)
        throwMissingInstance();
      return instance;
    }

    // Fortran callbacks
    virtual void upInit();
    virtual void upEvnt();
    virtual bool upVeto();

    static const std::string kFortranInstance;

  private:
    // list all the Fortran callbacks here
    friend void gen::upinit_();
    friend void gen::upevnt_();
    friend void gen::upveto_(int *);

    // internal methods
    static void throwMissingInstance();

    // how many times enter() was called
    // this is to make sure leave() will release the instance
    // after the same number of calls
    // nesting can in theory occur if the Fortran callback calls
    // into Herwig again
    int instanceNesting;

    // this points to the Herwig instance that is currently being executed
    static FortranInstance *currentInstance;
  };

}  // namespace gen

#endif  // gen_FortranInstance_h
