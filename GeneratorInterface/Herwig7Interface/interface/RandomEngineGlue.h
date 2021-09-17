#ifndef GeneratorInterface_Herwig7Interface_RandomEngineGlue_h
#define GeneratorInterface_Herwig7Interface_RandomEngineGlue_h

#include <string>

#include <ThePEG/Interface/ClassDocumentation.h>
#include <ThePEG/Interface/InterfacedBase.h>
#include <ThePEG/Interface/Parameter.h>
#include <ThePEG/Utilities/ClassTraits.h>

#include <ThePEG/Repository/StandardRandom.h>

#include "GeneratorInterface/Herwig7Interface/interface/Proxy.h"
#include "CLHEP/Random/RandomEngine.h"

namespace CLHEP {
  class HepRandomEngine;  // forward declaration
}

namespace ThePEG {

  class RandomEngineGlue : public RandomGenerator {
  public:
    RandomEngineGlue();
    ~RandomEngineGlue() override;

    void setRandomEngine(CLHEP::HepRandomEngine* v) { randomEngine = v; }
    CLHEP::HepRandomEngine* getRandomEngine() const { return randomEngine; }
    void flush();

    static void Init();

    class Proxy : public ThePEG::Proxy<Proxy> {
    public:
      RandomEngineGlue* getInstance() const { return instance; }

      CLHEP::HepRandomEngine* getRandomEngine() const { return randomEngine; }
      void setRandomEngine(CLHEP::HepRandomEngine* v) { randomEngine = v; }

    private:
      friend class RandomEngineGlue;
      friend class ThePEG::Proxy<Proxy>;

      inline Proxy(ProxyID id) : Base(id), instance(nullptr) {}

      RandomEngineGlue* instance;

      // I do not like putting this here, but I could not
      // think of an alternative without modifying the
      // external code in ThePEG. The problem is the
      // function ThePEG::Repository::makeRun both
      // sets the pointer in the proxy and uses the
      // engine. There is no opportunity to set the
      // engine pointer before it is used without passing
      // it in through the proxy.
      CLHEP::HepRandomEngine* randomEngine;
    };

  protected:
    void fill() override;
    void setSeed(long seed) override;

    IBPtr clone() const override { return new_ptr(*this); }
    IBPtr fullclone() const override { return new_ptr(*this); }

    void doinit() noexcept(false) override;

  private:
    Proxy::ProxyID proxyID;
    CLHEP::HepRandomEngine* randomEngine;

    static ClassDescription<RandomEngineGlue> initRandomEngineGlue;
  };

  template <>
  struct BaseClassTrait<RandomEngineGlue, 1> : public ClassTraitsType {
    /** Typedef of the first base class of RandomEngineGlue. */
    typedef RandomGenerator NthBase;
  };

  /** This template specialization informs ThePEG about the name of the
 *  RandomEngineGlue class. */
  template <>
  struct ClassTraits<RandomEngineGlue> : public ClassTraitsBase<RandomEngineGlue> {
    /** Return a platform-independent class name */
    static string className() { return "ThePEG::RandomEngineGlue"; }
    static string library() { return "libGeneratorInterfaceHerwig7Interface.so"; }
  };

}  // namespace ThePEG

#endif  // GeneratorInterface_Herwig7Interface_RandomEngineGlue_h
