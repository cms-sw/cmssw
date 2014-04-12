#ifndef GeneratorInterface_ThePEGInterface_RandomEngineGlue_h
#define GeneratorInterface_ThePEGInterface_RandomEngineGlue_h

#include <string>

#include <boost/shared_ptr.hpp>

#include <ThePEG/Interface/ClassDocumentation.h>
#include <ThePEG/Interface/InterfacedBase.h>
#include <ThePEG/Interface/Parameter.h>
#include <ThePEG/Utilities/ClassTraits.h>

#include <ThePEG/Repository/StandardRandom.h>

#include "GeneratorInterface/ThePEGInterface/interface/Proxy.h"

namespace CLHEP {
	class HepRandomEngine;	// forward declaration
}

namespace ThePEG {

class RandomEngineGlue : public RandomGenerator {
    public:
	RandomEngineGlue();
	virtual ~RandomEngineGlue();

	void setRandomEngine(CLHEP::HepRandomEngine* v) { randomEngine = v; }

	void flush();

	static void Init();

	class Proxy : public ThePEG::Proxy<Proxy> {
	    public:
		RandomEngineGlue *getInstance() const { return instance; }

                CLHEP::HepRandomEngine* getRandomEngine() const { return randomEngine; }
                void setRandomEngine(CLHEP::HepRandomEngine* v) { randomEngine = v; }

	    private:
		friend class RandomEngineGlue;
		friend class ThePEG::Proxy<Proxy>;

		inline Proxy(ProxyID id) : Base(id), instance(0) {}

		RandomEngineGlue *instance;

                // I do not like putting this here, but I could not
                // think of an alternative without modifying the
                // external code in ThePEG. The problem is the
                // function ThePEG::Repository::makeRun both
                // sets the pointer in the proxy and uses the
                // engine. There is no opportunity to set the
                // engine pointer before it is used without passing
                // it in through the proxy.
                CLHEP::HepRandomEngine  *randomEngine;
	};

    protected:
	virtual void fill();
	virtual void setSeed(long seed);

	virtual IBPtr clone() const { return new_ptr(*this); }
	virtual IBPtr fullclone() const { return new_ptr(*this); }

	virtual void doinit() throw(InitException);

    private:
	Proxy::ProxyID		proxyID;
	CLHEP::HepRandomEngine  *randomEngine;

	static ClassDescription<RandomEngineGlue> initRandomEngineGlue;
};

template<>
struct BaseClassTrait<RandomEngineGlue, 1> : public ClassTraitsType {
	/** Typedef of the first base class of RandomEngineGlue. */
	typedef RandomGenerator NthBase;
};

/** This template specialization informs ThePEG about the name of the
 *  RandomEngineGlue class. */
template<>
struct ClassTraits<RandomEngineGlue> :
			public ClassTraitsBase<RandomEngineGlue> {
	/** Return a platform-independent class name */
	static string className() { return "ThePEG::RandomEngineGlue"; }
	static string library() { return "libGeneratorInterfaceThePEGInterface.so"; }
};

} // namespace ThePEG

#endif // GeneratorInterface_ThePEGInterface_RandomEngineGlue_h
