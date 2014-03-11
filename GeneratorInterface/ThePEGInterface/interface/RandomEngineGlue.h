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

	void flush();

	static void Init();

	class Proxy : public ThePEG::Proxy<Proxy> {
	    public:
		RandomEngineGlue *getInstance() const { return instance; }

	    private:
		friend class RandomEngineGlue;
		friend class ThePEG::Proxy<Proxy>;

		inline Proxy(ProxyID id) : Base(id), instance(0) {}

		RandomEngineGlue *instance;
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
