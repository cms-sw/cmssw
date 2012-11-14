#include "JetMETCorrections/InterpolationTables/interface/StorableMultivariateFunctorReader.h"

// Include headers for all classes derived from StorableMultivariateFunctor
// which are known at this point in code development
//
#include "JetMETCorrections/InterpolationTables/interface/InterpolationFunctorInstances.h"
#include "JetMETCorrections/InterpolationTables/interface/HistoNDFunctorInstances.h"

// Simple macro for adding a reader for a class derived from 
// StorableMultivariateFunctor
#define add_reader(Derived) do {                                        \
    const gs::ClassId& id(gs::ClassId::makeId<Derived >());             \
    (*this)[id.name()] =                                                \
        new gs::ConcreteReader<StorableMultivariateFunctor,Derived >(); \
} while(0);

namespace npstat {
    StorableMultivariateFunctorReader::StorableMultivariateFunctorReader()
    {
        add_reader(DoubleInterpolationFunctor);
        add_reader(DoubleUAInterpolationFunctor);
        add_reader(DoubleNUInterpolationFunctor);
        add_reader(FloatInterpolationFunctor);
        add_reader(FloatUAInterpolationFunctor);
        add_reader(FloatNUInterpolationFunctor);

        add_reader(DoubleHistoNDFunctor);
        add_reader(DoubleUAHistoNDFunctor);
        add_reader(DoubleNUHistoNDFunctor);
        add_reader(FloatHistoNDFunctor);
        add_reader(FloatUAHistoNDFunctor);
        add_reader(FloatNUHistoNDFunctor);
    }
}
