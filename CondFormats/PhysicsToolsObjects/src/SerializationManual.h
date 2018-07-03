
#include "boost/serialization/assume_abstract.hpp"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"

// take care of instantiating the concrete templates:

COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram<double, double>);
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram<float, float>);

COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram2D<double, double, double>);
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram2D<float, float, float>);

COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram3D<double, double, double, double>);
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram3D<float, float, float, float>);

COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Range<double>);
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Range<float>);

// take care of inhertitance chains:

BOOST_SERIALIZATION_ASSUME_ABSTRACT(PerformancePayload);

COND_SERIALIZABLE_POLYMORPHIC(PerformancePayloadFromTable)
COND_SERIALIZABLE_POLYMORPHIC(PerformancePayloadFromTFormula)
COND_SERIALIZABLE_POLYMORPHIC(PerformancePayloadFromBinnedTFormula)


BOOST_SERIALIZATION_ASSUME_ABSTRACT(PhysicsTools::Calibration::VarProcessor); 

COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::VarProcessor)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcOptional)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcCount)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcClassed)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcSplitter)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcForeach)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcSort)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcCategory)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcNormalize)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcLikelihood)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcLinear)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcMultiply)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcMatrix)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcExternal)
COND_SERIALIZABLE_POLYMORPHIC(PhysicsTools::Calibration::ProcMLP)
