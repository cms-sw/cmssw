COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram<double, double>);
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram<float, float>);

COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram2D<double, double, double>);
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram2D<float, float, float>);

COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram3D<double, double, double, double>);
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Histogram3D<float, float, float, float>);

COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Range<double>);
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Range<float>);


COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcOptional)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcCount)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcClassed)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcSplitter)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcForeach)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcSort)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcCategory)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcNormalize)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcLikelihood)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcLinear)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcMultiply)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcMatrix)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcExternal)
COND_SERIALIZATION_REGISTER_POLYMORPHIC(PhysicsTools::Calibration::ProcMLP)
