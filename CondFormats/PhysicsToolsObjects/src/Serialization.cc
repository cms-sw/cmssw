
#include "CondFormats/PhysicsToolsObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void BinningVariables::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(BinningVariables);

template <class Archive>
void PerformancePayload::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(PerformancePayload);

template <class Archive>
void PerformancePayloadFromBinnedTFormula::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PerformancePayload", boost::serialization::base_object<PerformancePayload>(*this));
    ar & BOOST_SERIALIZATION_NVP(pls);
    ar & BOOST_SERIALIZATION_NVP(results_);
    ar & BOOST_SERIALIZATION_NVP(variables_);
}
COND_SERIALIZATION_INSTANTIATE(PerformancePayloadFromBinnedTFormula);

template <class Archive>
void PerformancePayloadFromTFormula::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PerformancePayload", boost::serialization::base_object<PerformancePayload>(*this));
    ar & BOOST_SERIALIZATION_NVP(pl);
    ar & BOOST_SERIALIZATION_NVP(results_);
    ar & BOOST_SERIALIZATION_NVP(variables_);
}
COND_SERIALIZATION_INSTANTIATE(PerformancePayloadFromTFormula);

template <class Archive>
void PerformancePayloadFromTable::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PerformancePayload", boost::serialization::base_object<PerformancePayload>(*this));
    ar & BOOST_SERIALIZATION_NVP(pl);
    ar & BOOST_SERIALIZATION_NVP(results_);
    ar & BOOST_SERIALIZATION_NVP(binning_);
}
COND_SERIALIZATION_INSTANTIATE(PerformancePayloadFromTable);

template <class Archive>
void PerformanceResult::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(PerformanceResult);

template <class Archive>
void PerformanceWorkingPoint::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cut_);
    ar & BOOST_SERIALIZATION_NVP(dname_);
}
COND_SERIALIZATION_INSTANTIATE(PerformanceWorkingPoint);

template <class Archive>
void PhysicsPerformancePayload::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(stride_);
    ar & BOOST_SERIALIZATION_NVP(table_);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsPerformancePayload);

template <class Archive>
void PhysicsTFormulaPayload::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(limits_);
    ar & BOOST_SERIALIZATION_NVP(formulas_);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTFormulaPayload);

template <class Archive>
void PhysicsTools::Calibration::BitSet::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(store);
    ar & BOOST_SERIALIZATION_NVP(bitsInLast);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::BitSet);

template <typename Value_t, typename AxisX_t, typename AxisY_t>
template <class Archive>
void PhysicsTools::Calibration::Histogram2D<Value_t, AxisX_t, AxisY_t>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(stride);
    ar & BOOST_SERIALIZATION_NVP(binULimitsX);
    ar & BOOST_SERIALIZATION_NVP(binULimitsY);
    ar & BOOST_SERIALIZATION_NVP(binValues);
    ar & BOOST_SERIALIZATION_NVP(limitsX);
    ar & BOOST_SERIALIZATION_NVP(limitsY);
}

template <typename Value_t, typename AxisX_t, typename AxisY_t, typename AxisZ_t>
template <class Archive>
void PhysicsTools::Calibration::Histogram3D<Value_t, AxisX_t, AxisY_t, AxisZ_t>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(strideX);
    ar & BOOST_SERIALIZATION_NVP(strideY);
    ar & BOOST_SERIALIZATION_NVP(binULimitsX);
    ar & BOOST_SERIALIZATION_NVP(binULimitsY);
    ar & BOOST_SERIALIZATION_NVP(binULimitsZ);
    ar & BOOST_SERIALIZATION_NVP(binValues);
    ar & BOOST_SERIALIZATION_NVP(limitsX);
    ar & BOOST_SERIALIZATION_NVP(limitsY);
    ar & BOOST_SERIALIZATION_NVP(limitsZ);
}

template <typename Value_t, typename Axis_t>
template <class Archive>
void PhysicsTools::Calibration::Histogram<Value_t, Axis_t>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(binULimits);
    ar & BOOST_SERIALIZATION_NVP(binValues);
    ar & BOOST_SERIALIZATION_NVP(limits);
}

template <class Archive>
void PhysicsTools::Calibration::MVAComputer::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(inputSet);
    ar & BOOST_SERIALIZATION_NVP(output);
    ar & BOOST_SERIALIZATION_NVP(processors);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::MVAComputer);

template <class Archive>
void PhysicsTools::Calibration::MVAComputerContainer::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(entries);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::MVAComputerContainer);

template <class Archive>
void PhysicsTools::Calibration::Matrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(elements);
    ar & BOOST_SERIALIZATION_NVP(rows);
    ar & BOOST_SERIALIZATION_NVP(columns);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Matrix);

template <class Archive>
void PhysicsTools::Calibration::ProcCategory::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(variableBinLimits);
    ar & BOOST_SERIALIZATION_NVP(categoryMapping);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcCategory);

template <class Archive>
void PhysicsTools::Calibration::ProcClassed::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(nClasses);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcClassed);

template <class Archive>
void PhysicsTools::Calibration::ProcCount::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcCount);

template <class Archive>
void PhysicsTools::Calibration::ProcExternal::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(method);
    ar & BOOST_SERIALIZATION_NVP(store);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcExternal);

template <class Archive>
void PhysicsTools::Calibration::ProcForeach::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(nProcs);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcForeach);

template <class Archive>
void PhysicsTools::Calibration::ProcLikelihood::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(pdfs);
    ar & BOOST_SERIALIZATION_NVP(bias);
    ar & BOOST_SERIALIZATION_NVP(categoryIdx);
    ar & BOOST_SERIALIZATION_NVP(logOutput);
    ar & BOOST_SERIALIZATION_NVP(individual);
    ar & BOOST_SERIALIZATION_NVP(neverUndefined);
    ar & BOOST_SERIALIZATION_NVP(keepEmpty);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcLikelihood);

template <class Archive>
void PhysicsTools::Calibration::ProcLikelihood::SigBkg::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(background);
    ar & BOOST_SERIALIZATION_NVP(signal);
    ar & BOOST_SERIALIZATION_NVP(useSplines);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcLikelihood::SigBkg);

template <class Archive>
void PhysicsTools::Calibration::ProcLinear::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(coeffs);
    ar & BOOST_SERIALIZATION_NVP(offset);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcLinear);

template <class Archive>
void PhysicsTools::Calibration::ProcMLP::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(layers);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcMLP);

template <class Archive>
void PhysicsTools::Calibration::ProcMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(matrix);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcMatrix);

template <class Archive>
void PhysicsTools::Calibration::ProcMultiply::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(in);
    ar & BOOST_SERIALIZATION_NVP(out);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcMultiply);

template <class Archive>
void PhysicsTools::Calibration::ProcNormalize::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(distr);
    ar & BOOST_SERIALIZATION_NVP(categoryIdx);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcNormalize);

template <class Archive>
void PhysicsTools::Calibration::ProcOptional::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(neutralPos);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcOptional);

template <class Archive>
void PhysicsTools::Calibration::ProcSort::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(sortByIndex);
    ar & BOOST_SERIALIZATION_NVP(descending);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcSort);

template <class Archive>
void PhysicsTools::Calibration::ProcSplitter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(nFirst);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::ProcSplitter);

template <typename Axis_t>
template <class Archive>
void PhysicsTools::Calibration::Range<Axis_t>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(min);
    ar & BOOST_SERIALIZATION_NVP(max);
}

template <class Archive>
void PhysicsTools::Calibration::VHistogramD2D::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(vHist);
    ar & BOOST_SERIALIZATION_NVP(vValues);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::VHistogramD2D);

template <class Archive>
void PhysicsTools::Calibration::VarProcessor::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(inputVars);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::VarProcessor);

template <class Archive>
void PhysicsTools::Calibration::Variable::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(name);
}
COND_SERIALIZATION_INSTANTIATE(PhysicsTools::Calibration::Variable);

#include "CondFormats/PhysicsToolsObjects/src/SerializationManual.h"
