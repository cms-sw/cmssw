#ifndef CondFormats_PhysicsToolsObjects_Serialization_H
#define CondFormats_PhysicsToolsObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void BinningVariables::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void PerformancePayload::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void PerformancePayloadFromBinnedTFormula::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PerformancePayload", boost::serialization::base_object<PerformancePayload>(*this));
    ar & BOOST_SERIALIZATION_NVP(pls);
    ar & BOOST_SERIALIZATION_NVP(results_);
    ar & BOOST_SERIALIZATION_NVP(variables_);
}

template <class Archive>
void PerformancePayloadFromTFormula::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PerformancePayload", boost::serialization::base_object<PerformancePayload>(*this));
    ar & BOOST_SERIALIZATION_NVP(pl);
    ar & BOOST_SERIALIZATION_NVP(results_);
    ar & BOOST_SERIALIZATION_NVP(variables_);
}

template <class Archive>
void PerformancePayloadFromTable::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PerformancePayload", boost::serialization::base_object<PerformancePayload>(*this));
    ar & BOOST_SERIALIZATION_NVP(pl);
    ar & BOOST_SERIALIZATION_NVP(results_);
    ar & BOOST_SERIALIZATION_NVP(binning_);
}

template <class Archive>
void PerformanceResult::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void PerformanceWorkingPoint::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cut_);
    ar & BOOST_SERIALIZATION_NVP(dname_);
}

template <class Archive>
void PhysicsPerformancePayload::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(stride_);
    ar & BOOST_SERIALIZATION_NVP(table_);
}

template <class Archive>
void PhysicsTFormulaPayload::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(limits_);
    ar & BOOST_SERIALIZATION_NVP(formulas_);
}

template <class Archive>
void PhysicsTools::Calibration::BitSet::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(store);
    ar & BOOST_SERIALIZATION_NVP(bitsInLast);
}

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

template <class Archive>
void PhysicsTools::Calibration::MVAComputerContainer::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(entries);
}

template <class Archive>
void PhysicsTools::Calibration::Matrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(elements);
    ar & BOOST_SERIALIZATION_NVP(rows);
    ar & BOOST_SERIALIZATION_NVP(columns);
}

template <class Archive>
void PhysicsTools::Calibration::ProcCategory::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(variableBinLimits);
    ar & BOOST_SERIALIZATION_NVP(categoryMapping);
}

template <class Archive>
void PhysicsTools::Calibration::ProcClassed::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(nClasses);
}

template <class Archive>
void PhysicsTools::Calibration::ProcCount::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
}

template <class Archive>
void PhysicsTools::Calibration::ProcExternal::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(method);
    ar & BOOST_SERIALIZATION_NVP(store);
}

template <class Archive>
void PhysicsTools::Calibration::ProcForeach::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(nProcs);
}

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

template <class Archive>
void PhysicsTools::Calibration::ProcLikelihood::SigBkg::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(background);
    ar & BOOST_SERIALIZATION_NVP(signal);
    ar & BOOST_SERIALIZATION_NVP(useSplines);
}

template <class Archive>
void PhysicsTools::Calibration::ProcLinear::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(coeffs);
    ar & BOOST_SERIALIZATION_NVP(offset);
}

template <class Archive>
void PhysicsTools::Calibration::ProcMLP::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(layers);
}

template <class Archive>
void PhysicsTools::Calibration::ProcMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(matrix);
}

template <class Archive>
void PhysicsTools::Calibration::ProcMultiply::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(in);
    ar & BOOST_SERIALIZATION_NVP(out);
}

template <class Archive>
void PhysicsTools::Calibration::ProcNormalize::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(distr);
    ar & BOOST_SERIALIZATION_NVP(categoryIdx);
}

template <class Archive>
void PhysicsTools::Calibration::ProcOptional::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(neutralPos);
}

template <class Archive>
void PhysicsTools::Calibration::ProcSort::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(sortByIndex);
    ar & BOOST_SERIALIZATION_NVP(descending);
}

template <class Archive>
void PhysicsTools::Calibration::ProcSplitter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PhysicsTools::Calibration::VarProcessor", boost::serialization::base_object<PhysicsTools::Calibration::VarProcessor>(*this));
    ar & BOOST_SERIALIZATION_NVP(nFirst);
}

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

template <class Archive>
void PhysicsTools::Calibration::VarProcessor::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(inputVars);
}

template <class Archive>
void PhysicsTools::Calibration::Variable::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(name);
}

namespace cond {
namespace serialization {

template <>
struct access<BinningVariables>
{
    static bool equal_(const BinningVariables & first, const BinningVariables & second)
    {
        return true
        ;
    }
};

template <>
struct access<PerformancePayload>
{
    static bool equal_(const PerformancePayload & first, const PerformancePayload & second)
    {
        return true
        ;
    }
};

template <>
struct access<PerformancePayloadFromBinnedTFormula>
{
    static bool equal_(const PerformancePayloadFromBinnedTFormula & first, const PerformancePayloadFromBinnedTFormula & second)
    {
        return true
            and (equal(static_cast<const PerformancePayload &>(first), static_cast<const PerformancePayload &>(second)))
            and (equal(first.pls, second.pls))
            and (equal(first.results_, second.results_))
            and (equal(first.variables_, second.variables_))
        ;
    }
};

template <>
struct access<PerformancePayloadFromTFormula>
{
    static bool equal_(const PerformancePayloadFromTFormula & first, const PerformancePayloadFromTFormula & second)
    {
        return true
            and (equal(static_cast<const PerformancePayload &>(first), static_cast<const PerformancePayload &>(second)))
            and (equal(first.pl, second.pl))
            and (equal(first.results_, second.results_))
            and (equal(first.variables_, second.variables_))
        ;
    }
};

template <>
struct access<PerformancePayloadFromTable>
{
    static bool equal_(const PerformancePayloadFromTable & first, const PerformancePayloadFromTable & second)
    {
        return true
            and (equal(static_cast<const PerformancePayload &>(first), static_cast<const PerformancePayload &>(second)))
            and (equal(first.pl, second.pl))
            and (equal(first.results_, second.results_))
            and (equal(first.binning_, second.binning_))
        ;
    }
};

template <>
struct access<PerformanceResult>
{
    static bool equal_(const PerformanceResult & first, const PerformanceResult & second)
    {
        return true
        ;
    }
};

template <>
struct access<PerformanceWorkingPoint>
{
    static bool equal_(const PerformanceWorkingPoint & first, const PerformanceWorkingPoint & second)
    {
        return true
            and (equal(first.cut_, second.cut_))
            and (equal(first.dname_, second.dname_))
        ;
    }
};

template <>
struct access<PhysicsPerformancePayload>
{
    static bool equal_(const PhysicsPerformancePayload & first, const PhysicsPerformancePayload & second)
    {
        return true
            and (equal(first.stride_, second.stride_))
            and (equal(first.table_, second.table_))
        ;
    }
};

template <>
struct access<PhysicsTFormulaPayload>
{
    static bool equal_(const PhysicsTFormulaPayload & first, const PhysicsTFormulaPayload & second)
    {
        return true
            and (equal(first.limits_, second.limits_))
            and (equal(first.formulas_, second.formulas_))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::BitSet>
{
    static bool equal_(const PhysicsTools::Calibration::BitSet & first, const PhysicsTools::Calibration::BitSet & second)
    {
        return true
            and (equal(first.store, second.store))
            and (equal(first.bitsInLast, second.bitsInLast))
        ;
    }
};

template <typename Value_t, typename AxisX_t, typename AxisY_t>
struct access<PhysicsTools::Calibration::Histogram2D<Value_t, AxisX_t, AxisY_t>>
{
    static bool equal_(const PhysicsTools::Calibration::Histogram2D<Value_t, AxisX_t, AxisY_t> & first, const PhysicsTools::Calibration::Histogram2D<Value_t, AxisX_t, AxisY_t> & second)
    {
        return true
            and (equal(first.stride, second.stride))
            and (equal(first.binULimitsX, second.binULimitsX))
            and (equal(first.binULimitsY, second.binULimitsY))
            and (equal(first.binValues, second.binValues))
            and (equal(first.limitsX, second.limitsX))
            and (equal(first.limitsY, second.limitsY))
        ;
    }
};

template <typename Value_t, typename AxisX_t, typename AxisY_t, typename AxisZ_t>
struct access<PhysicsTools::Calibration::Histogram3D<Value_t, AxisX_t, AxisY_t, AxisZ_t>>
{
    static bool equal_(const PhysicsTools::Calibration::Histogram3D<Value_t, AxisX_t, AxisY_t, AxisZ_t> & first, const PhysicsTools::Calibration::Histogram3D<Value_t, AxisX_t, AxisY_t, AxisZ_t> & second)
    {
        return true
            and (equal(first.strideX, second.strideX))
            and (equal(first.strideY, second.strideY))
            and (equal(first.binULimitsX, second.binULimitsX))
            and (equal(first.binULimitsY, second.binULimitsY))
            and (equal(first.binULimitsZ, second.binULimitsZ))
            and (equal(first.binValues, second.binValues))
            and (equal(first.limitsX, second.limitsX))
            and (equal(first.limitsY, second.limitsY))
            and (equal(first.limitsZ, second.limitsZ))
        ;
    }
};

template <typename Value_t, typename Axis_t>
struct access<PhysicsTools::Calibration::Histogram<Value_t, Axis_t>>
{
    static bool equal_(const PhysicsTools::Calibration::Histogram<Value_t, Axis_t> & first, const PhysicsTools::Calibration::Histogram<Value_t, Axis_t> & second)
    {
        return true
            and (equal(first.binULimits, second.binULimits))
            and (equal(first.binValues, second.binValues))
            and (equal(first.limits, second.limits))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::MVAComputer>
{
    static bool equal_(const PhysicsTools::Calibration::MVAComputer & first, const PhysicsTools::Calibration::MVAComputer & second)
    {
        return true
            and (equal(first.inputSet, second.inputSet))
            and (equal(first.output, second.output))
            and (equal(first.processors, second.processors))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::MVAComputerContainer>
{
    static bool equal_(const PhysicsTools::Calibration::MVAComputerContainer & first, const PhysicsTools::Calibration::MVAComputerContainer & second)
    {
        return true
            and (equal(first.entries, second.entries))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::Matrix>
{
    static bool equal_(const PhysicsTools::Calibration::Matrix & first, const PhysicsTools::Calibration::Matrix & second)
    {
        return true
            and (equal(first.elements, second.elements))
            and (equal(first.rows, second.rows))
            and (equal(first.columns, second.columns))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcCategory>
{
    static bool equal_(const PhysicsTools::Calibration::ProcCategory & first, const PhysicsTools::Calibration::ProcCategory & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.variableBinLimits, second.variableBinLimits))
            and (equal(first.categoryMapping, second.categoryMapping))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcClassed>
{
    static bool equal_(const PhysicsTools::Calibration::ProcClassed & first, const PhysicsTools::Calibration::ProcClassed & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.nClasses, second.nClasses))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcCount>
{
    static bool equal_(const PhysicsTools::Calibration::ProcCount & first, const PhysicsTools::Calibration::ProcCount & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcExternal>
{
    static bool equal_(const PhysicsTools::Calibration::ProcExternal & first, const PhysicsTools::Calibration::ProcExternal & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.method, second.method))
            and (equal(first.store, second.store))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcForeach>
{
    static bool equal_(const PhysicsTools::Calibration::ProcForeach & first, const PhysicsTools::Calibration::ProcForeach & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.nProcs, second.nProcs))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcLikelihood>
{
    static bool equal_(const PhysicsTools::Calibration::ProcLikelihood & first, const PhysicsTools::Calibration::ProcLikelihood & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.pdfs, second.pdfs))
            and (equal(first.bias, second.bias))
            and (equal(first.categoryIdx, second.categoryIdx))
            and (equal(first.logOutput, second.logOutput))
            and (equal(first.individual, second.individual))
            and (equal(first.neverUndefined, second.neverUndefined))
            and (equal(first.keepEmpty, second.keepEmpty))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcLikelihood::SigBkg>
{
    static bool equal_(const PhysicsTools::Calibration::ProcLikelihood::SigBkg & first, const PhysicsTools::Calibration::ProcLikelihood::SigBkg & second)
    {
        return true
            and (equal(first.background, second.background))
            and (equal(first.signal, second.signal))
            and (equal(first.useSplines, second.useSplines))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcLinear>
{
    static bool equal_(const PhysicsTools::Calibration::ProcLinear & first, const PhysicsTools::Calibration::ProcLinear & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.coeffs, second.coeffs))
            and (equal(first.offset, second.offset))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcMLP>
{
    static bool equal_(const PhysicsTools::Calibration::ProcMLP & first, const PhysicsTools::Calibration::ProcMLP & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.layers, second.layers))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcMatrix>
{
    static bool equal_(const PhysicsTools::Calibration::ProcMatrix & first, const PhysicsTools::Calibration::ProcMatrix & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.matrix, second.matrix))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcMultiply>
{
    static bool equal_(const PhysicsTools::Calibration::ProcMultiply & first, const PhysicsTools::Calibration::ProcMultiply & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.in, second.in))
            and (equal(first.out, second.out))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcNormalize>
{
    static bool equal_(const PhysicsTools::Calibration::ProcNormalize & first, const PhysicsTools::Calibration::ProcNormalize & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.distr, second.distr))
            and (equal(first.categoryIdx, second.categoryIdx))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcOptional>
{
    static bool equal_(const PhysicsTools::Calibration::ProcOptional & first, const PhysicsTools::Calibration::ProcOptional & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.neutralPos, second.neutralPos))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcSort>
{
    static bool equal_(const PhysicsTools::Calibration::ProcSort & first, const PhysicsTools::Calibration::ProcSort & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.sortByIndex, second.sortByIndex))
            and (equal(first.descending, second.descending))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::ProcSplitter>
{
    static bool equal_(const PhysicsTools::Calibration::ProcSplitter & first, const PhysicsTools::Calibration::ProcSplitter & second)
    {
        return true
            and (equal(static_cast<const PhysicsTools::Calibration::VarProcessor &>(first), static_cast<const PhysicsTools::Calibration::VarProcessor &>(second)))
            and (equal(first.nFirst, second.nFirst))
        ;
    }
};

template <typename Axis_t>
struct access<PhysicsTools::Calibration::Range<Axis_t>>
{
    static bool equal_(const PhysicsTools::Calibration::Range<Axis_t> & first, const PhysicsTools::Calibration::Range<Axis_t> & second)
    {
        return true
            and (equal(first.min, second.min))
            and (equal(first.max, second.max))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::VHistogramD2D>
{
    static bool equal_(const PhysicsTools::Calibration::VHistogramD2D & first, const PhysicsTools::Calibration::VHistogramD2D & second)
    {
        return true
            and (equal(first.vHist, second.vHist))
            and (equal(first.vValues, second.vValues))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::VarProcessor>
{
    static bool equal_(const PhysicsTools::Calibration::VarProcessor & first, const PhysicsTools::Calibration::VarProcessor & second)
    {
        return true
            and (equal(first.inputVars, second.inputVars))
        ;
    }
};

template <>
struct access<PhysicsTools::Calibration::Variable>
{
    static bool equal_(const PhysicsTools::Calibration::Variable & first, const PhysicsTools::Calibration::Variable & second)
    {
        return true
            and (equal(first.name, second.name))
        ;
    }
};

}
}

#endif
