#ifndef CondFormats_OptAlignObjects_Serialization_H
#define CondFormats_OptAlignObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void CSCRSensorData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(sensorType_);
    ar & BOOST_SERIALIZATION_NVP(sensorNo_);
    ar & BOOST_SERIALIZATION_NVP(meLayer_);
    ar & BOOST_SERIALIZATION_NVP(logicalAlignmentName_);
    ar & BOOST_SERIALIZATION_NVP(cernDesignator_);
    ar & BOOST_SERIALIZATION_NVP(cernBarcode_);
    ar & BOOST_SERIALIZATION_NVP(absSlope_);
    ar & BOOST_SERIALIZATION_NVP(absSlopeError_);
    ar & BOOST_SERIALIZATION_NVP(normSlope_);
    ar & BOOST_SERIALIZATION_NVP(normSlopeError_);
    ar & BOOST_SERIALIZATION_NVP(absIntercept_);
    ar & BOOST_SERIALIZATION_NVP(absInterceptError_);
    ar & BOOST_SERIALIZATION_NVP(normIntercept_);
    ar & BOOST_SERIALIZATION_NVP(normInterceptError_);
    ar & BOOST_SERIALIZATION_NVP(shifts_);
}

template <class Archive>
void CSCRSensors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cscRSens_);
}

template <class Archive>
void CSCZSensorData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(sensorType_);
    ar & BOOST_SERIALIZATION_NVP(sensorNo_);
    ar & BOOST_SERIALIZATION_NVP(meLayer_);
    ar & BOOST_SERIALIZATION_NVP(logicalAlignmentName_);
    ar & BOOST_SERIALIZATION_NVP(cernDesignator_);
    ar & BOOST_SERIALIZATION_NVP(cernBarcode_);
    ar & BOOST_SERIALIZATION_NVP(absSlope_);
    ar & BOOST_SERIALIZATION_NVP(absSlopeError_);
    ar & BOOST_SERIALIZATION_NVP(normSlope_);
    ar & BOOST_SERIALIZATION_NVP(normSlopeError_);
    ar & BOOST_SERIALIZATION_NVP(absIntercept_);
    ar & BOOST_SERIALIZATION_NVP(absInterceptError_);
    ar & BOOST_SERIALIZATION_NVP(normIntercept_);
    ar & BOOST_SERIALIZATION_NVP(normInterceptError_);
    ar & BOOST_SERIALIZATION_NVP(shifts_);
}

template <class Archive>
void CSCZSensors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cscZSens_);
}

template <class Archive>
void Inclinometers::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_inclinometers);
}

template <class Archive>
void Inclinometers::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(Sensor_type);
    ar & BOOST_SERIALIZATION_NVP(Sensor_number);
    ar & BOOST_SERIALIZATION_NVP(ME_layer);
    ar & BOOST_SERIALIZATION_NVP(Logical_Alignment_Name);
    ar & BOOST_SERIALIZATION_NVP(CERN_Designator);
    ar & BOOST_SERIALIZATION_NVP(CERN_Barcode);
    ar & BOOST_SERIALIZATION_NVP(Inclination_Direction);
    ar & BOOST_SERIALIZATION_NVP(Abs_Slope);
    ar & BOOST_SERIALIZATION_NVP(Abs_Slope_Error);
    ar & BOOST_SERIALIZATION_NVP(Norm_Slope);
    ar & BOOST_SERIALIZATION_NVP(Norm_Slope_Error);
    ar & BOOST_SERIALIZATION_NVP(Abs_Intercept);
    ar & BOOST_SERIALIZATION_NVP(Abs_Intercept_Error);
    ar & BOOST_SERIALIZATION_NVP(Norm_Intercept);
    ar & BOOST_SERIALIZATION_NVP(Norm_Intercept_Error);
    ar & BOOST_SERIALIZATION_NVP(Shifts_due_to_shims_etc);
}

template <class Archive>
void MBAChBenchCalPlate::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mbaChBenchCalPlate_);
}

template <class Archive>
void MBAChBenchCalPlateData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(plate_);
    ar & BOOST_SERIALIZATION_NVP(side_);
    ar & BOOST_SERIALIZATION_NVP(object_);
    ar & BOOST_SERIALIZATION_NVP(posX_);
    ar & BOOST_SERIALIZATION_NVP(posY_);
    ar & BOOST_SERIALIZATION_NVP(posZ_);
    ar & BOOST_SERIALIZATION_NVP(measDateTime_);
}

template <class Archive>
void MBAChBenchSurveyPlate::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mbaChBenchSurveyPlate_);
}

template <class Archive>
void MBAChBenchSurveyPlateData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(edmsID_);
    ar & BOOST_SERIALIZATION_NVP(surveyCode_);
    ar & BOOST_SERIALIZATION_NVP(line_);
    ar & BOOST_SERIALIZATION_NVP(plate_);
    ar & BOOST_SERIALIZATION_NVP(side_);
    ar & BOOST_SERIALIZATION_NVP(object_);
    ar & BOOST_SERIALIZATION_NVP(posX_);
    ar & BOOST_SERIALIZATION_NVP(posY_);
    ar & BOOST_SERIALIZATION_NVP(posZ_);
    ar & BOOST_SERIALIZATION_NVP(measDateTime_);
}

template <class Archive>
void OpticalAlignInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(x_);
    ar & BOOST_SERIALIZATION_NVP(y_);
    ar & BOOST_SERIALIZATION_NVP(z_);
    ar & BOOST_SERIALIZATION_NVP(angx_);
    ar & BOOST_SERIALIZATION_NVP(angy_);
    ar & BOOST_SERIALIZATION_NVP(angz_);
    ar & BOOST_SERIALIZATION_NVP(extraEntries_);
    ar & BOOST_SERIALIZATION_NVP(type_);
    ar & BOOST_SERIALIZATION_NVP(name_);
    ar & BOOST_SERIALIZATION_NVP(parentName_);
    ar & BOOST_SERIALIZATION_NVP(ID_);
}

template <class Archive>
void OpticalAlignMeasurementInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(type_);
    ar & BOOST_SERIALIZATION_NVP(name_);
    ar & BOOST_SERIALIZATION_NVP(measObjectNames_);
    ar & BOOST_SERIALIZATION_NVP(isSimulatedValue_);
    ar & BOOST_SERIALIZATION_NVP(values_);
    ar & BOOST_SERIALIZATION_NVP(ID_);
}

template <class Archive>
void OpticalAlignMeasurements::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(oaMeasurements_);
}

template <class Archive>
void OpticalAlignParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(value_);
    ar & BOOST_SERIALIZATION_NVP(error_);
    ar & BOOST_SERIALIZATION_NVP(quality_);
    ar & BOOST_SERIALIZATION_NVP(name_);
    ar & BOOST_SERIALIZATION_NVP(dim_type_);
}

template <class Archive>
void OpticalAlignments::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(opticalAlignments_);
}

template <class Archive>
void PXsensors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_PXsensors);
}

template <class Archive>
void PXsensors::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(Sensor_type);
    ar & BOOST_SERIALIZATION_NVP(Sensor_number);
    ar & BOOST_SERIALIZATION_NVP(ME_layer);
    ar & BOOST_SERIALIZATION_NVP(Logical_Alignment_Name);
    ar & BOOST_SERIALIZATION_NVP(CERN_Designator);
    ar & BOOST_SERIALIZATION_NVP(CERN_Barcode);
    ar & BOOST_SERIALIZATION_NVP(Abs_Slope);
    ar & BOOST_SERIALIZATION_NVP(Abs_Slope_Error);
    ar & BOOST_SERIALIZATION_NVP(Norm_Slope);
    ar & BOOST_SERIALIZATION_NVP(Norm_Slope_Error);
    ar & BOOST_SERIALIZATION_NVP(Abs_Intercept);
    ar & BOOST_SERIALIZATION_NVP(Abs_Intercept_Error);
    ar & BOOST_SERIALIZATION_NVP(Norm_Intercept);
    ar & BOOST_SERIALIZATION_NVP(Norm_Intercept_Error);
    ar & BOOST_SERIALIZATION_NVP(Shifts_due_to_shims_etc);
}

namespace cond {
namespace serialization {

template <>
struct access<CSCRSensorData>
{
    static bool equal_(const CSCRSensorData & first, const CSCRSensorData & second)
    {
        return true
            and (equal(first.sensorType_, second.sensorType_))
            and (equal(first.sensorNo_, second.sensorNo_))
            and (equal(first.meLayer_, second.meLayer_))
            and (equal(first.logicalAlignmentName_, second.logicalAlignmentName_))
            and (equal(first.cernDesignator_, second.cernDesignator_))
            and (equal(first.cernBarcode_, second.cernBarcode_))
            and (equal(first.absSlope_, second.absSlope_))
            and (equal(first.absSlopeError_, second.absSlopeError_))
            and (equal(first.normSlope_, second.normSlope_))
            and (equal(first.normSlopeError_, second.normSlopeError_))
            and (equal(first.absIntercept_, second.absIntercept_))
            and (equal(first.absInterceptError_, second.absInterceptError_))
            and (equal(first.normIntercept_, second.normIntercept_))
            and (equal(first.normInterceptError_, second.normInterceptError_))
            and (equal(first.shifts_, second.shifts_))
        ;
    }
};

template <>
struct access<CSCRSensors>
{
    static bool equal_(const CSCRSensors & first, const CSCRSensors & second)
    {
        return true
            and (equal(first.cscRSens_, second.cscRSens_))
        ;
    }
};

template <>
struct access<CSCZSensorData>
{
    static bool equal_(const CSCZSensorData & first, const CSCZSensorData & second)
    {
        return true
            and (equal(first.sensorType_, second.sensorType_))
            and (equal(first.sensorNo_, second.sensorNo_))
            and (equal(first.meLayer_, second.meLayer_))
            and (equal(first.logicalAlignmentName_, second.logicalAlignmentName_))
            and (equal(first.cernDesignator_, second.cernDesignator_))
            and (equal(first.cernBarcode_, second.cernBarcode_))
            and (equal(first.absSlope_, second.absSlope_))
            and (equal(first.absSlopeError_, second.absSlopeError_))
            and (equal(first.normSlope_, second.normSlope_))
            and (equal(first.normSlopeError_, second.normSlopeError_))
            and (equal(first.absIntercept_, second.absIntercept_))
            and (equal(first.absInterceptError_, second.absInterceptError_))
            and (equal(first.normIntercept_, second.normIntercept_))
            and (equal(first.normInterceptError_, second.normInterceptError_))
            and (equal(first.shifts_, second.shifts_))
        ;
    }
};

template <>
struct access<CSCZSensors>
{
    static bool equal_(const CSCZSensors & first, const CSCZSensors & second)
    {
        return true
            and (equal(first.cscZSens_, second.cscZSens_))
        ;
    }
};

template <>
struct access<Inclinometers>
{
    static bool equal_(const Inclinometers & first, const Inclinometers & second)
    {
        return true
            and (equal(first.m_inclinometers, second.m_inclinometers))
        ;
    }
};

template <>
struct access<Inclinometers::Item>
{
    static bool equal_(const Inclinometers::Item & first, const Inclinometers::Item & second)
    {
        return true
            and (equal(first.Sensor_type, second.Sensor_type))
            and (equal(first.Sensor_number, second.Sensor_number))
            and (equal(first.ME_layer, second.ME_layer))
            and (equal(first.Logical_Alignment_Name, second.Logical_Alignment_Name))
            and (equal(first.CERN_Designator, second.CERN_Designator))
            and (equal(first.CERN_Barcode, second.CERN_Barcode))
            and (equal(first.Inclination_Direction, second.Inclination_Direction))
            and (equal(first.Abs_Slope, second.Abs_Slope))
            and (equal(first.Abs_Slope_Error, second.Abs_Slope_Error))
            and (equal(first.Norm_Slope, second.Norm_Slope))
            and (equal(first.Norm_Slope_Error, second.Norm_Slope_Error))
            and (equal(first.Abs_Intercept, second.Abs_Intercept))
            and (equal(first.Abs_Intercept_Error, second.Abs_Intercept_Error))
            and (equal(first.Norm_Intercept, second.Norm_Intercept))
            and (equal(first.Norm_Intercept_Error, second.Norm_Intercept_Error))
            and (equal(first.Shifts_due_to_shims_etc, second.Shifts_due_to_shims_etc))
        ;
    }
};

template <>
struct access<MBAChBenchCalPlate>
{
    static bool equal_(const MBAChBenchCalPlate & first, const MBAChBenchCalPlate & second)
    {
        return true
            and (equal(first.mbaChBenchCalPlate_, second.mbaChBenchCalPlate_))
        ;
    }
};

template <>
struct access<MBAChBenchCalPlateData>
{
    static bool equal_(const MBAChBenchCalPlateData & first, const MBAChBenchCalPlateData & second)
    {
        return true
            and (equal(first.plate_, second.plate_))
            and (equal(first.side_, second.side_))
            and (equal(first.object_, second.object_))
            and (equal(first.posX_, second.posX_))
            and (equal(first.posY_, second.posY_))
            and (equal(first.posZ_, second.posZ_))
            and (equal(first.measDateTime_, second.measDateTime_))
        ;
    }
};

template <>
struct access<MBAChBenchSurveyPlate>
{
    static bool equal_(const MBAChBenchSurveyPlate & first, const MBAChBenchSurveyPlate & second)
    {
        return true
            and (equal(first.mbaChBenchSurveyPlate_, second.mbaChBenchSurveyPlate_))
        ;
    }
};

template <>
struct access<MBAChBenchSurveyPlateData>
{
    static bool equal_(const MBAChBenchSurveyPlateData & first, const MBAChBenchSurveyPlateData & second)
    {
        return true
            and (equal(first.edmsID_, second.edmsID_))
            and (equal(first.surveyCode_, second.surveyCode_))
            and (equal(first.line_, second.line_))
            and (equal(first.plate_, second.plate_))
            and (equal(first.side_, second.side_))
            and (equal(first.object_, second.object_))
            and (equal(first.posX_, second.posX_))
            and (equal(first.posY_, second.posY_))
            and (equal(first.posZ_, second.posZ_))
            and (equal(first.measDateTime_, second.measDateTime_))
        ;
    }
};

template <>
struct access<OpticalAlignInfo>
{
    static bool equal_(const OpticalAlignInfo & first, const OpticalAlignInfo & second)
    {
        return true
            and (equal(first.x_, second.x_))
            and (equal(first.y_, second.y_))
            and (equal(first.z_, second.z_))
            and (equal(first.angx_, second.angx_))
            and (equal(first.angy_, second.angy_))
            and (equal(first.angz_, second.angz_))
            and (equal(first.extraEntries_, second.extraEntries_))
            and (equal(first.type_, second.type_))
            and (equal(first.name_, second.name_))
            and (equal(first.parentName_, second.parentName_))
            and (equal(first.ID_, second.ID_))
        ;
    }
};

template <>
struct access<OpticalAlignMeasurementInfo>
{
    static bool equal_(const OpticalAlignMeasurementInfo & first, const OpticalAlignMeasurementInfo & second)
    {
        return true
            and (equal(first.type_, second.type_))
            and (equal(first.name_, second.name_))
            and (equal(first.measObjectNames_, second.measObjectNames_))
            and (equal(first.isSimulatedValue_, second.isSimulatedValue_))
            and (equal(first.values_, second.values_))
            and (equal(first.ID_, second.ID_))
        ;
    }
};

template <>
struct access<OpticalAlignMeasurements>
{
    static bool equal_(const OpticalAlignMeasurements & first, const OpticalAlignMeasurements & second)
    {
        return true
            and (equal(first.oaMeasurements_, second.oaMeasurements_))
        ;
    }
};

template <>
struct access<OpticalAlignParam>
{
    static bool equal_(const OpticalAlignParam & first, const OpticalAlignParam & second)
    {
        return true
            and (equal(first.value_, second.value_))
            and (equal(first.error_, second.error_))
            and (equal(first.quality_, second.quality_))
            and (equal(first.name_, second.name_))
            and (equal(first.dim_type_, second.dim_type_))
        ;
    }
};

template <>
struct access<OpticalAlignments>
{
    static bool equal_(const OpticalAlignments & first, const OpticalAlignments & second)
    {
        return true
            and (equal(first.opticalAlignments_, second.opticalAlignments_))
        ;
    }
};

template <>
struct access<PXsensors>
{
    static bool equal_(const PXsensors & first, const PXsensors & second)
    {
        return true
            and (equal(first.m_PXsensors, second.m_PXsensors))
        ;
    }
};

template <>
struct access<PXsensors::Item>
{
    static bool equal_(const PXsensors::Item & first, const PXsensors::Item & second)
    {
        return true
            and (equal(first.Sensor_type, second.Sensor_type))
            and (equal(first.Sensor_number, second.Sensor_number))
            and (equal(first.ME_layer, second.ME_layer))
            and (equal(first.Logical_Alignment_Name, second.Logical_Alignment_Name))
            and (equal(first.CERN_Designator, second.CERN_Designator))
            and (equal(first.CERN_Barcode, second.CERN_Barcode))
            and (equal(first.Abs_Slope, second.Abs_Slope))
            and (equal(first.Abs_Slope_Error, second.Abs_Slope_Error))
            and (equal(first.Norm_Slope, second.Norm_Slope))
            and (equal(first.Norm_Slope_Error, second.Norm_Slope_Error))
            and (equal(first.Abs_Intercept, second.Abs_Intercept))
            and (equal(first.Abs_Intercept_Error, second.Abs_Intercept_Error))
            and (equal(first.Norm_Intercept, second.Norm_Intercept))
            and (equal(first.Norm_Intercept_Error, second.Norm_Intercept_Error))
            and (equal(first.Shifts_due_to_shims_etc, second.Shifts_due_to_shims_etc))
        ;
    }
};

}
}

#endif
