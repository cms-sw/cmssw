
#include "CondFormats/OptAlignObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

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
COND_SERIALIZATION_INSTANTIATE(CSCRSensorData);

template <class Archive>
void CSCRSensors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cscRSens_);
}
COND_SERIALIZATION_INSTANTIATE(CSCRSensors);

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
COND_SERIALIZATION_INSTANTIATE(CSCZSensorData);

template <class Archive>
void CSCZSensors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cscZSens_);
}
COND_SERIALIZATION_INSTANTIATE(CSCZSensors);

template <class Archive>
void Inclinometers::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_inclinometers);
}
COND_SERIALIZATION_INSTANTIATE(Inclinometers);

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
COND_SERIALIZATION_INSTANTIATE(Inclinometers::Item);

template <class Archive>
void MBAChBenchCalPlate::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mbaChBenchCalPlate_);
}
COND_SERIALIZATION_INSTANTIATE(MBAChBenchCalPlate);

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
COND_SERIALIZATION_INSTANTIATE(MBAChBenchCalPlateData);

template <class Archive>
void MBAChBenchSurveyPlate::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mbaChBenchSurveyPlate_);
}
COND_SERIALIZATION_INSTANTIATE(MBAChBenchSurveyPlate);

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
COND_SERIALIZATION_INSTANTIATE(MBAChBenchSurveyPlateData);

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
COND_SERIALIZATION_INSTANTIATE(OpticalAlignInfo);

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
COND_SERIALIZATION_INSTANTIATE(OpticalAlignMeasurementInfo);

template <class Archive>
void OpticalAlignMeasurements::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(oaMeasurements_);
}
COND_SERIALIZATION_INSTANTIATE(OpticalAlignMeasurements);

template <class Archive>
void OpticalAlignParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(value_);
    ar & BOOST_SERIALIZATION_NVP(error_);
    ar & BOOST_SERIALIZATION_NVP(quality_);
    ar & BOOST_SERIALIZATION_NVP(name_);
    ar & BOOST_SERIALIZATION_NVP(dim_type_);
}
COND_SERIALIZATION_INSTANTIATE(OpticalAlignParam);

template <class Archive>
void OpticalAlignments::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(opticalAlignments_);
}
COND_SERIALIZATION_INSTANTIATE(OpticalAlignments);

template <class Archive>
void PXsensors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_PXsensors);
}
COND_SERIALIZATION_INSTANTIATE(PXsensors);

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
COND_SERIALIZATION_INSTANTIATE(PXsensors::Item);

#include "CondFormats/OptAlignObjects/src/SerializationManual.h"
