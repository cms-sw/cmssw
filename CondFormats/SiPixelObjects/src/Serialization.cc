
#include "CondFormats/SiPixelObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void CaenChannel::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(isOn);
    ar & BOOST_SERIALIZATION_NVP(iMon);
    ar & BOOST_SERIALIZATION_NVP(vMon);
}
COND_SERIALIZATION_INSTANTIATE(CaenChannel);

template <typename T>
template <class Archive>
void PixelDCSObject<T>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(items);
}

template <typename T>
template <class Archive>
void PixelDCSObject<T>::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(name);
    ar & BOOST_SERIALIZATION_NVP(value);
}

template <class Archive>
void SiPixelCPEGenericErrorParm::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(errors_);
    ar & BOOST_SERIALIZATION_NVP(errorsBinSize_);
    ar & BOOST_SERIALIZATION_NVP(version_);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelCPEGenericErrorParm);

template <class Archive>
void SiPixelCPEGenericErrorParm::DbEntry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(sigma);
    ar & BOOST_SERIALIZATION_NVP(rms);
    ar & BOOST_SERIALIZATION_NVP(bias);
    ar & BOOST_SERIALIZATION_NVP(pix_height);
    ar & BOOST_SERIALIZATION_NVP(ave_Qclus);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelCPEGenericErrorParm::DbEntry);

template <class Archive>
void SiPixelCPEGenericErrorParm::DbEntryBinSize::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(partBin_size);
    ar & BOOST_SERIALIZATION_NVP(sizeBin_size);
    ar & BOOST_SERIALIZATION_NVP(alphaBin_size);
    ar & BOOST_SERIALIZATION_NVP(betaBin_size);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelCPEGenericErrorParm::DbEntryBinSize);

template <class Archive>
void SiPixelCalibConfiguration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fNTriggers);
    ar & BOOST_SERIALIZATION_NVP(fRowPattern);
    ar & BOOST_SERIALIZATION_NVP(fColumnPattern);
    ar & BOOST_SERIALIZATION_NVP(fVCalValues);
    ar & BOOST_SERIALIZATION_NVP(fMode);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelCalibConfiguration);

template <class Archive>
void SiPixelDbItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(packedVal_);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelDbItem);

template <class Archive>
void SiPixelDisabledModules::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theDisabledModules);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelDisabledModules);

template <class Archive>
void SiPixelFedCabling::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(SiPixelFedCabling);

template <class Archive>
void SiPixelFedCablingMap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("SiPixelFedCabling", boost::serialization::base_object<SiPixelFedCabling>(*this));
    ar & BOOST_SERIALIZATION_NVP(theVersion);
    ar & BOOST_SERIALIZATION_NVP(theMap);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelFedCablingMap);

template <class Archive>
void SiPixelFedCablingMap::Key::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fed);
    ar & BOOST_SERIALIZATION_NVP(link);
    ar & BOOST_SERIALIZATION_NVP(roc);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelFedCablingMap::Key);

template <class Archive>
void SiPixelGainCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_pedestals);
    ar & BOOST_SERIALIZATION_NVP(indexes);
    ar & BOOST_SERIALIZATION_NVP(minPed_);
    ar & BOOST_SERIALIZATION_NVP(maxPed_);
    ar & BOOST_SERIALIZATION_NVP(minGain_);
    ar & BOOST_SERIALIZATION_NVP(maxGain_);
    ar & BOOST_SERIALIZATION_NVP(numberOfRowsToAverageOver_);
    ar & BOOST_SERIALIZATION_NVP(nBinsToUseForEncoding_);
    ar & BOOST_SERIALIZATION_NVP(deadFlag_);
    ar & BOOST_SERIALIZATION_NVP(noisyFlag_);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelGainCalibration);

template <class Archive>
void SiPixelGainCalibration::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
    ar & BOOST_SERIALIZATION_NVP(ncols);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelGainCalibration::DetRegistry);

template <class Archive>
void SiPixelGainCalibrationForHLT::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_pedestals);
    ar & BOOST_SERIALIZATION_NVP(indexes);
    ar & BOOST_SERIALIZATION_NVP(minPed_);
    ar & BOOST_SERIALIZATION_NVP(maxPed_);
    ar & BOOST_SERIALIZATION_NVP(minGain_);
    ar & BOOST_SERIALIZATION_NVP(maxGain_);
    ar & BOOST_SERIALIZATION_NVP(pedPrecision);
    ar & BOOST_SERIALIZATION_NVP(gainPrecision);
    ar & BOOST_SERIALIZATION_NVP(numberOfRowsToAverageOver_);
    ar & BOOST_SERIALIZATION_NVP(nBinsToUseForEncoding_);
    ar & BOOST_SERIALIZATION_NVP(deadFlag_);
    ar & BOOST_SERIALIZATION_NVP(noisyFlag_);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelGainCalibrationForHLT);

template <class Archive>
void SiPixelGainCalibrationForHLT::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
    ar & BOOST_SERIALIZATION_NVP(ncols);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelGainCalibrationForHLT::DetRegistry);

template <class Archive>
void SiPixelGainCalibrationOffline::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_pedestals);
    ar & BOOST_SERIALIZATION_NVP(indexes);
    ar & BOOST_SERIALIZATION_NVP(minPed_);
    ar & BOOST_SERIALIZATION_NVP(maxPed_);
    ar & BOOST_SERIALIZATION_NVP(minGain_);
    ar & BOOST_SERIALIZATION_NVP(maxGain_);
    ar & BOOST_SERIALIZATION_NVP(numberOfRowsToAverageOver_);
    ar & BOOST_SERIALIZATION_NVP(nBinsToUseForEncoding_);
    ar & BOOST_SERIALIZATION_NVP(deadFlag_);
    ar & BOOST_SERIALIZATION_NVP(noisyFlag_);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelGainCalibrationOffline);

template <class Archive>
void SiPixelGainCalibrationOffline::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
    ar & BOOST_SERIALIZATION_NVP(ncols);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelGainCalibrationOffline::DetRegistry);

template <class Archive>
void SiPixelLorentzAngle::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_LA);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelLorentzAngle);

template <class Archive>
void SiPixelPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_pedestals);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelPedestals);

template <class Archive>
void SiPixelPerformanceSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(timeStamp_);
    ar & BOOST_SERIALIZATION_NVP(runNumber_);
    ar & BOOST_SERIALIZATION_NVP(luminosityBlock_);
    ar & BOOST_SERIALIZATION_NVP(numberOfEvents_);
    ar & BOOST_SERIALIZATION_NVP(allDetSummaries_);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelPerformanceSummary);

template <class Archive>
void SiPixelPerformanceSummary::DetSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detId_);
    ar & BOOST_SERIALIZATION_NVP(performanceValues_);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelPerformanceSummary::DetSummary);

template <class Archive>
void SiPixelQuality::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theDisabledModules);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelQuality);

template <class Archive>
void SiPixelQuality::disabledModuleType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(DetID);
    ar & BOOST_SERIALIZATION_NVP(errorType);
    ar & BOOST_SERIALIZATION_NVP(BadRocs);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelQuality::disabledModuleType);

template <class Archive>
void SiPixelTemplateDBObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(index_);
    ar & BOOST_SERIALIZATION_NVP(maxIndex_);
    ar & BOOST_SERIALIZATION_NVP(numOfTempl_);
    ar & BOOST_SERIALIZATION_NVP(version_);
    ar & BOOST_SERIALIZATION_NVP(isInvalid_);
    ar & BOOST_SERIALIZATION_NVP(sVector_);
    ar & BOOST_SERIALIZATION_NVP(templ_ID);
}
COND_SERIALIZATION_INSTANTIATE(SiPixelTemplateDBObject);

template <class Archive>
void sipixelobjects::PixelROC::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theDetUnit);
    ar & BOOST_SERIALIZATION_NVP(theIdDU);
    ar & BOOST_SERIALIZATION_NVP(theIdLk);
}
COND_SERIALIZATION_INSTANTIATE(sipixelobjects::PixelROC);

#include "CondFormats/SiPixelObjects/src/SerializationManual.h"
