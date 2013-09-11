#ifndef CondFormats_SiPixelObjects_Serialization_H
#define CondFormats_SiPixelObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

#include "CondFormats/SiPixelObjects/interface/SerializationManual.h"

template <class Archive>
void CaenChannel::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(isOn);
    ar & BOOST_SERIALIZATION_NVP(iMon);
    ar & BOOST_SERIALIZATION_NVP(vMon);
}

template <typename T>
template <class Archive>
void PixelDCSObject<T>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(items);
}

template <class Archive>
void SiPixelCPEGenericErrorParm::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(errors_);
    ar & BOOST_SERIALIZATION_NVP(errorsBinSize_);
    ar & BOOST_SERIALIZATION_NVP(version_);
}

template <class Archive>
void SiPixelCPEGenericErrorParm::DbEntry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(sigma);
    ar & BOOST_SERIALIZATION_NVP(rms);
    ar & BOOST_SERIALIZATION_NVP(bias);
    ar & BOOST_SERIALIZATION_NVP(pix_height);
    ar & BOOST_SERIALIZATION_NVP(ave_Qclus);
}

template <class Archive>
void SiPixelCPEGenericErrorParm::DbEntryBinSize::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(partBin_size);
    ar & BOOST_SERIALIZATION_NVP(sizeBin_size);
    ar & BOOST_SERIALIZATION_NVP(alphaBin_size);
    ar & BOOST_SERIALIZATION_NVP(betaBin_size);
}

template <class Archive>
void SiPixelCalibConfiguration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fNTriggers);
    ar & BOOST_SERIALIZATION_NVP(fRowPattern);
    ar & BOOST_SERIALIZATION_NVP(fColumnPattern);
    ar & BOOST_SERIALIZATION_NVP(fVCalValues);
    ar & BOOST_SERIALIZATION_NVP(fMode);
}

template <class Archive>
void SiPixelDbItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(packedVal_);
}

template <class Archive>
void SiPixelDisabledModules::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theDisabledModules);
}

template <class Archive>
void SiPixelFedCabling::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void SiPixelFedCablingMap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("SiPixelFedCabling", boost::serialization::base_object<SiPixelFedCabling>(*this));
    ar & BOOST_SERIALIZATION_NVP(theVersion);
    ar & BOOST_SERIALIZATION_NVP(theMap);
}

template <class Archive>
void SiPixelFedCablingMap::Key::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fed);
    ar & BOOST_SERIALIZATION_NVP(link);
    ar & BOOST_SERIALIZATION_NVP(roc);
}

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

template <class Archive>
void SiPixelGainCalibration::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
    ar & BOOST_SERIALIZATION_NVP(ncols);
}

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

template <class Archive>
void SiPixelGainCalibrationForHLT::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
    ar & BOOST_SERIALIZATION_NVP(ncols);
}

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

template <class Archive>
void SiPixelGainCalibrationOffline::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
    ar & BOOST_SERIALIZATION_NVP(ncols);
}

template <class Archive>
void SiPixelLorentzAngle::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_LA);
}

template <class Archive>
void SiPixelPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_pedestals);
}

template <class Archive>
void SiPixelPerformanceSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(timeStamp_);
    ar & BOOST_SERIALIZATION_NVP(runNumber_);
    ar & BOOST_SERIALIZATION_NVP(luminosityBlock_);
    ar & BOOST_SERIALIZATION_NVP(numberOfEvents_);
    ar & BOOST_SERIALIZATION_NVP(allDetSummaries_);
}

template <class Archive>
void SiPixelPerformanceSummary::DetSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detId_);
    ar & BOOST_SERIALIZATION_NVP(performanceValues_);
}

template <class Archive>
void SiPixelQuality::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theDisabledModules);
}

template <class Archive>
void SiPixelQuality::disabledModuleType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(DetID);
    ar & BOOST_SERIALIZATION_NVP(errorType);
    ar & BOOST_SERIALIZATION_NVP(BadRocs);
}

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

template <class Archive>
void sipixelobjects::PixelROC::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theDetUnit);
    ar & BOOST_SERIALIZATION_NVP(theIdDU);
    ar & BOOST_SERIALIZATION_NVP(theIdLk);
}

namespace cond {
namespace serialization {

template <>
struct access<CaenChannel>
{
    static bool equal_(const CaenChannel & first, const CaenChannel & second)
    {
        return true
            and (equal(first.isOn, second.isOn))
            and (equal(first.iMon, second.iMon))
            and (equal(first.vMon, second.vMon))
        ;
    }
};

template <typename T>
struct access<PixelDCSObject<T>>
{
    static bool equal_(const PixelDCSObject<T> & first, const PixelDCSObject<T> & second)
    {
        return true
            and (equal(first.items, second.items))
        ;
    }
};

template <>
struct access<SiPixelCPEGenericErrorParm>
{
    static bool equal_(const SiPixelCPEGenericErrorParm & first, const SiPixelCPEGenericErrorParm & second)
    {
        return true
            and (equal(first.errors_, second.errors_))
            and (equal(first.errorsBinSize_, second.errorsBinSize_))
            and (equal(first.version_, second.version_))
        ;
    }
};

template <>
struct access<SiPixelCPEGenericErrorParm::DbEntry>
{
    static bool equal_(const SiPixelCPEGenericErrorParm::DbEntry & first, const SiPixelCPEGenericErrorParm::DbEntry & second)
    {
        return true
            and (equal(first.sigma, second.sigma))
            and (equal(first.rms, second.rms))
            and (equal(first.bias, second.bias))
            and (equal(first.pix_height, second.pix_height))
            and (equal(first.ave_Qclus, second.ave_Qclus))
        ;
    }
};

template <>
struct access<SiPixelCPEGenericErrorParm::DbEntryBinSize>
{
    static bool equal_(const SiPixelCPEGenericErrorParm::DbEntryBinSize & first, const SiPixelCPEGenericErrorParm::DbEntryBinSize & second)
    {
        return true
            and (equal(first.partBin_size, second.partBin_size))
            and (equal(first.sizeBin_size, second.sizeBin_size))
            and (equal(first.alphaBin_size, second.alphaBin_size))
            and (equal(first.betaBin_size, second.betaBin_size))
        ;
    }
};

template <>
struct access<SiPixelCalibConfiguration>
{
    static bool equal_(const SiPixelCalibConfiguration & first, const SiPixelCalibConfiguration & second)
    {
        return true
            and (equal(first.fNTriggers, second.fNTriggers))
            and (equal(first.fRowPattern, second.fRowPattern))
            and (equal(first.fColumnPattern, second.fColumnPattern))
            and (equal(first.fVCalValues, second.fVCalValues))
            and (equal(first.fMode, second.fMode))
        ;
    }
};

template <>
struct access<SiPixelDbItem>
{
    static bool equal_(const SiPixelDbItem & first, const SiPixelDbItem & second)
    {
        return true
            and (equal(first.packedVal_, second.packedVal_))
        ;
    }
};

template <>
struct access<SiPixelDisabledModules>
{
    static bool equal_(const SiPixelDisabledModules & first, const SiPixelDisabledModules & second)
    {
        return true
            and (equal(first.theDisabledModules, second.theDisabledModules))
        ;
    }
};

template <>
struct access<SiPixelFedCabling>
{
    static bool equal_(const SiPixelFedCabling & first, const SiPixelFedCabling & second)
    {
        return true
        ;
    }
};

template <>
struct access<SiPixelFedCablingMap>
{
    static bool equal_(const SiPixelFedCablingMap & first, const SiPixelFedCablingMap & second)
    {
        return true
            and (equal(static_cast<const SiPixelFedCabling &>(first), static_cast<const SiPixelFedCabling &>(second)))
            and (equal(first.theVersion, second.theVersion))
            and (equal(first.theMap, second.theMap))
        ;
    }
};

template <>
struct access<SiPixelFedCablingMap::Key>
{
    static bool equal_(const SiPixelFedCablingMap::Key & first, const SiPixelFedCablingMap::Key & second)
    {
        return true
            and (equal(first.fed, second.fed))
            and (equal(first.link, second.link))
            and (equal(first.roc, second.roc))
        ;
    }
};

template <>
struct access<SiPixelGainCalibration>
{
    static bool equal_(const SiPixelGainCalibration & first, const SiPixelGainCalibration & second)
    {
        return true
            and (equal(first.v_pedestals, second.v_pedestals))
            and (equal(first.indexes, second.indexes))
            and (equal(first.minPed_, second.minPed_))
            and (equal(first.maxPed_, second.maxPed_))
            and (equal(first.minGain_, second.minGain_))
            and (equal(first.maxGain_, second.maxGain_))
            and (equal(first.numberOfRowsToAverageOver_, second.numberOfRowsToAverageOver_))
            and (equal(first.nBinsToUseForEncoding_, second.nBinsToUseForEncoding_))
            and (equal(first.deadFlag_, second.deadFlag_))
            and (equal(first.noisyFlag_, second.noisyFlag_))
        ;
    }
};

template <>
struct access<SiPixelGainCalibration::DetRegistry>
{
    static bool equal_(const SiPixelGainCalibration::DetRegistry & first, const SiPixelGainCalibration::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
            and (equal(first.iend, second.iend))
            and (equal(first.ncols, second.ncols))
        ;
    }
};

template <>
struct access<SiPixelGainCalibrationForHLT>
{
    static bool equal_(const SiPixelGainCalibrationForHLT & first, const SiPixelGainCalibrationForHLT & second)
    {
        return true
            and (equal(first.v_pedestals, second.v_pedestals))
            and (equal(first.indexes, second.indexes))
            and (equal(first.minPed_, second.minPed_))
            and (equal(first.maxPed_, second.maxPed_))
            and (equal(first.minGain_, second.minGain_))
            and (equal(first.maxGain_, second.maxGain_))
            and (equal(first.pedPrecision, second.pedPrecision))
            and (equal(first.gainPrecision, second.gainPrecision))
            and (equal(first.numberOfRowsToAverageOver_, second.numberOfRowsToAverageOver_))
            and (equal(first.nBinsToUseForEncoding_, second.nBinsToUseForEncoding_))
            and (equal(first.deadFlag_, second.deadFlag_))
            and (equal(first.noisyFlag_, second.noisyFlag_))
        ;
    }
};

template <>
struct access<SiPixelGainCalibrationForHLT::DetRegistry>
{
    static bool equal_(const SiPixelGainCalibrationForHLT::DetRegistry & first, const SiPixelGainCalibrationForHLT::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
            and (equal(first.iend, second.iend))
            and (equal(first.ncols, second.ncols))
        ;
    }
};

template <>
struct access<SiPixelGainCalibrationOffline>
{
    static bool equal_(const SiPixelGainCalibrationOffline & first, const SiPixelGainCalibrationOffline & second)
    {
        return true
            and (equal(first.v_pedestals, second.v_pedestals))
            and (equal(first.indexes, second.indexes))
            and (equal(first.minPed_, second.minPed_))
            and (equal(first.maxPed_, second.maxPed_))
            and (equal(first.minGain_, second.minGain_))
            and (equal(first.maxGain_, second.maxGain_))
            and (equal(first.numberOfRowsToAverageOver_, second.numberOfRowsToAverageOver_))
            and (equal(first.nBinsToUseForEncoding_, second.nBinsToUseForEncoding_))
            and (equal(first.deadFlag_, second.deadFlag_))
            and (equal(first.noisyFlag_, second.noisyFlag_))
        ;
    }
};

template <>
struct access<SiPixelGainCalibrationOffline::DetRegistry>
{
    static bool equal_(const SiPixelGainCalibrationOffline::DetRegistry & first, const SiPixelGainCalibrationOffline::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
            and (equal(first.iend, second.iend))
            and (equal(first.ncols, second.ncols))
        ;
    }
};

template <>
struct access<SiPixelLorentzAngle>
{
    static bool equal_(const SiPixelLorentzAngle & first, const SiPixelLorentzAngle & second)
    {
        return true
            and (equal(first.m_LA, second.m_LA))
        ;
    }
};

template <>
struct access<SiPixelPedestals>
{
    static bool equal_(const SiPixelPedestals & first, const SiPixelPedestals & second)
    {
        return true
            and (equal(first.m_pedestals, second.m_pedestals))
        ;
    }
};

template <>
struct access<SiPixelPerformanceSummary>
{
    static bool equal_(const SiPixelPerformanceSummary & first, const SiPixelPerformanceSummary & second)
    {
        return true
            and (equal(first.timeStamp_, second.timeStamp_))
            and (equal(first.runNumber_, second.runNumber_))
            and (equal(first.luminosityBlock_, second.luminosityBlock_))
            and (equal(first.numberOfEvents_, second.numberOfEvents_))
            and (equal(first.allDetSummaries_, second.allDetSummaries_))
        ;
    }
};

template <>
struct access<SiPixelPerformanceSummary::DetSummary>
{
    static bool equal_(const SiPixelPerformanceSummary::DetSummary & first, const SiPixelPerformanceSummary::DetSummary & second)
    {
        return true
            and (equal(first.detId_, second.detId_))
            and (equal(first.performanceValues_, second.performanceValues_))
        ;
    }
};

template <>
struct access<SiPixelQuality>
{
    static bool equal_(const SiPixelQuality & first, const SiPixelQuality & second)
    {
        return true
            and (equal(first.theDisabledModules, second.theDisabledModules))
        ;
    }
};

template <>
struct access<SiPixelQuality::disabledModuleType>
{
    static bool equal_(const SiPixelQuality::disabledModuleType & first, const SiPixelQuality::disabledModuleType & second)
    {
        return true
            and (equal(first.DetID, second.DetID))
            and (equal(first.errorType, second.errorType))
            and (equal(first.BadRocs, second.BadRocs))
        ;
    }
};

template <>
struct access<SiPixelTemplateDBObject>
{
    static bool equal_(const SiPixelTemplateDBObject & first, const SiPixelTemplateDBObject & second)
    {
        return true
            and (equal(first.index_, second.index_))
            and (equal(first.maxIndex_, second.maxIndex_))
            and (equal(first.numOfTempl_, second.numOfTempl_))
            and (equal(first.version_, second.version_))
            and (equal(first.isInvalid_, second.isInvalid_))
            and (equal(first.sVector_, second.sVector_))
            and (equal(first.templ_ID, second.templ_ID))
        ;
    }
};

template <>
struct access<sipixelobjects::PixelROC>
{
    static bool equal_(const sipixelobjects::PixelROC & first, const sipixelobjects::PixelROC & second)
    {
        return true
            and (equal(first.theDetUnit, second.theDetUnit))
            and (equal(first.theIdDU, second.theIdDU))
            and (equal(first.theIdLk, second.theIdLk))
        ;
    }
};

}
}

#endif
