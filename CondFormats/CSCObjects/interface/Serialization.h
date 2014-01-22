#ifndef CondFormats_CSCObjects_Serialization_H
#define CondFormats_CSCObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void CSCBadChambers::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(numberOfBadChambers);
    ar & BOOST_SERIALIZATION_NVP(chambers);
}

template <class Archive>
void CSCBadStrips::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(numberOfBadChannels);
    ar & BOOST_SERIALIZATION_NVP(chambers);
    ar & BOOST_SERIALIZATION_NVP(channels);
}

template <class Archive>
void CSCBadStrips::BadChamber::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(chamber_index);
    ar & BOOST_SERIALIZATION_NVP(pointer);
    ar & BOOST_SERIALIZATION_NVP(bad_channels);
}

template <class Archive>
void CSCBadStrips::BadChannel::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(layer);
    ar & BOOST_SERIALIZATION_NVP(channel);
    ar & BOOST_SERIALIZATION_NVP(flag1);
    ar & BOOST_SERIALIZATION_NVP(flag2);
    ar & BOOST_SERIALIZATION_NVP(flag3);
}

template <class Archive>
void CSCBadWires::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(numberOfBadChannels);
    ar & BOOST_SERIALIZATION_NVP(chambers);
    ar & BOOST_SERIALIZATION_NVP(channels);
}

template <class Archive>
void CSCBadWires::BadChamber::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(chamber_index);
    ar & BOOST_SERIALIZATION_NVP(pointer);
    ar & BOOST_SERIALIZATION_NVP(bad_channels);
}

template <class Archive>
void CSCBadWires::BadChannel::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(layer);
    ar & BOOST_SERIALIZATION_NVP(channel);
    ar & BOOST_SERIALIZATION_NVP(flag1);
    ar & BOOST_SERIALIZATION_NVP(flag2);
    ar & BOOST_SERIALIZATION_NVP(flag3);
}

template <class Archive>
void CSCChamberIndex::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ch_index);
}

template <class Archive>
void CSCChamberMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ch_map);
}

template <class Archive>
void CSCChamberTimeCorrections::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_precision);
    ar & BOOST_SERIALIZATION_NVP(chamberCorrections);
}

template <class Archive>
void CSCChamberTimeCorrections::ChamberTimeCorrections::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cfeb_length);
    ar & BOOST_SERIALIZATION_NVP(cfeb_rev);
    ar & BOOST_SERIALIZATION_NVP(alct_length);
    ar & BOOST_SERIALIZATION_NVP(alct_rev);
    ar & BOOST_SERIALIZATION_NVP(cfeb_tmb_skew_delay);
    ar & BOOST_SERIALIZATION_NVP(cfeb_timing_corr);
    ar & BOOST_SERIALIZATION_NVP(cfeb_cable_delay);
    ar & BOOST_SERIALIZATION_NVP(anode_bx_offset);
}

template <class Archive>
void CSCCrateMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(crate_map);
}

template <class Archive>
void CSCDBChipSpeedCorrection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_speedCorr);
    ar & BOOST_SERIALIZATION_NVP(chipSpeedCorr);
}

template <class Archive>
void CSCDBChipSpeedCorrection::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(speedCorr);
}

template <class Archive>
void CSCDBCrosstalk::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_slope);
    ar & BOOST_SERIALIZATION_NVP(factor_intercept);
    ar & BOOST_SERIALIZATION_NVP(crosstalk);
}

template <class Archive>
void CSCDBCrosstalk::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(xtalk_slope_right);
    ar & BOOST_SERIALIZATION_NVP(xtalk_intercept_right);
    ar & BOOST_SERIALIZATION_NVP(xtalk_slope_left);
    ar & BOOST_SERIALIZATION_NVP(xtalk_intercept_left);
}

template <class Archive>
void CSCDBGains::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_gain);
    ar & BOOST_SERIALIZATION_NVP(gains);
}

template <class Archive>
void CSCDBGains::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gain_slope);
}

template <class Archive>
void CSCDBGasGainCorrection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gasGainCorr);
}

template <class Archive>
void CSCDBGasGainCorrection::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gainCorr);
}

template <class Archive>
void CSCDBL1TPParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_alct_fifo_tbins);
    ar & BOOST_SERIALIZATION_NVP(m_alct_fifo_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_alct_drift_delay);
    ar & BOOST_SERIALIZATION_NVP(m_alct_nplanes_hit_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_alct_nplanes_hit_accel_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_alct_nplanes_hit_pattern);
    ar & BOOST_SERIALIZATION_NVP(m_alct_nplanes_hit_accel_pattern);
    ar & BOOST_SERIALIZATION_NVP(m_alct_trig_mode);
    ar & BOOST_SERIALIZATION_NVP(m_alct_accel_mode);
    ar & BOOST_SERIALIZATION_NVP(m_alct_l1a_window_width);
    ar & BOOST_SERIALIZATION_NVP(m_clct_fifo_tbins);
    ar & BOOST_SERIALIZATION_NVP(m_clct_fifo_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_clct_hit_persist);
    ar & BOOST_SERIALIZATION_NVP(m_clct_drift_delay);
    ar & BOOST_SERIALIZATION_NVP(m_clct_nplanes_hit_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_clct_nplanes_hit_pattern);
    ar & BOOST_SERIALIZATION_NVP(m_clct_pid_thresh_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_clct_min_separation);
    ar & BOOST_SERIALIZATION_NVP(m_mpc_block_me1a);
    ar & BOOST_SERIALIZATION_NVP(m_alct_trig_enable);
    ar & BOOST_SERIALIZATION_NVP(m_clct_trig_enable);
    ar & BOOST_SERIALIZATION_NVP(m_match_trig_enable);
    ar & BOOST_SERIALIZATION_NVP(m_match_trig_window_size);
    ar & BOOST_SERIALIZATION_NVP(m_tmb_l1a_window_size);
}

template <class Archive>
void CSCDBNoiseMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_noise);
    ar & BOOST_SERIALIZATION_NVP(matrix);
}

template <class Archive>
void CSCDBNoiseMatrix::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(elem33);
    ar & BOOST_SERIALIZATION_NVP(elem34);
    ar & BOOST_SERIALIZATION_NVP(elem35);
    ar & BOOST_SERIALIZATION_NVP(elem44);
    ar & BOOST_SERIALIZATION_NVP(elem45);
    ar & BOOST_SERIALIZATION_NVP(elem46);
    ar & BOOST_SERIALIZATION_NVP(elem55);
    ar & BOOST_SERIALIZATION_NVP(elem56);
    ar & BOOST_SERIALIZATION_NVP(elem57);
    ar & BOOST_SERIALIZATION_NVP(elem66);
    ar & BOOST_SERIALIZATION_NVP(elem67);
    ar & BOOST_SERIALIZATION_NVP(elem77);
}

template <class Archive>
void CSCDBPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_ped);
    ar & BOOST_SERIALIZATION_NVP(factor_rms);
    ar & BOOST_SERIALIZATION_NVP(pedestals);
}

template <class Archive>
void CSCDBPedestals::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ped);
    ar & BOOST_SERIALIZATION_NVP(rms);
}

template <class Archive>
void CSCDDUMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ddu_map);
}

template <class Archive>
void CSCGains::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gains);
}

template <class Archive>
void CSCGains::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gain_slope);
    ar & BOOST_SERIALIZATION_NVP(gain_intercept);
    ar & BOOST_SERIALIZATION_NVP(gain_chi2);
}

template <class Archive>
void CSCIdentifier::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(identifier);
}

template <class Archive>
void CSCIdentifier::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(CSCid);
}

template <class Archive>
void CSCL1TPParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_alct_fifo_tbins);
    ar & BOOST_SERIALIZATION_NVP(m_alct_fifo_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_alct_drift_delay);
    ar & BOOST_SERIALIZATION_NVP(m_alct_nplanes_hit_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_alct_nplanes_hit_accel_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_alct_nplanes_hit_pattern);
    ar & BOOST_SERIALIZATION_NVP(m_alct_nplanes_hit_accel_pattern);
    ar & BOOST_SERIALIZATION_NVP(m_alct_trig_mode);
    ar & BOOST_SERIALIZATION_NVP(m_alct_accel_mode);
    ar & BOOST_SERIALIZATION_NVP(m_alct_l1a_window_width);
    ar & BOOST_SERIALIZATION_NVP(m_clct_fifo_tbins);
    ar & BOOST_SERIALIZATION_NVP(m_clct_fifo_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_clct_hit_persist);
    ar & BOOST_SERIALIZATION_NVP(m_clct_drift_delay);
    ar & BOOST_SERIALIZATION_NVP(m_clct_nplanes_hit_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_clct_nplanes_hit_pattern);
    ar & BOOST_SERIALIZATION_NVP(m_clct_pid_thresh_pretrig);
    ar & BOOST_SERIALIZATION_NVP(m_clct_min_separation);
}

template <class Archive>
void CSCMapItem::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void CSCMapItem::MapItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(chamberLabel);
    ar & BOOST_SERIALIZATION_NVP(chamberId);
    ar & BOOST_SERIALIZATION_NVP(endcap);
    ar & BOOST_SERIALIZATION_NVP(station);
    ar & BOOST_SERIALIZATION_NVP(ring);
    ar & BOOST_SERIALIZATION_NVP(chamber);
    ar & BOOST_SERIALIZATION_NVP(cscIndex);
    ar & BOOST_SERIALIZATION_NVP(layerIndex);
    ar & BOOST_SERIALIZATION_NVP(stripIndex);
    ar & BOOST_SERIALIZATION_NVP(anodeIndex);
    ar & BOOST_SERIALIZATION_NVP(strips);
    ar & BOOST_SERIALIZATION_NVP(anodes);
    ar & BOOST_SERIALIZATION_NVP(crateLabel);
    ar & BOOST_SERIALIZATION_NVP(crateid);
    ar & BOOST_SERIALIZATION_NVP(sector);
    ar & BOOST_SERIALIZATION_NVP(trig_sector);
    ar & BOOST_SERIALIZATION_NVP(dmb);
    ar & BOOST_SERIALIZATION_NVP(cscid);
    ar & BOOST_SERIALIZATION_NVP(ddu);
    ar & BOOST_SERIALIZATION_NVP(ddu_input);
    ar & BOOST_SERIALIZATION_NVP(slink);
    ar & BOOST_SERIALIZATION_NVP(fed_crate);
    ar & BOOST_SERIALIZATION_NVP(ddu_slot);
    ar & BOOST_SERIALIZATION_NVP(dcc_fifo);
    ar & BOOST_SERIALIZATION_NVP(fiber_crate);
    ar & BOOST_SERIALIZATION_NVP(fiber_pos);
    ar & BOOST_SERIALIZATION_NVP(fiber_socket);
}

template <class Archive>
void CSCNoiseMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(matrix);
}

template <class Archive>
void CSCNoiseMatrix::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(elem33);
    ar & BOOST_SERIALIZATION_NVP(elem34);
    ar & BOOST_SERIALIZATION_NVP(elem35);
    ar & BOOST_SERIALIZATION_NVP(elem44);
    ar & BOOST_SERIALIZATION_NVP(elem45);
    ar & BOOST_SERIALIZATION_NVP(elem46);
    ar & BOOST_SERIALIZATION_NVP(elem55);
    ar & BOOST_SERIALIZATION_NVP(elem56);
    ar & BOOST_SERIALIZATION_NVP(elem57);
    ar & BOOST_SERIALIZATION_NVP(elem66);
    ar & BOOST_SERIALIZATION_NVP(elem67);
    ar & BOOST_SERIALIZATION_NVP(elem77);
}

template <class Archive>
void CSCPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pedestals);
}

template <class Archive>
void CSCPedestals::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ped);
    ar & BOOST_SERIALIZATION_NVP(rms);
}

template <class Archive>
void CSCReadoutMapping::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mapping_);
    ar & BOOST_SERIALIZATION_NVP(sw2hw_);
}

template <class Archive>
void CSCReadoutMapping::CSCLabel::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(endcap_);
    ar & BOOST_SERIALIZATION_NVP(station_);
    ar & BOOST_SERIALIZATION_NVP(ring_);
    ar & BOOST_SERIALIZATION_NVP(chamber_);
    ar & BOOST_SERIALIZATION_NVP(vmecrate_);
    ar & BOOST_SERIALIZATION_NVP(dmb_);
    ar & BOOST_SERIALIZATION_NVP(tmb_);
    ar & BOOST_SERIALIZATION_NVP(tsector_);
    ar & BOOST_SERIALIZATION_NVP(cscid_);
    ar & BOOST_SERIALIZATION_NVP(ddu_);
    ar & BOOST_SERIALIZATION_NVP(dcc_);
}

template <class Archive>
void CSCTriggerMapping::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mapping_);
}

template <class Archive>
void CSCTriggerMapping::CSCTriggerConnection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(rendcap_);
    ar & BOOST_SERIALIZATION_NVP(rstation_);
    ar & BOOST_SERIALIZATION_NVP(rsector_);
    ar & BOOST_SERIALIZATION_NVP(rsubsector_);
    ar & BOOST_SERIALIZATION_NVP(rcscid_);
    ar & BOOST_SERIALIZATION_NVP(cendcap_);
    ar & BOOST_SERIALIZATION_NVP(cstation_);
    ar & BOOST_SERIALIZATION_NVP(csector_);
    ar & BOOST_SERIALIZATION_NVP(csubsector_);
    ar & BOOST_SERIALIZATION_NVP(ccscid_);
}

template <class Archive>
void CSCcrosstalk::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(crosstalk);
}

template <class Archive>
void CSCcrosstalk::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(xtalk_slope_right);
    ar & BOOST_SERIALIZATION_NVP(xtalk_intercept_right);
    ar & BOOST_SERIALIZATION_NVP(xtalk_chi2_right);
    ar & BOOST_SERIALIZATION_NVP(xtalk_slope_left);
    ar & BOOST_SERIALIZATION_NVP(xtalk_intercept_left);
    ar & BOOST_SERIALIZATION_NVP(xtalk_chi2_left);
}

template <class Archive>
void cscdqm::DCSAddressType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(iendcap);
    ar & BOOST_SERIALIZATION_NVP(istation);
    ar & BOOST_SERIALIZATION_NVP(iring);
    ar & BOOST_SERIALIZATION_NVP(ichamber);
}

template <class Archive>
void cscdqm::DCSData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(temp_meas);
    ar & BOOST_SERIALIZATION_NVP(hvv_meas);
    ar & BOOST_SERIALIZATION_NVP(lvv_meas);
    ar & BOOST_SERIALIZATION_NVP(lvi_meas);
    ar & BOOST_SERIALIZATION_NVP(temp_mode);
    ar & BOOST_SERIALIZATION_NVP(hvv_mode);
    ar & BOOST_SERIALIZATION_NVP(lvv_mode);
    ar & BOOST_SERIALIZATION_NVP(lvi_mode);
    ar & BOOST_SERIALIZATION_NVP(iov);
    ar & BOOST_SERIALIZATION_NVP(last_change);
}

template <class Archive>
void cscdqm::HVVMeasType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(adr);
    ar & BOOST_SERIALIZATION_NVP(position);
    ar & BOOST_SERIALIZATION_NVP(value);
}

template <class Archive>
void cscdqm::LVIMeasType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(adr);
    ar & BOOST_SERIALIZATION_NVP(board);
    ar & BOOST_SERIALIZATION_NVP(boardId);
    ar & BOOST_SERIALIZATION_NVP(nominal_v);
    ar & BOOST_SERIALIZATION_NVP(value);
}

template <class Archive>
void cscdqm::LVVMeasType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(adr);
    ar & BOOST_SERIALIZATION_NVP(board);
    ar & BOOST_SERIALIZATION_NVP(boardId);
    ar & BOOST_SERIALIZATION_NVP(nominal_v);
}

template <class Archive>
void cscdqm::TempMeasType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(adr);
    ar & BOOST_SERIALIZATION_NVP(board);
    ar & BOOST_SERIALIZATION_NVP(boardId);
    ar & BOOST_SERIALIZATION_NVP(value);
}

namespace cond {
namespace serialization {

template <>
struct access<CSCBadChambers>
{
    static bool equal_(const CSCBadChambers & first, const CSCBadChambers & second)
    {
        return true
            and (equal(first.numberOfBadChambers, second.numberOfBadChambers))
            and (equal(first.chambers, second.chambers))
        ;
    }
};

template <>
struct access<CSCBadStrips>
{
    static bool equal_(const CSCBadStrips & first, const CSCBadStrips & second)
    {
        return true
            and (equal(first.numberOfBadChannels, second.numberOfBadChannels))
            and (equal(first.chambers, second.chambers))
            and (equal(first.channels, second.channels))
        ;
    }
};

template <>
struct access<CSCBadStrips::BadChamber>
{
    static bool equal_(const CSCBadStrips::BadChamber & first, const CSCBadStrips::BadChamber & second)
    {
        return true
            and (equal(first.chamber_index, second.chamber_index))
            and (equal(first.pointer, second.pointer))
            and (equal(first.bad_channels, second.bad_channels))
        ;
    }
};

template <>
struct access<CSCBadStrips::BadChannel>
{
    static bool equal_(const CSCBadStrips::BadChannel & first, const CSCBadStrips::BadChannel & second)
    {
        return true
            and (equal(first.layer, second.layer))
            and (equal(first.channel, second.channel))
            and (equal(first.flag1, second.flag1))
            and (equal(first.flag2, second.flag2))
            and (equal(first.flag3, second.flag3))
        ;
    }
};

template <>
struct access<CSCBadWires>
{
    static bool equal_(const CSCBadWires & first, const CSCBadWires & second)
    {
        return true
            and (equal(first.numberOfBadChannels, second.numberOfBadChannels))
            and (equal(first.chambers, second.chambers))
            and (equal(first.channels, second.channels))
        ;
    }
};

template <>
struct access<CSCBadWires::BadChamber>
{
    static bool equal_(const CSCBadWires::BadChamber & first, const CSCBadWires::BadChamber & second)
    {
        return true
            and (equal(first.chamber_index, second.chamber_index))
            and (equal(first.pointer, second.pointer))
            and (equal(first.bad_channels, second.bad_channels))
        ;
    }
};

template <>
struct access<CSCBadWires::BadChannel>
{
    static bool equal_(const CSCBadWires::BadChannel & first, const CSCBadWires::BadChannel & second)
    {
        return true
            and (equal(first.layer, second.layer))
            and (equal(first.channel, second.channel))
            and (equal(first.flag1, second.flag1))
            and (equal(first.flag2, second.flag2))
            and (equal(first.flag3, second.flag3))
        ;
    }
};

template <>
struct access<CSCChamberIndex>
{
    static bool equal_(const CSCChamberIndex & first, const CSCChamberIndex & second)
    {
        return true
            and (equal(first.ch_index, second.ch_index))
        ;
    }
};

template <>
struct access<CSCChamberMap>
{
    static bool equal_(const CSCChamberMap & first, const CSCChamberMap & second)
    {
        return true
            and (equal(first.ch_map, second.ch_map))
        ;
    }
};

template <>
struct access<CSCChamberTimeCorrections>
{
    static bool equal_(const CSCChamberTimeCorrections & first, const CSCChamberTimeCorrections & second)
    {
        return true
            and (equal(first.factor_precision, second.factor_precision))
            and (equal(first.chamberCorrections, second.chamberCorrections))
        ;
    }
};

template <>
struct access<CSCChamberTimeCorrections::ChamberTimeCorrections>
{
    static bool equal_(const CSCChamberTimeCorrections::ChamberTimeCorrections & first, const CSCChamberTimeCorrections::ChamberTimeCorrections & second)
    {
        return true
            and (equal(first.cfeb_length, second.cfeb_length))
            and (equal(first.cfeb_rev, second.cfeb_rev))
            and (equal(first.alct_length, second.alct_length))
            and (equal(first.alct_rev, second.alct_rev))
            and (equal(first.cfeb_tmb_skew_delay, second.cfeb_tmb_skew_delay))
            and (equal(first.cfeb_timing_corr, second.cfeb_timing_corr))
            and (equal(first.cfeb_cable_delay, second.cfeb_cable_delay))
            and (equal(first.anode_bx_offset, second.anode_bx_offset))
        ;
    }
};

template <>
struct access<CSCCrateMap>
{
    static bool equal_(const CSCCrateMap & first, const CSCCrateMap & second)
    {
        return true
            and (equal(first.crate_map, second.crate_map))
        ;
    }
};

template <>
struct access<CSCDBChipSpeedCorrection>
{
    static bool equal_(const CSCDBChipSpeedCorrection & first, const CSCDBChipSpeedCorrection & second)
    {
        return true
            and (equal(first.factor_speedCorr, second.factor_speedCorr))
            and (equal(first.chipSpeedCorr, second.chipSpeedCorr))
        ;
    }
};

template <>
struct access<CSCDBChipSpeedCorrection::Item>
{
    static bool equal_(const CSCDBChipSpeedCorrection::Item & first, const CSCDBChipSpeedCorrection::Item & second)
    {
        return true
            and (equal(first.speedCorr, second.speedCorr))
        ;
    }
};

template <>
struct access<CSCDBCrosstalk>
{
    static bool equal_(const CSCDBCrosstalk & first, const CSCDBCrosstalk & second)
    {
        return true
            and (equal(first.factor_slope, second.factor_slope))
            and (equal(first.factor_intercept, second.factor_intercept))
            and (equal(first.crosstalk, second.crosstalk))
        ;
    }
};

template <>
struct access<CSCDBCrosstalk::Item>
{
    static bool equal_(const CSCDBCrosstalk::Item & first, const CSCDBCrosstalk::Item & second)
    {
        return true
            and (equal(first.xtalk_slope_right, second.xtalk_slope_right))
            and (equal(first.xtalk_intercept_right, second.xtalk_intercept_right))
            and (equal(first.xtalk_slope_left, second.xtalk_slope_left))
            and (equal(first.xtalk_intercept_left, second.xtalk_intercept_left))
        ;
    }
};

template <>
struct access<CSCDBGains>
{
    static bool equal_(const CSCDBGains & first, const CSCDBGains & second)
    {
        return true
            and (equal(first.factor_gain, second.factor_gain))
            and (equal(first.gains, second.gains))
        ;
    }
};

template <>
struct access<CSCDBGains::Item>
{
    static bool equal_(const CSCDBGains::Item & first, const CSCDBGains::Item & second)
    {
        return true
            and (equal(first.gain_slope, second.gain_slope))
        ;
    }
};

template <>
struct access<CSCDBGasGainCorrection>
{
    static bool equal_(const CSCDBGasGainCorrection & first, const CSCDBGasGainCorrection & second)
    {
        return true
            and (equal(first.gasGainCorr, second.gasGainCorr))
        ;
    }
};

template <>
struct access<CSCDBGasGainCorrection::Item>
{
    static bool equal_(const CSCDBGasGainCorrection::Item & first, const CSCDBGasGainCorrection::Item & second)
    {
        return true
            and (equal(first.gainCorr, second.gainCorr))
        ;
    }
};

template <>
struct access<CSCDBL1TPParameters>
{
    static bool equal_(const CSCDBL1TPParameters & first, const CSCDBL1TPParameters & second)
    {
        return true
            and (equal(first.m_alct_fifo_tbins, second.m_alct_fifo_tbins))
            and (equal(first.m_alct_fifo_pretrig, second.m_alct_fifo_pretrig))
            and (equal(first.m_alct_drift_delay, second.m_alct_drift_delay))
            and (equal(first.m_alct_nplanes_hit_pretrig, second.m_alct_nplanes_hit_pretrig))
            and (equal(first.m_alct_nplanes_hit_accel_pretrig, second.m_alct_nplanes_hit_accel_pretrig))
            and (equal(first.m_alct_nplanes_hit_pattern, second.m_alct_nplanes_hit_pattern))
            and (equal(first.m_alct_nplanes_hit_accel_pattern, second.m_alct_nplanes_hit_accel_pattern))
            and (equal(first.m_alct_trig_mode, second.m_alct_trig_mode))
            and (equal(first.m_alct_accel_mode, second.m_alct_accel_mode))
            and (equal(first.m_alct_l1a_window_width, second.m_alct_l1a_window_width))
            and (equal(first.m_clct_fifo_tbins, second.m_clct_fifo_tbins))
            and (equal(first.m_clct_fifo_pretrig, second.m_clct_fifo_pretrig))
            and (equal(first.m_clct_hit_persist, second.m_clct_hit_persist))
            and (equal(first.m_clct_drift_delay, second.m_clct_drift_delay))
            and (equal(first.m_clct_nplanes_hit_pretrig, second.m_clct_nplanes_hit_pretrig))
            and (equal(first.m_clct_nplanes_hit_pattern, second.m_clct_nplanes_hit_pattern))
            and (equal(first.m_clct_pid_thresh_pretrig, second.m_clct_pid_thresh_pretrig))
            and (equal(first.m_clct_min_separation, second.m_clct_min_separation))
            and (equal(first.m_mpc_block_me1a, second.m_mpc_block_me1a))
            and (equal(first.m_alct_trig_enable, second.m_alct_trig_enable))
            and (equal(first.m_clct_trig_enable, second.m_clct_trig_enable))
            and (equal(first.m_match_trig_enable, second.m_match_trig_enable))
            and (equal(first.m_match_trig_window_size, second.m_match_trig_window_size))
            and (equal(first.m_tmb_l1a_window_size, second.m_tmb_l1a_window_size))
        ;
    }
};

template <>
struct access<CSCDBNoiseMatrix>
{
    static bool equal_(const CSCDBNoiseMatrix & first, const CSCDBNoiseMatrix & second)
    {
        return true
            and (equal(first.factor_noise, second.factor_noise))
            and (equal(first.matrix, second.matrix))
        ;
    }
};

template <>
struct access<CSCDBNoiseMatrix::Item>
{
    static bool equal_(const CSCDBNoiseMatrix::Item & first, const CSCDBNoiseMatrix::Item & second)
    {
        return true
            and (equal(first.elem33, second.elem33))
            and (equal(first.elem34, second.elem34))
            and (equal(first.elem35, second.elem35))
            and (equal(first.elem44, second.elem44))
            and (equal(first.elem45, second.elem45))
            and (equal(first.elem46, second.elem46))
            and (equal(first.elem55, second.elem55))
            and (equal(first.elem56, second.elem56))
            and (equal(first.elem57, second.elem57))
            and (equal(first.elem66, second.elem66))
            and (equal(first.elem67, second.elem67))
            and (equal(first.elem77, second.elem77))
        ;
    }
};

template <>
struct access<CSCDBPedestals>
{
    static bool equal_(const CSCDBPedestals & first, const CSCDBPedestals & second)
    {
        return true
            and (equal(first.factor_ped, second.factor_ped))
            and (equal(first.factor_rms, second.factor_rms))
            and (equal(first.pedestals, second.pedestals))
        ;
    }
};

template <>
struct access<CSCDBPedestals::Item>
{
    static bool equal_(const CSCDBPedestals::Item & first, const CSCDBPedestals::Item & second)
    {
        return true
            and (equal(first.ped, second.ped))
            and (equal(first.rms, second.rms))
        ;
    }
};

template <>
struct access<CSCDDUMap>
{
    static bool equal_(const CSCDDUMap & first, const CSCDDUMap & second)
    {
        return true
            and (equal(first.ddu_map, second.ddu_map))
        ;
    }
};

template <>
struct access<CSCGains>
{
    static bool equal_(const CSCGains & first, const CSCGains & second)
    {
        return true
            and (equal(first.gains, second.gains))
        ;
    }
};

template <>
struct access<CSCGains::Item>
{
    static bool equal_(const CSCGains::Item & first, const CSCGains::Item & second)
    {
        return true
            and (equal(first.gain_slope, second.gain_slope))
            and (equal(first.gain_intercept, second.gain_intercept))
            and (equal(first.gain_chi2, second.gain_chi2))
        ;
    }
};

template <>
struct access<CSCIdentifier>
{
    static bool equal_(const CSCIdentifier & first, const CSCIdentifier & second)
    {
        return true
            and (equal(first.identifier, second.identifier))
        ;
    }
};

template <>
struct access<CSCIdentifier::Item>
{
    static bool equal_(const CSCIdentifier::Item & first, const CSCIdentifier::Item & second)
    {
        return true
            and (equal(first.CSCid, second.CSCid))
        ;
    }
};

template <>
struct access<CSCL1TPParameters>
{
    static bool equal_(const CSCL1TPParameters & first, const CSCL1TPParameters & second)
    {
        return true
            and (equal(first.m_alct_fifo_tbins, second.m_alct_fifo_tbins))
            and (equal(first.m_alct_fifo_pretrig, second.m_alct_fifo_pretrig))
            and (equal(first.m_alct_drift_delay, second.m_alct_drift_delay))
            and (equal(first.m_alct_nplanes_hit_pretrig, second.m_alct_nplanes_hit_pretrig))
            and (equal(first.m_alct_nplanes_hit_accel_pretrig, second.m_alct_nplanes_hit_accel_pretrig))
            and (equal(first.m_alct_nplanes_hit_pattern, second.m_alct_nplanes_hit_pattern))
            and (equal(first.m_alct_nplanes_hit_accel_pattern, second.m_alct_nplanes_hit_accel_pattern))
            and (equal(first.m_alct_trig_mode, second.m_alct_trig_mode))
            and (equal(first.m_alct_accel_mode, second.m_alct_accel_mode))
            and (equal(first.m_alct_l1a_window_width, second.m_alct_l1a_window_width))
            and (equal(first.m_clct_fifo_tbins, second.m_clct_fifo_tbins))
            and (equal(first.m_clct_fifo_pretrig, second.m_clct_fifo_pretrig))
            and (equal(first.m_clct_hit_persist, second.m_clct_hit_persist))
            and (equal(first.m_clct_drift_delay, second.m_clct_drift_delay))
            and (equal(first.m_clct_nplanes_hit_pretrig, second.m_clct_nplanes_hit_pretrig))
            and (equal(first.m_clct_nplanes_hit_pattern, second.m_clct_nplanes_hit_pattern))
            and (equal(first.m_clct_pid_thresh_pretrig, second.m_clct_pid_thresh_pretrig))
            and (equal(first.m_clct_min_separation, second.m_clct_min_separation))
        ;
    }
};

template <>
struct access<CSCMapItem>
{
    static bool equal_(const CSCMapItem & first, const CSCMapItem & second)
    {
        return true
        ;
    }
};

template <>
struct access<CSCMapItem::MapItem>
{
    static bool equal_(const CSCMapItem::MapItem & first, const CSCMapItem::MapItem & second)
    {
        return true
            and (equal(first.chamberLabel, second.chamberLabel))
            and (equal(first.chamberId, second.chamberId))
            and (equal(first.endcap, second.endcap))
            and (equal(first.station, second.station))
            and (equal(first.ring, second.ring))
            and (equal(first.chamber, second.chamber))
            and (equal(first.cscIndex, second.cscIndex))
            and (equal(first.layerIndex, second.layerIndex))
            and (equal(first.stripIndex, second.stripIndex))
            and (equal(first.anodeIndex, second.anodeIndex))
            and (equal(first.strips, second.strips))
            and (equal(first.anodes, second.anodes))
            and (equal(first.crateLabel, second.crateLabel))
            and (equal(first.crateid, second.crateid))
            and (equal(first.sector, second.sector))
            and (equal(first.trig_sector, second.trig_sector))
            and (equal(first.dmb, second.dmb))
            and (equal(first.cscid, second.cscid))
            and (equal(first.ddu, second.ddu))
            and (equal(first.ddu_input, second.ddu_input))
            and (equal(first.slink, second.slink))
            and (equal(first.fed_crate, second.fed_crate))
            and (equal(first.ddu_slot, second.ddu_slot))
            and (equal(first.dcc_fifo, second.dcc_fifo))
            and (equal(first.fiber_crate, second.fiber_crate))
            and (equal(first.fiber_pos, second.fiber_pos))
            and (equal(first.fiber_socket, second.fiber_socket))
        ;
    }
};

template <>
struct access<CSCNoiseMatrix>
{
    static bool equal_(const CSCNoiseMatrix & first, const CSCNoiseMatrix & second)
    {
        return true
            and (equal(first.matrix, second.matrix))
        ;
    }
};

template <>
struct access<CSCNoiseMatrix::Item>
{
    static bool equal_(const CSCNoiseMatrix::Item & first, const CSCNoiseMatrix::Item & second)
    {
        return true
            and (equal(first.elem33, second.elem33))
            and (equal(first.elem34, second.elem34))
            and (equal(first.elem35, second.elem35))
            and (equal(first.elem44, second.elem44))
            and (equal(first.elem45, second.elem45))
            and (equal(first.elem46, second.elem46))
            and (equal(first.elem55, second.elem55))
            and (equal(first.elem56, second.elem56))
            and (equal(first.elem57, second.elem57))
            and (equal(first.elem66, second.elem66))
            and (equal(first.elem67, second.elem67))
            and (equal(first.elem77, second.elem77))
        ;
    }
};

template <>
struct access<CSCPedestals>
{
    static bool equal_(const CSCPedestals & first, const CSCPedestals & second)
    {
        return true
            and (equal(first.pedestals, second.pedestals))
        ;
    }
};

template <>
struct access<CSCPedestals::Item>
{
    static bool equal_(const CSCPedestals::Item & first, const CSCPedestals::Item & second)
    {
        return true
            and (equal(first.ped, second.ped))
            and (equal(first.rms, second.rms))
        ;
    }
};

template <>
struct access<CSCReadoutMapping>
{
    static bool equal_(const CSCReadoutMapping & first, const CSCReadoutMapping & second)
    {
        return true
            and (equal(first.mapping_, second.mapping_))
            and (equal(first.sw2hw_, second.sw2hw_))
        ;
    }
};

template <>
struct access<CSCReadoutMapping::CSCLabel>
{
    static bool equal_(const CSCReadoutMapping::CSCLabel & first, const CSCReadoutMapping::CSCLabel & second)
    {
        return true
            and (equal(first.endcap_, second.endcap_))
            and (equal(first.station_, second.station_))
            and (equal(first.ring_, second.ring_))
            and (equal(first.chamber_, second.chamber_))
            and (equal(first.vmecrate_, second.vmecrate_))
            and (equal(first.dmb_, second.dmb_))
            and (equal(first.tmb_, second.tmb_))
            and (equal(first.tsector_, second.tsector_))
            and (equal(first.cscid_, second.cscid_))
            and (equal(first.ddu_, second.ddu_))
            and (equal(first.dcc_, second.dcc_))
        ;
    }
};

template <>
struct access<CSCTriggerMapping>
{
    static bool equal_(const CSCTriggerMapping & first, const CSCTriggerMapping & second)
    {
        return true
            and (equal(first.mapping_, second.mapping_))
        ;
    }
};

template <>
struct access<CSCTriggerMapping::CSCTriggerConnection>
{
    static bool equal_(const CSCTriggerMapping::CSCTriggerConnection & first, const CSCTriggerMapping::CSCTriggerConnection & second)
    {
        return true
            and (equal(first.rendcap_, second.rendcap_))
            and (equal(first.rstation_, second.rstation_))
            and (equal(first.rsector_, second.rsector_))
            and (equal(first.rsubsector_, second.rsubsector_))
            and (equal(first.rcscid_, second.rcscid_))
            and (equal(first.cendcap_, second.cendcap_))
            and (equal(first.cstation_, second.cstation_))
            and (equal(first.csector_, second.csector_))
            and (equal(first.csubsector_, second.csubsector_))
            and (equal(first.ccscid_, second.ccscid_))
        ;
    }
};

template <>
struct access<CSCcrosstalk>
{
    static bool equal_(const CSCcrosstalk & first, const CSCcrosstalk & second)
    {
        return true
            and (equal(first.crosstalk, second.crosstalk))
        ;
    }
};

template <>
struct access<CSCcrosstalk::Item>
{
    static bool equal_(const CSCcrosstalk::Item & first, const CSCcrosstalk::Item & second)
    {
        return true
            and (equal(first.xtalk_slope_right, second.xtalk_slope_right))
            and (equal(first.xtalk_intercept_right, second.xtalk_intercept_right))
            and (equal(first.xtalk_chi2_right, second.xtalk_chi2_right))
            and (equal(first.xtalk_slope_left, second.xtalk_slope_left))
            and (equal(first.xtalk_intercept_left, second.xtalk_intercept_left))
            and (equal(first.xtalk_chi2_left, second.xtalk_chi2_left))
        ;
    }
};

template <>
struct access<cscdqm::DCSAddressType>
{
    static bool equal_(const cscdqm::DCSAddressType & first, const cscdqm::DCSAddressType & second)
    {
        return true
            and (equal(first.iendcap, second.iendcap))
            and (equal(first.istation, second.istation))
            and (equal(first.iring, second.iring))
            and (equal(first.ichamber, second.ichamber))
        ;
    }
};

template <>
struct access<cscdqm::DCSData>
{
    static bool equal_(const cscdqm::DCSData & first, const cscdqm::DCSData & second)
    {
        return true
            and (equal(first.temp_meas, second.temp_meas))
            and (equal(first.hvv_meas, second.hvv_meas))
            and (equal(first.lvv_meas, second.lvv_meas))
            and (equal(first.lvi_meas, second.lvi_meas))
            and (equal(first.temp_mode, second.temp_mode))
            and (equal(first.hvv_mode, second.hvv_mode))
            and (equal(first.lvv_mode, second.lvv_mode))
            and (equal(first.lvi_mode, second.lvi_mode))
            and (equal(first.iov, second.iov))
            and (equal(first.last_change, second.last_change))
        ;
    }
};

template <>
struct access<cscdqm::HVVMeasType>
{
    static bool equal_(const cscdqm::HVVMeasType & first, const cscdqm::HVVMeasType & second)
    {
        return true
            and (equal(first.adr, second.adr))
            and (equal(first.position, second.position))
            and (equal(first.value, second.value))
        ;
    }
};

template <>
struct access<cscdqm::LVIMeasType>
{
    static bool equal_(const cscdqm::LVIMeasType & first, const cscdqm::LVIMeasType & second)
    {
        return true
            and (equal(first.adr, second.adr))
            and (equal(first.board, second.board))
            and (equal(first.boardId, second.boardId))
            and (equal(first.nominal_v, second.nominal_v))
            and (equal(first.value, second.value))
        ;
    }
};

template <>
struct access<cscdqm::LVVMeasType>
{
    static bool equal_(const cscdqm::LVVMeasType & first, const cscdqm::LVVMeasType & second)
    {
        return true
            and (equal(first.adr, second.adr))
            and (equal(first.board, second.board))
            and (equal(first.boardId, second.boardId))
            and (equal(first.nominal_v, second.nominal_v))
        ;
    }
};

template <>
struct access<cscdqm::TempMeasType>
{
    static bool equal_(const cscdqm::TempMeasType & first, const cscdqm::TempMeasType & second)
    {
        return true
            and (equal(first.adr, second.adr))
            and (equal(first.board, second.board))
            and (equal(first.boardId, second.boardId))
            and (equal(first.value, second.value))
        ;
    }
};

}
}

#endif
