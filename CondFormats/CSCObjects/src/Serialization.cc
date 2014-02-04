
#include "CondFormats/CSCObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void CSCBadChambers::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(numberOfBadChambers);
    ar & BOOST_SERIALIZATION_NVP(chambers);
}
COND_SERIALIZATION_INSTANTIATE(CSCBadChambers);

template <class Archive>
void CSCBadStrips::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(numberOfBadChannels);
    ar & BOOST_SERIALIZATION_NVP(chambers);
    ar & BOOST_SERIALIZATION_NVP(channels);
}
COND_SERIALIZATION_INSTANTIATE(CSCBadStrips);

template <class Archive>
void CSCBadStrips::BadChamber::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(chamber_index);
    ar & BOOST_SERIALIZATION_NVP(pointer);
    ar & BOOST_SERIALIZATION_NVP(bad_channels);
}
COND_SERIALIZATION_INSTANTIATE(CSCBadStrips::BadChamber);

template <class Archive>
void CSCBadStrips::BadChannel::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(layer);
    ar & BOOST_SERIALIZATION_NVP(channel);
    ar & BOOST_SERIALIZATION_NVP(flag1);
    ar & BOOST_SERIALIZATION_NVP(flag2);
    ar & BOOST_SERIALIZATION_NVP(flag3);
}
COND_SERIALIZATION_INSTANTIATE(CSCBadStrips::BadChannel);

template <class Archive>
void CSCBadWires::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(numberOfBadChannels);
    ar & BOOST_SERIALIZATION_NVP(chambers);
    ar & BOOST_SERIALIZATION_NVP(channels);
}
COND_SERIALIZATION_INSTANTIATE(CSCBadWires);

template <class Archive>
void CSCBadWires::BadChamber::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(chamber_index);
    ar & BOOST_SERIALIZATION_NVP(pointer);
    ar & BOOST_SERIALIZATION_NVP(bad_channels);
}
COND_SERIALIZATION_INSTANTIATE(CSCBadWires::BadChamber);

template <class Archive>
void CSCBadWires::BadChannel::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(layer);
    ar & BOOST_SERIALIZATION_NVP(channel);
    ar & BOOST_SERIALIZATION_NVP(flag1);
    ar & BOOST_SERIALIZATION_NVP(flag2);
    ar & BOOST_SERIALIZATION_NVP(flag3);
}
COND_SERIALIZATION_INSTANTIATE(CSCBadWires::BadChannel);

template <class Archive>
void CSCChamberIndex::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ch_index);
}
COND_SERIALIZATION_INSTANTIATE(CSCChamberIndex);

template <class Archive>
void CSCChamberMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ch_map);
}
COND_SERIALIZATION_INSTANTIATE(CSCChamberMap);

template <class Archive>
void CSCChamberTimeCorrections::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_precision);
    ar & BOOST_SERIALIZATION_NVP(chamberCorrections);
}
COND_SERIALIZATION_INSTANTIATE(CSCChamberTimeCorrections);

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
COND_SERIALIZATION_INSTANTIATE(CSCChamberTimeCorrections::ChamberTimeCorrections);

template <class Archive>
void CSCCrateMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(crate_map);
}
COND_SERIALIZATION_INSTANTIATE(CSCCrateMap);

template <class Archive>
void CSCDBChipSpeedCorrection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_speedCorr);
    ar & BOOST_SERIALIZATION_NVP(chipSpeedCorr);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBChipSpeedCorrection);

template <class Archive>
void CSCDBChipSpeedCorrection::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(speedCorr);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBChipSpeedCorrection::Item);

template <class Archive>
void CSCDBCrosstalk::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_slope);
    ar & BOOST_SERIALIZATION_NVP(factor_intercept);
    ar & BOOST_SERIALIZATION_NVP(crosstalk);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBCrosstalk);

template <class Archive>
void CSCDBCrosstalk::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(xtalk_slope_right);
    ar & BOOST_SERIALIZATION_NVP(xtalk_intercept_right);
    ar & BOOST_SERIALIZATION_NVP(xtalk_slope_left);
    ar & BOOST_SERIALIZATION_NVP(xtalk_intercept_left);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBCrosstalk::Item);

template <class Archive>
void CSCDBGains::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_gain);
    ar & BOOST_SERIALIZATION_NVP(gains);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBGains);

template <class Archive>
void CSCDBGains::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gain_slope);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBGains::Item);

template <class Archive>
void CSCDBGasGainCorrection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gasGainCorr);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBGasGainCorrection);

template <class Archive>
void CSCDBGasGainCorrection::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gainCorr);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBGasGainCorrection::Item);

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
COND_SERIALIZATION_INSTANTIATE(CSCDBL1TPParameters);

template <class Archive>
void CSCDBNoiseMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_noise);
    ar & BOOST_SERIALIZATION_NVP(matrix);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBNoiseMatrix);

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
COND_SERIALIZATION_INSTANTIATE(CSCDBNoiseMatrix::Item);

template <class Archive>
void CSCDBPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(factor_ped);
    ar & BOOST_SERIALIZATION_NVP(factor_rms);
    ar & BOOST_SERIALIZATION_NVP(pedestals);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBPedestals);

template <class Archive>
void CSCDBPedestals::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ped);
    ar & BOOST_SERIALIZATION_NVP(rms);
}
COND_SERIALIZATION_INSTANTIATE(CSCDBPedestals::Item);

template <class Archive>
void CSCDDUMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ddu_map);
}
COND_SERIALIZATION_INSTANTIATE(CSCDDUMap);

template <class Archive>
void CSCGains::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gains);
}
COND_SERIALIZATION_INSTANTIATE(CSCGains);

template <class Archive>
void CSCGains::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gain_slope);
    ar & BOOST_SERIALIZATION_NVP(gain_intercept);
    ar & BOOST_SERIALIZATION_NVP(gain_chi2);
}
COND_SERIALIZATION_INSTANTIATE(CSCGains::Item);

template <class Archive>
void CSCIdentifier::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(identifier);
}
COND_SERIALIZATION_INSTANTIATE(CSCIdentifier);

template <class Archive>
void CSCIdentifier::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(CSCid);
}
COND_SERIALIZATION_INSTANTIATE(CSCIdentifier::Item);

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
COND_SERIALIZATION_INSTANTIATE(CSCL1TPParameters);

template <class Archive>
void CSCMapItem::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(CSCMapItem);

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
COND_SERIALIZATION_INSTANTIATE(CSCMapItem::MapItem);

template <class Archive>
void CSCNoiseMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(matrix);
}
COND_SERIALIZATION_INSTANTIATE(CSCNoiseMatrix);

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
COND_SERIALIZATION_INSTANTIATE(CSCNoiseMatrix::Item);

template <class Archive>
void CSCPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pedestals);
}
COND_SERIALIZATION_INSTANTIATE(CSCPedestals);

template <class Archive>
void CSCPedestals::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ped);
    ar & BOOST_SERIALIZATION_NVP(rms);
}
COND_SERIALIZATION_INSTANTIATE(CSCPedestals::Item);

template <class Archive>
void CSCReadoutMapping::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mapping_);
    ar & BOOST_SERIALIZATION_NVP(sw2hw_);
}
COND_SERIALIZATION_INSTANTIATE(CSCReadoutMapping);

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
COND_SERIALIZATION_INSTANTIATE(CSCReadoutMapping::CSCLabel);

template <class Archive>
void CSCTriggerMapping::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mapping_);
}
COND_SERIALIZATION_INSTANTIATE(CSCTriggerMapping);

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
COND_SERIALIZATION_INSTANTIATE(CSCTriggerMapping::CSCTriggerConnection);

template <class Archive>
void CSCcrosstalk::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(crosstalk);
}
COND_SERIALIZATION_INSTANTIATE(CSCcrosstalk);

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
COND_SERIALIZATION_INSTANTIATE(CSCcrosstalk::Item);

template <class Archive>
void cscdqm::DCSAddressType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(iendcap);
    ar & BOOST_SERIALIZATION_NVP(istation);
    ar & BOOST_SERIALIZATION_NVP(iring);
    ar & BOOST_SERIALIZATION_NVP(ichamber);
}
COND_SERIALIZATION_INSTANTIATE(cscdqm::DCSAddressType);

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
COND_SERIALIZATION_INSTANTIATE(cscdqm::DCSData);

template <class Archive>
void cscdqm::HVVMeasType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(adr);
    ar & BOOST_SERIALIZATION_NVP(position);
    ar & BOOST_SERIALIZATION_NVP(value);
}
COND_SERIALIZATION_INSTANTIATE(cscdqm::HVVMeasType);

template <class Archive>
void cscdqm::LVIMeasType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(adr);
    ar & BOOST_SERIALIZATION_NVP(board);
    ar & BOOST_SERIALIZATION_NVP(boardId);
    ar & BOOST_SERIALIZATION_NVP(nominal_v);
    ar & BOOST_SERIALIZATION_NVP(value);
}
COND_SERIALIZATION_INSTANTIATE(cscdqm::LVIMeasType);

template <class Archive>
void cscdqm::LVVMeasType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(adr);
    ar & BOOST_SERIALIZATION_NVP(board);
    ar & BOOST_SERIALIZATION_NVP(boardId);
    ar & BOOST_SERIALIZATION_NVP(nominal_v);
}
COND_SERIALIZATION_INSTANTIATE(cscdqm::LVVMeasType);

template <class Archive>
void cscdqm::TempMeasType::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(adr);
    ar & BOOST_SERIALIZATION_NVP(board);
    ar & BOOST_SERIALIZATION_NVP(boardId);
    ar & BOOST_SERIALIZATION_NVP(value);
}
COND_SERIALIZATION_INSTANTIATE(cscdqm::TempMeasType);

#include "CondFormats/CSCObjects/src/SerializationManual.h"
