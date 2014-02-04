
#include "CondFormats/RunInfo/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void FillInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_isData);
    ar & BOOST_SERIALIZATION_NVP(m_lhcFill);
    ar & BOOST_SERIALIZATION_NVP(m_bunches1);
    ar & BOOST_SERIALIZATION_NVP(m_bunches2);
    ar & BOOST_SERIALIZATION_NVP(m_collidingBunches);
    ar & BOOST_SERIALIZATION_NVP(m_targetBunches);
    ar & BOOST_SERIALIZATION_NVP(m_fillType);
    ar & BOOST_SERIALIZATION_NVP(m_particles1);
    ar & BOOST_SERIALIZATION_NVP(m_particles2);
    ar & BOOST_SERIALIZATION_NVP(m_crossingAngle);
    ar & BOOST_SERIALIZATION_NVP(m_betastar);
    ar & BOOST_SERIALIZATION_NVP(m_intensity1);
    ar & BOOST_SERIALIZATION_NVP(m_intensity2);
    ar & BOOST_SERIALIZATION_NVP(m_energy);
    ar & BOOST_SERIALIZATION_NVP(m_createTime);
    ar & BOOST_SERIALIZATION_NVP(m_beginTime);
    ar & BOOST_SERIALIZATION_NVP(m_endTime);
    ar & BOOST_SERIALIZATION_NVP(m_injectionScheme);
    ar & BOOST_SERIALIZATION_NVP(m_bunchConfiguration1);
    ar & BOOST_SERIALIZATION_NVP(m_bunchConfiguration2);
}
COND_SERIALIZATION_INSTANTIATE(FillInfo);

template <class Archive>
void L1TriggerScaler::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_run);
    ar & BOOST_SERIALIZATION_NVP(m_lumisegment);
    ar & BOOST_SERIALIZATION_NVP(m_runnumber);
}
COND_SERIALIZATION_INSTANTIATE(L1TriggerScaler);

template <class Archive>
void L1TriggerScaler::Lumi::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_runnumber);
    ar & BOOST_SERIALIZATION_NVP(m_lumi_id);
    ar & BOOST_SERIALIZATION_NVP(m_start_time);
    ar & BOOST_SERIALIZATION_NVP(m_string_format);
    ar & BOOST_SERIALIZATION_NVP(m_rn);
    ar & BOOST_SERIALIZATION_NVP(m_lumisegment);
    ar & BOOST_SERIALIZATION_NVP(m_date);
    ar & BOOST_SERIALIZATION_NVP(m_GTAlgoCounts);
    ar & BOOST_SERIALIZATION_NVP(m_GTAlgoRates);
    ar & BOOST_SERIALIZATION_NVP(m_GTAlgoPrescaling);
    ar & BOOST_SERIALIZATION_NVP(m_GTTechCounts);
    ar & BOOST_SERIALIZATION_NVP(m_GTTechRates);
    ar & BOOST_SERIALIZATION_NVP(m_GTTechPrescaling);
    ar & BOOST_SERIALIZATION_NVP(m_GTPartition0TriggerCounts);
    ar & BOOST_SERIALIZATION_NVP(m_GTPartition0TriggerRates);
    ar & BOOST_SERIALIZATION_NVP(m_GTPartition0DeadTime);
    ar & BOOST_SERIALIZATION_NVP(m_GTPartition0DeadTimeRatio);
}
COND_SERIALIZATION_INSTANTIATE(L1TriggerScaler::Lumi);

template <class Archive>
void MixingInputConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(t_);
    ar & BOOST_SERIALIZATION_NVP(an_);
    ar & BOOST_SERIALIZATION_NVP(dpfv_);
    ar & BOOST_SERIALIZATION_NVP(dp_);
    ar & BOOST_SERIALIZATION_NVP(moot_);
    ar & BOOST_SERIALIZATION_NVP(ioot_);
}
COND_SERIALIZATION_INSTANTIATE(MixingInputConfig);

template <class Archive>
void MixingModuleConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(configs_);
    ar & BOOST_SERIALIZATION_NVP(minb_);
    ar & BOOST_SERIALIZATION_NVP(maxb_);
    ar & BOOST_SERIALIZATION_NVP(bs_);
}
COND_SERIALIZATION_INSTANTIATE(MixingModuleConfig);

template <class Archive>
void RunInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_run);
    ar & BOOST_SERIALIZATION_NVP(m_start_time_ll);
    ar & BOOST_SERIALIZATION_NVP(m_start_time_str);
    ar & BOOST_SERIALIZATION_NVP(m_stop_time_ll);
    ar & BOOST_SERIALIZATION_NVP(m_stop_time_str);
    ar & BOOST_SERIALIZATION_NVP(m_fed_in);
    ar & BOOST_SERIALIZATION_NVP(m_start_current);
    ar & BOOST_SERIALIZATION_NVP(m_stop_current);
    ar & BOOST_SERIALIZATION_NVP(m_avg_current);
    ar & BOOST_SERIALIZATION_NVP(m_max_current);
    ar & BOOST_SERIALIZATION_NVP(m_min_current);
    ar & BOOST_SERIALIZATION_NVP(m_run_intervall_micros);
    ar & BOOST_SERIALIZATION_NVP(m_current);
    ar & BOOST_SERIALIZATION_NVP(m_times_of_currents);
}
COND_SERIALIZATION_INSTANTIATE(RunInfo);

template <class Archive>
void RunSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_run);
    ar & BOOST_SERIALIZATION_NVP(m_name);
    ar & BOOST_SERIALIZATION_NVP(m_start_time_ll);
    ar & BOOST_SERIALIZATION_NVP(m_start_time_str);
    ar & BOOST_SERIALIZATION_NVP(m_stop_time_ll);
    ar & BOOST_SERIALIZATION_NVP(m_stop_time_str);
    ar & BOOST_SERIALIZATION_NVP(m_lumisections);
    ar & BOOST_SERIALIZATION_NVP(m_subdt_in);
    ar & BOOST_SERIALIZATION_NVP(m_hltkey);
    ar & BOOST_SERIALIZATION_NVP(m_nevents);
    ar & BOOST_SERIALIZATION_NVP(m_rate);
}
COND_SERIALIZATION_INSTANTIATE(RunSummary);

template <class Archive>
void runinfo_test::RunNumber::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_runnumber);
}
COND_SERIALIZATION_INSTANTIATE(runinfo_test::RunNumber);

template <class Archive>
void runinfo_test::RunNumber::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_run);
    ar & BOOST_SERIALIZATION_NVP(m_id_start);
    ar & BOOST_SERIALIZATION_NVP(m_id_stop);
    ar & BOOST_SERIALIZATION_NVP(m_number);
    ar & BOOST_SERIALIZATION_NVP(m_name);
    ar & BOOST_SERIALIZATION_NVP(m_start_time_sll);
    ar & BOOST_SERIALIZATION_NVP(m_start_time_str);
    ar & BOOST_SERIALIZATION_NVP(m_stop_time_sll);
    ar & BOOST_SERIALIZATION_NVP(m_stop_time_str);
    ar & BOOST_SERIALIZATION_NVP(m_lumisections);
    ar & BOOST_SERIALIZATION_NVP(m_subdt_joined);
    ar & BOOST_SERIALIZATION_NVP(m_subdt_in);
}
COND_SERIALIZATION_INSTANTIATE(runinfo_test::RunNumber::Item);

#include "CondFormats/RunInfo/src/SerializationManual.h"
