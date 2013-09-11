#ifndef CondFormats_RunInfo_Serialization_H
#define CondFormats_RunInfo_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

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

template <class Archive>
void L1TriggerScaler::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_run);
    ar & BOOST_SERIALIZATION_NVP(m_lumisegment);
    ar & BOOST_SERIALIZATION_NVP(m_runnumber);
}

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

template <class Archive>
void MixingModuleConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(configs_);
    ar & BOOST_SERIALIZATION_NVP(minb_);
    ar & BOOST_SERIALIZATION_NVP(maxb_);
    ar & BOOST_SERIALIZATION_NVP(bs_);
}

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

template <class Archive>
void runinfo_test::RunNumber::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_runnumber);
}

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

namespace cond {
namespace serialization {

template <>
struct access<FillInfo>
{
    static bool equal_(const FillInfo & first, const FillInfo & second)
    {
        return true
            and (equal(first.m_isData, second.m_isData))
            and (equal(first.m_lhcFill, second.m_lhcFill))
            and (equal(first.m_bunches1, second.m_bunches1))
            and (equal(first.m_bunches2, second.m_bunches2))
            and (equal(first.m_collidingBunches, second.m_collidingBunches))
            and (equal(first.m_targetBunches, second.m_targetBunches))
            and (equal(first.m_fillType, second.m_fillType))
            and (equal(first.m_particles1, second.m_particles1))
            and (equal(first.m_particles2, second.m_particles2))
            and (equal(first.m_crossingAngle, second.m_crossingAngle))
            and (equal(first.m_betastar, second.m_betastar))
            and (equal(first.m_intensity1, second.m_intensity1))
            and (equal(first.m_intensity2, second.m_intensity2))
            and (equal(first.m_energy, second.m_energy))
            and (equal(first.m_createTime, second.m_createTime))
            and (equal(first.m_beginTime, second.m_beginTime))
            and (equal(first.m_endTime, second.m_endTime))
            and (equal(first.m_injectionScheme, second.m_injectionScheme))
            and (equal(first.m_bunchConfiguration1, second.m_bunchConfiguration1))
            and (equal(first.m_bunchConfiguration2, second.m_bunchConfiguration2))
        ;
    }
};

template <>
struct access<L1TriggerScaler>
{
    static bool equal_(const L1TriggerScaler & first, const L1TriggerScaler & second)
    {
        return true
            and (equal(first.m_run, second.m_run))
            and (equal(first.m_lumisegment, second.m_lumisegment))
            and (equal(first.m_runnumber, second.m_runnumber))
        ;
    }
};

template <>
struct access<L1TriggerScaler::Lumi>
{
    static bool equal_(const L1TriggerScaler::Lumi & first, const L1TriggerScaler::Lumi & second)
    {
        return true
            and (equal(first.m_runnumber, second.m_runnumber))
            and (equal(first.m_lumi_id, second.m_lumi_id))
            and (equal(first.m_start_time, second.m_start_time))
            and (equal(first.m_string_format, second.m_string_format))
            and (equal(first.m_rn, second.m_rn))
            and (equal(first.m_lumisegment, second.m_lumisegment))
            and (equal(first.m_date, second.m_date))
            and (equal(first.m_GTAlgoCounts, second.m_GTAlgoCounts))
            and (equal(first.m_GTAlgoRates, second.m_GTAlgoRates))
            and (equal(first.m_GTAlgoPrescaling, second.m_GTAlgoPrescaling))
            and (equal(first.m_GTTechCounts, second.m_GTTechCounts))
            and (equal(first.m_GTTechRates, second.m_GTTechRates))
            and (equal(first.m_GTTechPrescaling, second.m_GTTechPrescaling))
            and (equal(first.m_GTPartition0TriggerCounts, second.m_GTPartition0TriggerCounts))
            and (equal(first.m_GTPartition0TriggerRates, second.m_GTPartition0TriggerRates))
            and (equal(first.m_GTPartition0DeadTime, second.m_GTPartition0DeadTime))
            and (equal(first.m_GTPartition0DeadTimeRatio, second.m_GTPartition0DeadTimeRatio))
        ;
    }
};

template <>
struct access<MixingInputConfig>
{
    static bool equal_(const MixingInputConfig & first, const MixingInputConfig & second)
    {
        return true
            and (equal(first.t_, second.t_))
            and (equal(first.an_, second.an_))
            and (equal(first.dpfv_, second.dpfv_))
            and (equal(first.dp_, second.dp_))
            and (equal(first.moot_, second.moot_))
            and (equal(first.ioot_, second.ioot_))
        ;
    }
};

template <>
struct access<MixingModuleConfig>
{
    static bool equal_(const MixingModuleConfig & first, const MixingModuleConfig & second)
    {
        return true
            and (equal(first.configs_, second.configs_))
            and (equal(first.minb_, second.minb_))
            and (equal(first.maxb_, second.maxb_))
            and (equal(first.bs_, second.bs_))
        ;
    }
};

template <>
struct access<RunInfo>
{
    static bool equal_(const RunInfo & first, const RunInfo & second)
    {
        return true
            and (equal(first.m_run, second.m_run))
            and (equal(first.m_start_time_ll, second.m_start_time_ll))
            and (equal(first.m_start_time_str, second.m_start_time_str))
            and (equal(first.m_stop_time_ll, second.m_stop_time_ll))
            and (equal(first.m_stop_time_str, second.m_stop_time_str))
            and (equal(first.m_fed_in, second.m_fed_in))
            and (equal(first.m_start_current, second.m_start_current))
            and (equal(first.m_stop_current, second.m_stop_current))
            and (equal(first.m_avg_current, second.m_avg_current))
            and (equal(first.m_max_current, second.m_max_current))
            and (equal(first.m_min_current, second.m_min_current))
            and (equal(first.m_run_intervall_micros, second.m_run_intervall_micros))
            and (equal(first.m_current, second.m_current))
            and (equal(first.m_times_of_currents, second.m_times_of_currents))
        ;
    }
};

template <>
struct access<RunSummary>
{
    static bool equal_(const RunSummary & first, const RunSummary & second)
    {
        return true
            and (equal(first.m_run, second.m_run))
            and (equal(first.m_name, second.m_name))
            and (equal(first.m_start_time_ll, second.m_start_time_ll))
            and (equal(first.m_start_time_str, second.m_start_time_str))
            and (equal(first.m_stop_time_ll, second.m_stop_time_ll))
            and (equal(first.m_stop_time_str, second.m_stop_time_str))
            and (equal(first.m_lumisections, second.m_lumisections))
            and (equal(first.m_subdt_in, second.m_subdt_in))
            and (equal(first.m_hltkey, second.m_hltkey))
            and (equal(first.m_nevents, second.m_nevents))
            and (equal(first.m_rate, second.m_rate))
        ;
    }
};

template <>
struct access<runinfo_test::RunNumber>
{
    static bool equal_(const runinfo_test::RunNumber & first, const runinfo_test::RunNumber & second)
    {
        return true
            and (equal(first.m_runnumber, second.m_runnumber))
        ;
    }
};

template <>
struct access<runinfo_test::RunNumber::Item>
{
    static bool equal_(const runinfo_test::RunNumber::Item & first, const runinfo_test::RunNumber::Item & second)
    {
        return true
            and (equal(first.m_run, second.m_run))
            and (equal(first.m_id_start, second.m_id_start))
            and (equal(first.m_id_stop, second.m_id_stop))
            and (equal(first.m_number, second.m_number))
            and (equal(first.m_name, second.m_name))
            and (equal(first.m_start_time_sll, second.m_start_time_sll))
            and (equal(first.m_start_time_str, second.m_start_time_str))
            and (equal(first.m_stop_time_sll, second.m_stop_time_sll))
            and (equal(first.m_stop_time_str, second.m_stop_time_str))
            and (equal(first.m_lumisections, second.m_lumisections))
            and (equal(first.m_subdt_joined, second.m_subdt_joined))
            and (equal(first.m_subdt_in, second.m_subdt_in))
        ;
    }
};

}
}

#endif
