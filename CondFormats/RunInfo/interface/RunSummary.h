#ifndef RunSummary_h
#define RunSummary_h

#include "CondFormats/Common/interface/Time.h"
#include <iostream>
#include <sstream>
#include <string>

/*
 *  \class RunSummary
 *  
 *  hosting light run information, above all the run start and stop time, the number of lumisections, and the average magnet currents.
 *
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN (Sep-24-2008)
 *  \author Salvatore di Guida (diguida) - CERN (Feb-19-2010)
 *
*/

class RunSummary {
  public:
	RunSummary();
	virtual ~RunSummary(){};
	cond::Time_t m_run;
	std::string m_sequenceName, m_globalConfKey;
	cond::Time_t m_start_time_ll, m_stop_time_ll;
	//cond::Time_t m_start_time_packed, m_stop_time_packed;
	std::string m_start_time_str, m_stop_time_str;
	int m_fill, m_energy;
	double m_lumisections;
	unsigned int m_HLTKey;
	std::string m_HLTKeyDesc;
	unsigned long long m_eventNumber, m_triggerNumber;
	float m_avgTriggerRate;
	float m_start_current;
	float m_stop_current;
	float m_avg_current;
	float m_max_current;
	float m_min_current;
	float m_run_intervall_micros;

	static RunSummary* Fake_RunSummary();
	virtual void printAllValues() const;
	virtual void print(std::stringstream& ss) const;
};

std::ostream& operator<< (std::ostream&, RunSummary runSummary);

#endif
