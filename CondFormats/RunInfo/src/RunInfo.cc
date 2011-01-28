#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondFormats/RunInfo/interface/RunInfoDBNames.h"
#include "CondFormats/RunInfo/interface/StringTokenize.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/TimeStamp.h"
#include <boost/foreach.hpp>

void RunInfo::RunSessionDelimiter::fill(std::string const & name, coral::AttributeList const & row) {
	if(name == "CMS.LVL0:RUNSECTION_DELIMITER_TYPE") {
		m_type = row[RunInfoDBNames::RunSessionDelimiterViewNames::ValueString()].data<std::string>();
	} else if (name == "CMS.LVL0:RUNSECTION_DELIMITER_DCSLHCFLAGS") {
		std::string const & unparsed = row[RunInfoDBNames::RunSessionDelimiterViewNames::ValueString()].data<std::string>();
		//parse the string for obtaining DCSFlags
		std::vector<std::string> helper;
		stringTokenize(unparsed, helper, '%');
		std::vector<std::string>::const_iterator itBegin = helper.begin();
		std::vector<std::string>::const_iterator itEnd = helper.end();
		std::vector<std::string> helper2;
		for(std::vector<std::string>::const_iterator it= itBegin;
			it != itEnd; ++it) {
			stringTokenize(*it, helper2, '&');
			DCSFlags.push_back(Triplet(helper2));
			helper2.clear();
		}
		helper.clear();
	} else if(name == "CMS.LVL0:RUNSECTION_DELIMITER_AUTOACTION") {
		m_auto = row[RunInfoDBNames::RunSessionDelimiterViewNames::ValueBool()].data<bool>();
	} else if(name == "CMS.LVL0:RUNSECTION_DELIMITER_TIME") {
		coral::TimeStamp const & timestamp = row[RunInfoDBNames::RunSessionDelimiterViewNames::ValueTime()].data<coral::TimeStamp>();
		unsigned long long m_time_packed = cond::time::from_boost(timestamp.time());
		cond::UnpackedTime unpacked = cond::time::unpack(m_time_packed);
		m_time = 1000000000 * unpacked.first + unpacked.second;
	} else if(name == "CMS.LVL0:RUNSECTION_DELIMITER_LS") {
		m_lumisection = row[RunInfoDBNames::RunSessionDelimiterViewNames::ValueDouble()].data<double>();
	} else if(name == "CMS.LVL0:RUNSECTION_DELIMITER_EVTNO") {
		m_eventNumber = row[RunInfoDBNames::RunSessionDelimiterViewNames::ValueInt()].data<unsigned long long>();
	} else if(name == "CMS.LVL0:RUNSECTION_DELIMITER_TRGNO") {
		m_triggerNumber = row[RunInfoDBNames::RunSessionDelimiterViewNames::ValueInt()].data<unsigned long long>();
	}
}

unsigned int const RunInfo::SUBDETECTOR_LIST_MAX;

RunInfo::Subdetector const RunInfo::subdetectorList[SUBDETECTOR_LIST_MAX] = {DAQ, DCS, PIXEL, TRACKER, ECAL, ES, HCAL, DT, CSC, RPC, CASTOR, DQM, COW};

RunInfo::Subdetector const RunInfo::subdetectorValues[] = {DAQ, DCS, PIXEL, TRACKER, ECAL, ES, HCAL, DT, CSC, RPC, CASTOR, DQM, COW};

std::string const RunInfo::subdetectorNames[] = {"DAQ", "DCS", "PIXEL", "TRACKER", "ECAL", "ES", "HCAL", "DT", "CSC", "RPC", "CASTOR", "DQM", "COW"};

RunInfo::SubdetectorSpecs const RunInfo::subdetectorSpecs[] = {
	SubdetectorTraits<DAQ>::specs()
	,SubdetectorTraits<DCS>::specs()
	,SubdetectorTraits<PIXEL>::specs()
	,SubdetectorTraits<TRACKER>::specs()
	,SubdetectorTraits<ECAL>::specs()
	,SubdetectorTraits<ES>::specs()
	,SubdetectorTraits<HCAL>::specs()
	,SubdetectorTraits<DT>::specs()
	,SubdetectorTraits<CSC>::specs()
	,SubdetectorTraits<RPC>::specs()
	,SubdetectorTraits<CASTOR>::specs()
	,SubdetectorTraits<DQM>::specs()
	,SubdetectorTraits<COW>::specs()
};

RunInfo::RunInfo():
	m_fills()
	,m_energies()
	,m_fed_in()
	,m_subdet_in()
	,m_triggerRates()
	//,m_current()
	//,m_times_of_currents()
	,m_timesAndCurrents()
	,m_sessions() {}

RunInfo::~RunInfo() {}

RunInfo * RunInfo::Fake_RunInfo() {
	RunInfo * sum = new RunInfo();
	return sum;
}

std::string const & RunInfo::subdetectorName(int const s) {
	return subdetectorNames[s];
}

RunInfo::SubdetectorSpecs const & RunInfo::findSpecs(std::string const & name) {
	for (unsigned int i = 0; i < SUBDETECTOR_LIST_MAX; ++i)
		if(name == subdetectorSpecs[i].name)
			return subdetectorSpecs[i];
	throw cms::Exception("invalid subdetector: " + name);
	return subdetectorSpecs[0]; //compiler happy
}

std::vector<std::string> RunInfo::getSubdetIn() const {
	std::vector<std::string> local;
	local.reserve(SUBDETECTOR_LIST_MAX);
	BOOST_FOREACH(int i, m_subdet_in) {
		local.push_back(subdetectorName(i));
	}
	return local;
}

void RunInfo::printAllValues() const {
	RunSummary::printAllValues();
	std::cout << "fill numbers in the run: ";
	std::copy(m_fills.begin(), m_fills.end(), std::ostream_iterator<int>(std::cout, ", "));
	std::cout << "\nbeam energies in the run: ";
	std::copy(m_energies.begin(), m_energies.end(), std::ostream_iterator<int>(std::cout, ", "));
	std::cout << "\nids of fed in run: ";
	std::copy(m_fed_in.begin(), m_fed_in.end(), std::ostream_iterator<int>(std::cout, ", "));
	std::cout << "\nsubdetectors in the run: ";
	std::copy(getSubdetIn().begin(), getSubdetIn().end(), std::ostream_iterator<std::string>(std::cout, ", "));
	/*std::cout << "\nB current in run: ";
	std::copy(m_current.begin(), m_current.end(), std::ostream_iterator<float>(std::cout, ", "));
	std::cout << "\ncorrespondent time (from run start) in nanoseconds for B currents in run: ";
	std::copy(m_times_of_currents.begin(), m_times_of_currents.end(), std::ostream_iterator<float>(std::cout, ", "));*/
	std::cout << std::endl;
}

void RunInfo::print(std::stringstream & ss) const {
	RunSummary::print(ss);
	std::copy(m_fills.begin(), m_fills.end(), std::ostream_iterator<int>(ss, ", "));
	ss << "\nbeam energies in the run: ";
	  std::copy(m_energies.begin(), m_energies.end(), std::ostream_iterator<int>(ss, ", "));
	ss << "\nids of FEDs in run: ";
	std::copy(m_fed_in.begin(), m_fed_in.end(), std::ostream_iterator<int>(ss, ", "));
	ss << "\nsubdetectors in the run: ";
	std::copy(getSubdetIn().begin(), getSubdetIn().end(), std::ostream_iterator<std::string>(ss, ", "));
	/*ss << "\nB current in run: ";
	std::copy(m_current.begin(), m_current.end(), std::ostream_iterator<float>(ss, ", "));
	ss << "\ncorrespondent time (from run start) in nanoseconds for B currents in run: ";
	std::copy(m_times_of_currents.begin(), m_times_of_currents.end(), std::ostream_iterator<float>(ss, ", "));*/
	ss << std::endl;
}

std::ostream& operator<< (std::ostream& os, RunInfo runInfo) {
	std::stringstream ss;
	runInfo.print(ss);
	os << ss.str();
	return os;
}
