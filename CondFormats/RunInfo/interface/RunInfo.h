#ifndef RunInfo_h
#define RunInfo_h

#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CoralBase/AttributeList.h"
#include <utility>
#include <vector>


/*
 *  \class RunInfo
 *  
 *  hosting run information, above all the run start and stop time, the list of fed joining, the .  
 *
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN (Oct-10-2008)
 *  \author Salvatore di Guida (diguida) - CERN (Feb-19-2010)
 *
*/

class RunInfo: public RunSummary {
 public:
	//internal structures
	struct Triplet {
		Triplet() {};
		explicit Triplet(std::vector<std::string> const & tokens) {
			if(tokens.size() >= 3) {
				m_flag = tokens.at(0);
				m_used = tokens.at(1);
				m_fromDCS = tokens.at(2);
			} else {
				std::cout << "Wrong vector" << std::endl;
			}
		}
		virtual ~Triplet() {};
		std::string m_flag, m_used, m_fromDCS;
		//int m_used, m_fromDCS;
	};

	struct RunSessionDelimiter {
		RunSessionDelimiter():
			m_type("null")
			,m_auto(0)
			,m_time(0)
			//,m_time_packed(0)
			,m_lumisection(0)
			,m_eventNumber(0)
			,m_triggerNumber(0)
			,DCSFlags() {};
		RunSessionDelimiter(RunSessionDelimiter const & rhs):
			m_type(rhs.m_type)
			,m_auto(rhs.m_auto)
			,m_time(rhs.m_time)
			//,m_time_packed(rhs.m_time_packed)
			,m_lumisection(rhs.m_lumisection)
			,m_eventNumber(rhs.m_eventNumber)
			,m_triggerNumber(rhs.m_triggerNumber)
			,DCSFlags(rhs.DCSFlags) {}
		RunSessionDelimiter & operator=(RunSessionDelimiter const & rhs) {
			m_type = rhs.m_type;
			m_auto = rhs.m_auto;
			m_time = rhs.m_time;
			//m_time_packed = rhs.m_time_packed;
			m_lumisection = rhs.m_lumisection;
			m_eventNumber = rhs.m_eventNumber;
			m_triggerNumber = rhs.m_triggerNumber;
			DCSFlags = rhs.DCSFlags;
			return *this;
		}
		virtual ~RunSessionDelimiter() {};
		std::string m_type;
		bool m_auto;
		cond::Time_t m_time;
		//cond::Time_t m_time_packed;
		double m_lumisection;
		unsigned long long m_eventNumber, m_triggerNumber;
		std::vector<Triplet> DCSFlags;
		void fill(std::string const & name, coral::AttributeList const & row);
	};

	struct RunSession {
		RunSession():
			m_autoBegin(0)
			,m_autoEnd(0)
			,m_beginTime(0)
			,m_endTime(0)
			,m_lumisection(0)
			,m_eventNumber(0)
			,m_triggerNumber(0)
			,DCSFlags() {};
		virtual ~RunSession() {};
		bool m_autoBegin, m_autoEnd;
		cond::Time_t m_beginTime, m_endTime;
		//cond::Time_t m_beginTime_packed, m_endTime_packed;
		double m_lumisection;
		unsigned long long m_eventNumber, m_triggerNumber;
		std::vector<Triplet> DCSFlags;
	};

	enum Subdetector { DAQ=0, DCS, PIXEL, TRACKER, ECAL, ES, HCAL, DT, CSC, RPC, CASTOR, DQM, COW };

	static unsigned int const SUBDETECTOR_LIST_MAX = 13;
	static Subdetector const subdetectorList[SUBDETECTOR_LIST_MAX];
	static Subdetector const subdetectorValues[];
	static std::string const subdetectorNames[];

	struct SubdetectorSpecs {
		SubdetectorSpecs() {};
		explicit SubdetectorSpecs(Subdetector const & subdet, std::string const & subName):
			sub(subdet)
			,name(subName) {};
		~SubdetectorSpecs() {};
		// the enum
		Subdetector sub;
		//the name
		std::string name;
	};

	template<Subdetector sub>
	struct SubdetectorTraits {
		static SubdetectorSpecs const & specs() {
			static SubdetectorSpecs const local(sub, subdetectorName(sub));
			//static SubdetectorSpecs const local = {sub, subdetectorName(sub)};
			return local;
		}
	};

	static SubdetectorSpecs const subdetectorSpecs[];

	RunInfo();
	virtual ~RunInfo();
	std::vector<int> m_fills, m_energies;
	std::vector<int> m_fed_in;
	std::vector<int> m_subdet_in;
	std::vector<float> m_triggerRates;
	std::vector<std::pair<cond::Time_t, float> > m_timesAndCurrents;
	//std::vector<float> m_current;
	//std::vector<float> m_times_of_currents;
	std::vector<RunSession> m_sessions;
	static RunInfo* Fake_RunInfo();

	static std::string const & subdetectorName(int const s);
	// find spec by name
	static SubdetectorSpecs const & findSpecs(std::string const & name);
	std::vector<std::string> getSubdetIn() const;
	void printAllValues() const;
	void print(std::stringstream& ss) const;
};

std::ostream& operator<< (std::ostream&, RunInfo runInfo);

#endif
