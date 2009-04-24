#ifndef ESPedestalClient_H
#define ESPedestalClient_H


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"

#include <TF1.h>


//
// class decleration
//


class MonitorElement;
class DQMStore;

class ESPedestalClient : public ESClient{
	public:
		ESPedestalClient(const edm::ParameterSet& ps);
		virtual ~ESPedestalClient();
		void analyze(void);
		void beginJob(DQMStore* dqmStore);
		void endJob(void);
		void beginRun(void);
		void endRun(void);
		void setup(void);
		void cleanup(void);
	
//		inline int getEvtPerJob() { return EvtperJob_; }
//		inline int getEvtPerRun() { return EvtperRun_; }


	private:

		int EvtperJob_;
		int EvtperRun_;
		bool enableCleanup_;

		edm::FileInPath lookup_;
		std::string prefixME_;

		DQMStore* dqmStore_;

		MonitorElement *hPed_[2][2][40][40];
		MonitorElement *hTotN_[2][2][40][40];

		TF1 *fg;

		int nLines_;
//		int runNum_, runtype_, seqtype_, dac_, gain_, precision_;
//		int firstDAC_, nDAC_, isPed_, vDAC_[5], layer_;

		int senZ_[4288], senP_[4288], senX_[4288], senY_[4288];

};

#endif  //ESPedestalClient_H
