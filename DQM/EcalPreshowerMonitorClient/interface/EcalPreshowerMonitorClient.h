#ifndef EcalPreshowerMonitorClient_H
#define EcalPreshowerMonitorClient_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESIntegrityClient.h"


class DQMOldReceiver;
class DQMStore;

class EcalPreshowerMonitorClient : public edm::EDAnalyzer{
	public:
		EcalPreshowerMonitorClient(const edm::ParameterSet& ps);
		virtual ~EcalPreshowerMonitorClient();


	private:
		virtual void analyze(const edm::Event &, const edm::EventSetup &);
		virtual void analyze();

		virtual void beginJob(const edm::EventSetup & c) ;
		virtual void endJob() ;
		virtual void beginRun() ;
		virtual void endRun() ;


		void htmlOutput(int);

		// ----------member data ---------------------------

		std::string outputFile_;
		std::string inputFile_;
		std::string prefixME_;

		bool enableMonitorDaemon_;

		std::string clientName_;
		std::string hostName_;
		int hostPort_;

		DQMOldReceiver* mui_;
		DQMStore* dqmStore_;

		bool begin_run_;
		bool end_run_;
		bool debug_;

		int prescaleFactor_;		
                int EvtperJob_;
                int EvtperRun_;

		ESPedestalClient* PedestalClient_;
		ESIntegrityClient* IntegrityClient_;

		int nLines_, runNum_;
		int runtype_, seqtype_, dac_, gain_, precision_;
		int firstDAC_, nDAC_, isPed_, vDAC_[5], layer_;

		int senZ_[4288], senP_[4288], senX_[4288], senY_[4288];
  		int qt[40][40], qtCriteria;
		

};

#endif
