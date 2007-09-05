#ifndef PROTO_O2O_ANALYZER_H
#define PROTO_O2O_ANALYZER_H
//
// Original Author:  Marcin BOGUSZ
//         Created:  Tue Jul  3 10:48:22 CEST 2007

// system include files
#include <memory>

#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/PopCon/interface/OutputServiceWrapper.h"


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/PopCon/interface/StateCreator.h"
//#include "CondCore/PopCon/interface/Logger.h"

//
// class decleration
//

namespace popcon
{

	template <typename T>
		class PopConAnalyzer : public edm::EDAnalyzer {
			public:

				//One needs to inherit this class and implement the constructor to 
				// instantiate handler object
				explicit PopConAnalyzer(const edm::ParameterSet& pset, std::string objetct_name):tryToValidate(false),corrupted(true), greenLight (false), fixed(true), m_payload_name(objetct_name) 

				{
					//TODO set the policy (cfg or global configuration?)
					//Policy if corrupted data found
					m_debug = pset.getParameter< bool > ("debug");
					m_popcon_db = pset.getParameter<std::string> ("PopConDBSchema");
					//MANDATORY 
					m_offline_connection = pset.getParameter<std::string> ("OfflineDBSchema");
					m_catalog = pset.getParameter<std::string> ("catalog");
					sinceAppend = pset.getParameter<bool> ("SinceAppendMode");
					m_handler_object = 0;
				}
				~PopConAnalyzer()
				{
					//moved the code from endJob, as DBOutput service doesn't commit after the end of analyze !!!!!
					if(m_debug)
						std::cerr << "Destructor begins\n";	

					try{
						if (!fixed || corrupted)
						{	
							if(m_debug)
								std::cerr << "Corrupted | unfixed state | problem with PopCon DB\n";
							lgr->finalizeExecution(logMsg);
						}
						else //ok
						{
							if(m_debug)
								std::cerr << "OK, finalizing the log\n";
							lgr->finalizeExecution(logMsg);
							stc->generateStatusData();
							stc->storeStatusData();
							if(m_debug)
								std::cerr << "Deleting stc\n";	
							delete stc;
						}

						lgr->unlock();

					}
					catch(std::exception& e)
					{
						std::cerr << "Exception caught in destructor: "<< e.what();
					}

					if (m_handler_object != 0){
						if(m_debug)
							std::cerr << "Deleting the source handler\n";	
						delete m_handler_object;
					}
					if(m_debug)
						std::cerr << "Deleting lgr\n";	
					delete lgr;

					if(m_debug)
						std::cerr << "Destructor ends\n";	
				}
			private:
				//state management
				StateCreator* stc;
				//logs the runs 
				Logger* lgr;

				std::string m_popcon_db;
				//If state corruption is detected, this parameter specifies the program behaviour
				bool tryToValidate;
				//corrupted data detected, just write the log and exit
				bool corrupted;
				bool greenLight;
				//Someone claims to have fixed the problem indicated in exception section
				//TODO log it as well
				bool fixed;
				bool sinceAppend;
				std::string logMsg;

				virtual void beginJob(const edm::EventSetup& es)
				{	
					if(m_debug)
						std::cerr << "Begin Job\n"; 
					try{
						lgr = new Logger(m_popcon_db,m_payload_name,m_debug);
						//lock the run (other instances of analyzer of the same typename will be blocked till the end of execution)
						lgr->lock();
						//log the new app execution
						lgr->newExecution();

						stc = new StateCreator(m_popcon_db, m_offline_connection, m_payload_name, m_debug);

						//checks the exceptions, validates new data if necessary
						if (stc->previousExceptions(fixed))
						{
							std::cerr << "There's been a problem with a previous run" << std::endl;
							if (!fixed)
							{	
								//TODO - set the flag
								logMsg="Running with unfixed state, EXITING";
								return;
							}
							else
							{
								std::cerr << "Handled exception, attempting to validate" << std::endl;
								//TODO - implement ?
							}
						}

						if (stc->checkAndCompareState())
						{
							//std::cerr << "State OK" << std::endl;
							greenLight = true;
							logMsg="OK";
							corrupted = false;
							//	safe to run;
						}
						else
						{
							//	data is corrupted;
							if (tryToValidate)
							{
								std::cerr << "attempting to validate" << std::endl;
								corrupted = false;
							}
							else 
							{
								//Report an error 	
								std::cerr << "State Corruption with no validation flag set, EXITING!!!\n";
								logMsg="State corruption";
								return;
							}
						}
					}
					catch(popcon::Exception & er)
					{
						std::cerr << "Begin Job PopCon exception\n";
						greenLight = false;
					}
					catch(std::exception& e)
					{
						greenLight = false;
						std::cerr << "Begin Job std-based exception\n";
					}

				}

				//this method handles the transformation algorithm, 
				//Subdetector responsibility ends with returning the payload vector.
				//Therefore this code is stripped of DBOutput service, state management etc.	
				virtual void analyze(const edm::Event& evt, const edm::EventSetup& est)
				{
					if(m_debug)
						std::cerr << "Analyze Begins\n"; 
					try{
						if (greenLight)
						{
							//create source handling object, pass the eventsetup reference
							initSource(evt,est);
							//get New objects 
							takeTheData();
							if(m_debug)
								displayHelper();
							if (!m_payload_vect->empty()){
								edm::Service<popcon::service::PopConDBOutputService> poolDbService;
								OutputServiceWrapper<T>* wrpr = new OutputServiceWrapper<T>(poolDbService);
								unsigned int last_since = m_handler_object->getSinceForTag(poolDbService->getTag());
								//TODO check till as well
								wrpr->write(m_payload_vect,lgr,logMsg,last_since,sinceAppend);
								delete wrpr;
							}
							else 
								std::cout << "PopConDBOutputService - nothing to write \n"; 
						}
						if(m_debug)
							std::cerr << "Analyze Ends\n"; 
					}
					catch(std::exception& e)
					{
						std::cerr << "Analyzer Exception\n";
						std::cerr << e.what() << std::endl; 
					}

				}

				virtual void endJob()
				{	
				}

				//initialize the source handler
				virtual void initSource(const edm::Event& evt, const edm::EventSetup& est)=0;


				//This class takes ownership of the vector (and payload objects)
				void takeTheData()
				{
					m_payload_vect = m_handler_object->returnData();	
				}

				std::string m_payload_name;
				std::vector <std::pair<T*,popcon::IOVPair> >* m_payload_vect;
			protected:

				bool m_debug;
				//source handling object
				PopConSourceHandler<T>* m_handler_object;	
				//connect string for offline cond db access;
				std::string m_offline_connection;
				std::string m_catalog;

				void displayHelper()
				{
					typename std::vector<std::pair<T*,popcon::IOVPair> >::iterator it;
					for (it = m_payload_vect->begin(); it != m_payload_vect->end(); it++)
					{
						std::cerr<<"Since " <<(*it).second.since << " till " << (*it).second.till << std::endl;
					}
				}


		};
}
#endif
