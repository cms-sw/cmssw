#ifndef TEMPLATE_PAYLOAD_H
#define TEMPLATE_PAYLOAD_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>
#include "IOVPair.h"
#include "OfflineDBInterface.h"
#include "FWCore/Framework/interface/Event.h"

namespace popcon
{
	//Online DB source handler, aims at returning the vector of data to be 
	//transferred to the online database
	//Subdetector developers inherit over this class with template parameter of 
	//payload class; need to implement the getNewObjects method and overload the 
	//constructor; 
	
	template <class T>
		class PopConSourceHandler
		{
			public: 
				PopConSourceHandler(std::string name, std::string connect_string, std::string catalog, const edm::Event& evt, const edm::EventSetup& est) : myname(name), event(evt), esetup(est)
				{
					m_db_iface = new popcon::OfflineDBInterface(connect_string,catalog);
					m_to_transfer = new std::vector<std::pair<T*,popcon::IOVPair> >;

				}
				virtual ~PopConSourceHandler()
				{
					delete m_to_transfer;
					delete m_db_iface;
				}
				
				unsigned int getSinceForTag(std::string tag)
				{
					return (m_db_iface->getSpecificTagInfo(tag)).last_since;
				}
				
				std::vector<std::pair<T*, popcon::IOVPair> >* returnData()
				{
					this->getNewObjects();
					std::cout << "Source Handler returns data\n";
					return this->m_to_transfer;
				}
		
				//Implement to fill m_to_transfer vector
				//use getOfflineInfo to get the contents of offline DB
				virtual void getNewObjects()=0;

				//in case if there's a need to access the information passed as analyze 
				//arguments, this method is called by analyzer's analyze
				
			private:
				std::string myname;
				//Offline Database Interface object
				popcon::OfflineDBInterface* m_db_iface;

			protected:
				const edm::Event& event;
				const edm::EventSetup& esetup;
				
				//Is is sufficient for getNewObjects algorithm?
 				std::map<std::string, PayloadIOV> getOfflineInfo()
				{
					return m_db_iface->getStatusMap();
				}

				//vector of payload objects and iovinfo to be transferred
				//class looses ownership of payload object
				std::vector<std::pair<T*, popcon::IOVPair> >* m_to_transfer;

				
				
				
		};
}
#endif
