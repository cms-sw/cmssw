#ifndef popcon_popconDBOutputService_h
#define popcon_popconDBOutputService_h

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

namespace edm {
	class Event;
	class EventSetup;
	class ParameterSet;
}
namespace popcon{
	namespace service {
		class PopConDBOutputService : public cond::service::PoolDBOutputService{
			public:
				PopConDBOutputService( const edm::ParameterSet & iConfig, 
						edm::ActivityRegistry & iAR );

				std::string getTag();
				std::string getRecord();
			private:
				std::string m_tag, m_record;
		};
	}
}
#endif
