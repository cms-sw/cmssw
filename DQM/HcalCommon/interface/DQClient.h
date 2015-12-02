#ifndef DQClient_h
#define DQClient_h

/*
 *	file:			DQClient.h
 *	Author:			Viktor Khristenko
 *
 */

#include "DQM/HcalCommon/interface/DQModule.h"

namespace hcaldqm
{
	class DQClient : public DQModule
	{
		public:
			DQClient(edm::ParameterSet const&);
			virtual ~DQClient() {}

		protected:
	};
}

#endif










