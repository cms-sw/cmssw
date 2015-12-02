#ifndef ContainerCompact_h
#define ContainerCompact_h

/*
 *	file:			ContainerCompact.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *		1D Compact Container
 */

#include "DQM/HcalCommon/interface/Container1D.h"

namespace hcaldqm
{
	using namespace axis;
	using namespace mapper;
	using namespace constants;
	
	struct Compact
	{
		double _sum;
		double _sum2;
		unsigned int _entries;
	};

	class ContainerCompact
	{
		public:
			ContainerCompact(){}
			virtual ~ContainerCompact() {}

			//	fills
			virtual void fill(HcalDetId const&, double);
			virtual void dump(Container1D*, bool);

		protected:
			Compact		_data[SUBDET_NUM][IPHI_NUM][IETA_NUM][DEPTH_NUM];
	};
}

#endif


