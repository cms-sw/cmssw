#ifndef FlagAxis_h
#define FlagAxis_h

/*
 *	file:			FlagAxis.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *		Flag Axis plots only flag variables on the axis.
 *		Have to provide the number of flags only (int)
 *
 *		The mapping between the actual flags and bins is straghtforward:
 *		flags(0 -> n-1) - axis range
 *		bins (1 -> n)
 *		bin = value+1
 *
 *		Important to note that there are no predefine flags. All the flags
 *		are defined in specific tasks and all the labels are loaded at the 
 *		construction of the Task
 */

#include "DQM/HcalCommon/interface/Axis.h"

namespace hcaldqm
{
	namespace axis
	{
		using namespace hcaldqm::constants;
		class FlagAxis : public Axis
		{
			public:
				friend class hcaldqm::Container;
				friend class hcaldqm::Container1D;
				friend class hcaldqm::Container2D;
				friend class hcaldqm::ContainerProf1D;
				friend class hcaldqm::ContainerProf2D;
				friend class hcaldqm::ContainerSingle1D;
				friend class hcaldqm::ContainerSingle2D;
				friend class hcaldqm::ContainerSingleProf1D;

			public:
				FlagAxis();
				FlagAxis(AxisType,  std::string, int);
				virtual ~FlagAxis() {}
				virtual FlagAxis* makeCopy()
				{return new FlagAxis(_type, _title, _nbins);}

				virtual inline int getBin(int v) {return v+1;}

				virtual void loadLabels(std::vector<std::string> const&);
				virtual void setBinAxisFlag(TObject *o)
				{
					o->SetBit(BIT(BIT_OFFSET+BIT_AXIS_FLAG));	
				}

			protected:
				virtual void _setup();
		};
	}
}

#endif
