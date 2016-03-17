#include "DQM/HcalCommon/interface/FlagAxis.h"

namespace hcaldqm
{
	namespace axis
	{
		using namespace hcaldqm::constants;
		FlagAxis::FlagAxis():
			Axis()
		{}

		FlagAxis::FlagAxis(AxisType type, std::string name, int n):
			Axis(name, type, fFlag, n, 0, n, false)
		{}

		/* virtual */ void FlagAxis::_setup()
		{}

		/* virtual */ void FlagAxis::loadLabels(
			std::vector<std::string> const& labels)
		{
			_labels = labels;
		}
	}
}


