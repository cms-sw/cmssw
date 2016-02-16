
#include "DQM/HcalCommon/interface/Axis.h"

namespace hcaldqm
{
	namespace axis
	{
		Axis::Axis():
			_type(fXaxis), _qtype(fValue), _log(false)
		{
		}

		Axis::Axis(std::string title, AxisType type, AxisQType qtype,
			int n, double min, double max, bool log) :
			_nbins(n), _min(min), _max(max), _title(title), _type(type),
			_qtype(qtype), _log(log)
		{
		}
	}
}





