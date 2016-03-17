
#include "DQM/HcalCommon/interface/ValueAxis.h"

namespace hcaldqm
{
	using namespace constants;
	namespace axis
	{
		ValueAxis::ValueAxis():
			Axis(), _vtype(fEnergy)
		{}

		ValueAxis::ValueAxis(AxisType type, ValueType vtype, bool log):
			Axis(vtitle[vtype], type, fValue, vnbins[vtype],
				vmin[vtype], vmax[vtype], log), _vtype(vtype)
		{
			this->_setup();
		}

		ValueAxis::ValueAxis(AxisType type, ValueType vtype, int n,
			double min, double max, std::string title, bool log):
			Axis(title, type, fValue, n, min, max, log), _vtype(vtype)
		{
			this->_setup();
		}

		/* virtual */ int ValueAxis::getBin(int value)
		{
			//	only LS type right now
			int r=1;
			switch(_vtype)
			{
				case fLS:
					r = value;
					break;
				default:
					r = 1;
					break;
			}

			return r;
		}

		/* virtual */ int ValueAxis::getBin(double v)
		{
			return 1;
		}

		/* virtual */ void ValueAxis::_setup()
		{
		}
	}
}




