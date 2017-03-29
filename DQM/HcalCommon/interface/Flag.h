#ifndef DQM_HcalCommon_Flag_h
#define DQM_HcalCommon_Flag_h

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

namespace hcaldqm
{
	namespace flag
	{
		//	
		//	State Definition. In the increasing order of worsiness.
		//	States(as flags) can be added
		//	s1 + s2 = max(s1,s2) - that allows to set the worse of the 2
		//
		enum State
		{
			fNONE=0,			// No State - can't have... not used....
			fNCDAQ=1,			// not @cDAQ
			fNA = 2,			// Not Applicable
			fGOOD = 3,			// GOOD
			fPROBLEMATIC = 4,	// problem
			fBAD = 5,			//	bad
			fRESERVED = 6,		// reserved
			nState = 7
		};

		struct Flag
		{
			Flag() :
				_name("SOMEFLAG"), _state(fNA)
			{}
			Flag(std::string const& name, State s=fNA):
				_name(name), _state(s)
			{}
			Flag(Flag const& f):
				_name(f._name), _state(f._state)
			{}

			//
			//	add 2 flags
			//
			Flag operator+(Flag const& f)
			{
				return Flag(_name!=f._name?"SOMEFLAG":_name,
					(State)(std::max(_state, f._state)));
			}

			//	
			//	add 2 flags and save
			//
			Flag& operator+=(Flag const& f)
			{
				_state = (State)(std::max(_state, f._state));
				return *this;
			}

			//	
			//	compare 2 flags
			//	
			bool operator==(Flag const& f)
			{
				return (_state==f._state && _name==f._name);
			}

			//	
			//	Assignment
			//
			Flag& operator=(Flag const& f)
			{
				_name = f._name;
				_state = f._state;
				return *this;
			}

			//	reset the state to NA
			void reset() {_state = fNA;}

			std::string _name;
			State _state;
		};
	}
}

#endif
