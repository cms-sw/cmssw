#ifndef DQM_HcalCommon_DQClient_h
#define DQM_HcalCommon_DQClient_h

/**
 *	file:
 *	Author:
 *	Description:
 */

#include "DQM/HcalCommon/interface/DQModule.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Flag.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/Utilities.h"

namespace hcaldqm
{
	class DQClient : public DQModule
	{
		public:
			DQClient(std::string const&,
				std::string const&, edm::ParameterSet const&);
			virtual ~DQClient() {}

			virtual void beginRun(edm::Run const&, edm::EventSetup const&);
			virtual void endLuminosityBlock(DQMStore::IBooker&,
				DQMStore::IGetter&,
				edm::LuminosityBlock const&, edm::EventSetup const&);
			virtual std::vector<flag::Flag> endJob(DQMStore::IBooker&, 
				DQMStore::IGetter&);

		protected:
			struct LSSummary
			{
				//	vector of flags per each FED
				std::vector<std::vector<flag::Flag> > _vflags;
				int _LS;
			};
			//	task name
			std::string _taskname;

			//	counters
			int _totalLS;
			int _maxProcessedLS;

			//	emap
			HcalElectronicsMap const* _emap;

			//	FED and FED Eids lists
			std::vector<int> _vFEDs;
			std::vector<uint32_t> _vhashFEDs;

			//	Container of Quality masks
			ContainerXXX<uint32_t> _xQuality;

			//	vector of FEDs registered at cDAQ
			std::vector<uint32_t> _vcdaqEids;
	};
}

#endif
