#ifndef DQM_HcalTasks_PedestalRunSummary_h
#define DQM_HcalTasks_PedestalRunSummary_h

#include "DQM/HcalCommon/interface/DQClient.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

namespace hcaldqm
{
	class PedestalRunSummary : public DQClient
	{
		public:
			PedestalRunSummary(std::string const&, std::string const&,
				edm::ParameterSet const&);
			virtual ~PedestalRunSummary() {}

			virtual void beginRun(edm::Run const&, edm::EventSetup const&);
			virtual void endLuminosityBlock(DQMStore::IBooker&,
				DQMStore::IGetter&, edm::LuminosityBlock const&,
				edm::EventSetup const&);
			virtual std::vector<flag::Flag> endJob(
				DQMStore::IBooker&, DQMStore::IGetter&);

		protected:
	};
}

#endif
