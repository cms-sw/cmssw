#ifndef DQM_HcalTasks_RecoRunSummary_h
#define DQM_HcalTasks_RecoRunSummary_h

#include "DQM/HcalCommon/interface/DQClient.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

namespace hcaldqm
{
	class RecoRunSummary : public DQClient
	{
		public:
			RecoRunSummary(std::string const&, std::string const&,
				edm::ParameterSet const&);
			virtual ~RecoRunSummary() {}

			virtual void beginRun(edm::Run const&, edm::EventSetup const&);
			virtual void endLuminosityBlock(DQMStore::IBooker&,
				DQMStore::IGetter&, edm::LuminosityBlock const&,
				edm::EventSetup const&);
			virtual std::vector<flag::Flag> endJob(
				DQMStore::IBooker&, DQMStore::IGetter&);

		protected:
			double _thresh_unihf, _thresh_tcds;

			enum RecoFlag
			{
				fTCDS = 0,
				fUniSlotHF=1,
				nRecoFlag=2
			};
	};
}

#endif
