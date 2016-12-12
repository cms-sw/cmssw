#ifndef DQM_HcalTasks_TPRunSummary_h
#define DQM_HcalTasks_TPRunSummary_h

#include "DQM/HcalCommon/interface/DQClient.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

namespace hcaldqm
{
	class TPRunSummary : public DQClient
	{
		public:
			TPRunSummary(std::string const&, std::string const&,
				edm::ParameterSet const&);
			virtual ~TPRunSummary() {}

			virtual void beginRun(edm::Run const&, edm::EventSetup const&);
			virtual void endLuminosityBlock(DQMStore::IBooker&,
				DQMStore::IGetter&, edm::LuminosityBlock const&,
				edm::EventSetup const&);
			virtual std::vector<flag::Flag> endJob(
				DQMStore::IBooker&, DQMStore::IGetter&);

		protected:
			ContainerSingle2D _cEtMsmFraction_depthlike;
			ContainerSingle2D _cFGMsmFraction_depthlike;

			double _thresh_fgmsm, _thresh_etmsm;

			enum TPFlag
			{
				fEtMsm=0,
				fFGMsm=1,
				nTPFlag=2
			};
	};
}

#endif
