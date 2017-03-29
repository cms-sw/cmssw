#ifndef DQM_HcalTasks_RawRunSummary_h
#define DQM_HcalTasks_RawRunSummary_h

#include "DQM/HcalCommon/interface/DQClient.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

namespace hcaldqm
{
	class RawRunSummary : public DQClient
	{
		public:
			RawRunSummary(std::string const&, std::string const&,
				edm::ParameterSet const&);
			virtual ~RawRunSummary() {}

			virtual void beginRun(edm::Run const&, edm::EventSetup const&);
			virtual void endLuminosityBlock(DQMStore::IBooker&,
				DQMStore::IGetter&, edm::LuminosityBlock const&,
				edm::EventSetup const&);
			virtual std::vector<flag::Flag> endJob(
				DQMStore::IBooker&, DQMStore::IGetter&);

		protected:
			//	Flag Summary for each LS and Run as a whole
			std::vector<LSSummary> _vflagsLS; 

			electronicsmap::ElectronicsMap _ehashmap;

			//	some useful vectors - not to reintialize all the time
			std::vector<uint32_t> _vhashVME, _vhashuTCA;
			std::vector<int> _vFEDsVME, _vFEDsuTCA;
			filter::HashFilter _filter_VME, _filter_uTCA;

			//	Containers to store info for the whole run
			Container2D _cEvnMsm_ElectronicsVME, _cEvnMsm_ElectronicsuTCA;
			Container2D _cBcnMsm_ElectronicsVME, _cBcnMsm_ElectronicsuTCA;
			Container2D _cBadQuality_depth;

			bool _booked;

			//	
			ContainerXXX<uint32_t> _xEvn, _xBcn, _xBadQ;

			enum RawLSFlag
			{
				fEvnMsm=0,
				fBcnMsm=1,
				fBadQ=2,
				nRawFlag=3
			};
	};
}

#endif
