#ifndef DQM_HcalTasks_NoCQTask_h
#define DQM_HcalTasks_NoCQTask_h

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/Container1D.h"
 #include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"

using namespace hcaldqm;
using namespace hcaldqm::filter;
class NoCQTask : public DQTask
{
	public:
		NoCQTask(edm::ParameterSet const&);
		virtual ~NoCQTask() {}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(UpdateFreq);

		edm::InputTag _tagHBHE;
		edm::InputTag _tagHO;
		edm::InputTag _tagHF;
		edm::InputTag _tagReport;
		edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE;
		edm::EDGetTokenT<HODigiCollection> _tokHO;
		edm::EDGetTokenT<HFDigiCollection> _tokHF;
		edm::EDGetTokenT<HcalUnpackerReport> _tokReport;

		double _cutSumQ_HBHE, _cutSumQ_HO, _cutSumQ_HF;

		HcalElectronicsMap const* _emap;
		electronicsmap::ElectronicsMap _ehashmap;

		ContainerProf2D _cTimingCut_depth;
		Container2D _cOccupancy_depth;
		Container2D _cOccupancyCut_depth;
		Container2D _cBadQuality_depth;
};

#endif
