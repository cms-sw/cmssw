	
#include "DQM/HcalTasks/interface/RadDamTask.h"

using namespace hcaldqm;
RadDamTask::RadDamTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	//	List all the DetIds
	_vDetIds.push_back(HcalDetId(HcalForward, -30, 35, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -30, 71, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -32, 15, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -32, 51, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -34, 35, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -34, 71, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -36, 15, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -36, 51, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -38, 35, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -38, 71, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -40, 15, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -40, 51, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -41, 35, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -41, 71, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, -30, 15, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -30, 51, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -32, 35, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -32, 71, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -34, 15, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -34, 51, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -36, 35, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -36, 71, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -38, 15, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -38, 51, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -40, 35, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -40, 71, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -41, 15, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, -41, 51, 2));

	_vDetIds.push_back(HcalDetId(HcalForward, 30, 21, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 30, 57, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 32, 1, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 32, 37, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 34, 21, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 34, 57, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 36, 1, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 36, 37, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 38, 21, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 38, 57, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 40, 35, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 40, 71, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 41, 19, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 41, 55, 1));
	_vDetIds.push_back(HcalDetId(HcalForward, 30, 1, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 30, 37, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 32, 21, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 32, 57, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 34, 1, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 34, 37, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 36, 21, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 36, 57, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 38, 1, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 38, 37, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 40, 19, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 40, 55, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 41, 35, 2));
	_vDetIds.push_back(HcalDetId(HcalForward, 41, 71, 2));

	//	tags
	_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF", 
		edm::InputTag("hcalDigis"));
	_tokHF = consumes<HFDigiCollection>(_tagHF);
}

/* virtual */ void RadDamTask::bookHistograms(DQMStore::IBooker& ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	//	Initialize all the Single Containers
	for (std::vector<HcalDetId>::const_iterator it=_vDetIds.begin();
		it!=_vDetIds.end(); ++it)
	{
		_vcShape.push_back(ContainerSingle1D(_name, 
			"Shape",
			new quantity::ValueQuantity(quantity::fTiming_TS),
			new quantity::ValueQuantity(quantity::ffC_3000)));
	}

	DQTask::bookHistograms(ib, r, es);	
	char aux[200];
	for (unsigned int i=0; i<_vDetIds.size(); i++)
	{
		sprintf(aux, "ieta%diphi%dd%d", _vDetIds[i].ieta(), 
			_vDetIds[i].iphi(), _vDetIds[i].depth());
		_vcShape[i].book(ib, _subsystem, aux);
	}
}

/* virtual */ void RadDamTask::_process(edm::Event const &e,
	edm::EventSetup const& es)
{
	edm::Handle<HFDigiCollection> chf;
	if (!e.getByToken(_tokHF, chf))
		_logger.dqmthrow("Collection HFDigiCollection isn't avalaible"
			+ _tagHF.label() + " " + _tagHF.instance());

	for (HFDigiCollection::const_iterator it=chf->begin(); 
		it!=chf->end(); ++it)
	{
		const HFDataFrame digi = (const HFDataFrame)(*it);
		for (unsigned int i=0; i<_vDetIds.size(); i++)
			if (digi.id()==_vDetIds[i])
			{
				for (int j=0; j<digi.size(); j++)
					_vcShape[i].fill(j, 
						digi.sample(j).nominal_fC()-2.5);
			}
	}
}

/* virtual */ bool RadDamTask::_isApplicable(edm::Event const &e)
{
	if (_ptype==fOnline)
	{
		// globla-online
		int calibType = this->_getCalibType(e);
		return (calibType==hc_RADDAM);
	}
	else
	{
		//	local, just return true as all the settings will be done in cfg
		return true;
	}

	return false;
}

DEFINE_FWK_MODULE(RadDamTask);



