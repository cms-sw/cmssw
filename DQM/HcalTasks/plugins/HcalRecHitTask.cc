//	cmssw includes
#include "DQM/HcalTasks/interface/HcalRecHitTask.h"

//	system includes
#include <iostream>
#include <string>

HcalRecHitTask::HcalRecHitTask(edm::ParameterSet const&ps):
	hcaldqm::HcalDQSource(ps)
{
	for (unsigned int i=0; i<hcaldqm::constants::STD_NUMSUBS; i++)
		_numRecHits[i]=0;
}

/* virtual */ HcalRecHitTask::~HcalRecHitTask()
{
}

/* virtual */ void HcalRecHitTask::doWork(edm::Event const& e,
		edm::EventSetup const& es)
{
	edm::Handle<HBHERecHitCollection>	chbhe;
	edm::Handle<HORecHitCollection>		cho;
	edm::Handle<HFRecHitCollection>		chf;

	INITCOLL(_labels["HBHERecHit"], chbhe);
	INITCOLL(_labels["HORecHit"], cho);
	INITCOLL(_labels["HFRecHit"], chf);

	this->process(*chbhe, std::string("HB"));
	this->process(*chbhe, std::string("HE"));
	this->process(*cho, std::string("HO"));
	this->process(*chf, std::string("HF"));

	_mes["HB_RecHitOccupancy"].Fill(_numRecHits[0]);
	_mes["HE_RecHitOccupancy"].Fill(_numRecHits[1]);
	_mes["HO_RecHitOccupancy"].Fill(_numRecHits[2]);
	_mes["HF_RecHitOccupancy"].Fill(_numRecHits[3]);
	_mes["HB_RecHitOccupancyVSls"].Fill(_mi.currentLS, _numRecHits[0]);
	_mes["HE_RecHitOccupancyVSls"].Fill(_mi.currentLS, _numRecHits[1]);
	_mes["HO_RecHitOccupancyVSls"].Fill(_mi.currentLS, _numRecHits[2]);
	_mes["HF_RecHitOccupancyVSls"].Fill(_mi.currentLS, _numRecHits[3]);
}

//	reset
/* virtual */ void HcalRecHitTask::reset(int const periodflag)
{
	HcalDQSource::reset(periodflag);
	if (periodflag==0)
	{
		//	each event reset
		for (unsigned int i=0; i<hcaldqm::constants::STD_NUMSUBS; i++)
			_numRecHits[i]=0;
	}
	else if (periodflag==1)
	{
		//	each LS reset
	}
}

//	specializer
template<typename Hit>
void HcalRecHitTask::specialize(Hit const& hit, std::string const& nameRes,
		int const wtw)
{
	//	Obtain variables
	float en	= hit.energy();
	float time	= hit.time();
	int ieta	= hit.id().ieta();
	int iphi	= hit.id().iphi();
	int depth	= hit.id().depth();
	int subdet	= hit.id().subdet()-1;

	//	TODO:
	//	Put the Cut Class in place
	if (en<hcaldqm::constants::RECHIT_ZSCUT[subdet])
		return;

	//	Increment variables
	_numRecHits[subdet]++;

	//	Fill Plots
	_mes[nameRes+"_RecHitEnergy"].Fill(en);
	_mes[nameRes+"_RecHitTime"].Fill(time);
	_mes[nameRes+"_RecHitEnergyVSieta"].Fill(ieta, en);
	_mes[nameRes+"_RecHitTimeVSiphi"].Fill(iphi, time);
	_mes[nameRes+"_RecHitTimeVSieta"].Fill(ieta, time);
	_mes[nameRes+"_RecHitTimeVSenergy"].Fill(en, time);
	_mes[nameRes + "_RecHitEnergyVSls"].Fill(_mi.currentLS, en);
	if (subdet==hcaldqm::constants::STD_SUBDET_HO)
	{
		_mes["HOD4_EnergyMap"].Fill(ieta, iphi, en);
		_mes["HOD4_TimingMap"].Fill(ieta, iphi, time);
		_mes["HOD4_RecHitOccupancy"].Fill(ieta, iphi);
	}else if (subdet==hcaldqm::constants::STD_SUBDET_HB || 
			subdet==hcaldqm::constants::STD_SUBDET_HE ||
			subdet==hcaldqm::constants::STD_SUBDET_HF)
	{
		_mes["HBHEHFD" + 
			boost::lexical_cast<std::string>(depth) +
			"_EnergyMap"].Fill(ieta, iphi, en);
		_mes["HBHEHFD" + 
			boost::lexical_cast<std::string>(depth) +
			"_TimingMap"].Fill(ieta, iphi, time);
		_mes["HBHEHFD" + 
			boost::lexical_cast<std::string>(depth) + 
			"_RecHitOccupancy"].Fill(ieta, iphi);
	}

	if (subdet==hcaldqm::constants::STD_SUBDET_HB || 
		subdet==hcaldqm::constants::STD_SUBDET_HE)
	{
		if (iphi>=3 && iphi<=26)
			_mes["HBHE_RecHitTime_iphi3to26"].Fill(time);
		else if (iphi>=27 && iphi<=50)
			_mes["HBHE_RecHitTime_iphi27to50"].Fill(time);
		else
			_mes["HBHE_RecHitTime_iphi1to2_iphi51to72"].Fill(time);
	}

}

DEFINE_FWK_MODULE(HcalRecHitTask);



