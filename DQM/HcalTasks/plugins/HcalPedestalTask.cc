//	cmssw includes
#include "DQM/HcalTasks/interface/HcalPedestalTask.h"

//	system includes
#include <iostream>
#include <string>

HcalPedestalTask::HcalPedestalTask(edm::ParameterSet const&ps):
	hcaldqm::HcalDQSource(ps)
{	
	_packager[0] = hcaldqm::packaging::Packager(
			hcaldqm::constants::STD_HB_MINIPHI,
			hcaldqm::constants::STD_HB_MAXIPHI,
			hcaldqm::constants::STD_HB_STEPIPHI,
			hcaldqm::constants::STD_HB_MINIETA,
			hcaldqm::constants::STD_HB_MAXIETA,
			hcaldqm::constants::STD_HB_MINDEPTH,
			hcaldqm::constants::STD_HB_MAXDEPTH
	);
	_packager[1] = hcaldqm::packaging::Packager(
			hcaldqm::constants::STD_HE_MINIPHI,
			hcaldqm::constants::STD_HE_MAXIPHI,
			hcaldqm::constants::STD_HE_STEPIPHI,
			hcaldqm::constants::STD_HE_MINIETA,
			hcaldqm::constants::STD_HE_MAXIETA,
			hcaldqm::constants::STD_HE_MINDEPTH,
			hcaldqm::constants::STD_HE_MAXDEPTH
	);
	_packager[2] = hcaldqm::packaging::Packager(
			hcaldqm::constants::STD_HO_MINIPHI,
			hcaldqm::constants::STD_HO_MAXIPHI,
			hcaldqm::constants::STD_HO_STEPIPHI,
			hcaldqm::constants::STD_HO_MINIETA,
			hcaldqm::constants::STD_HO_MAXIETA,
			hcaldqm::constants::STD_HO_MINDEPTH,
			hcaldqm::constants::STD_HO_MAXDEPTH
	);
	_packager[3] = hcaldqm::packaging::Packager(
			hcaldqm::constants::STD_HF_MINIPHI,
			hcaldqm::constants::STD_HF_MAXIPHI,
			hcaldqm::constants::STD_HF_STEPIPHI,
			hcaldqm::constants::STD_HF_MINIETA,
			hcaldqm::constants::STD_HF_MAXIETA,
			hcaldqm::constants::STD_HF_MINDEPTH,
			hcaldqm::constants::STD_HF_MAXDEPTH
	);
}

/* virtual */ HcalPedestalTask::~HcalPedestalTask()
{}

/* virtual */ void HcalPedestalTask::endRun(const edm::Run& r,
		edm::EventSetup const& es)
{
	this->publish();
}

void HcalPedestalTask::publish()
{
	std::cout << "publishing" << std::endl;
//	_mes["HBHEHFD1_PedestalsMap_Summary"].Reset();
//	_mes["HBHEHFD2_PedestalsMap_Summary"].Reset();
//	_mes["HBHEHFD3_PedestalsMap_Summary"].Reset();
//	_mes["HOD4_PedestalsMap_Summary"].Reset();

	for (int i=0; i<hcaldqm::constants::STD_NUMSUBS; i++)
	{
		_mes[hcaldqm::constants::SUBNAMES[i] + 
			"_PedMeans_Summary"].Reset();
		_mes[hcaldqm::constants::SUBNAMES[i] + 
			"_PedRMSs_Summary"].Reset();
		for (int iieta=0; iieta<hcaldqm::constants::STD_NUMIETAS; iieta++)
			for (int iiphi=0; iiphi<hcaldqm::constants::STD_NUMIPHIS; iiphi++)
				for (int id=0; id<hcaldqm::constants::STD_NUMDEPTHS; id++)
				{
					double sumpedmeans = 0; double sumpedrmss = 0;
					int numcaps = 0;
					for (int ic=0; ic<hcaldqm::constants::STD_NUMCAPS; ic++)
					{
						std::pair<double, double> meanrms = 
							_pedData[i][iieta][iiphi][id][ic].average();
						double mean = meanrms.first;
						double rms = meanrms.second;
						if (mean==-1 || rms==-1)
							continue;

						sumpedmeans += mean; sumpedrmss += rms;
						numcaps++;
					}

					if (numcaps>0)
					{
						double pedmean = 
							sumpedmeans/numcaps;
						double pedrms = 
							sumpedrmss/numcaps;
						_mes[hcaldqm::constants::SUBNAMES[i] + 
							"_PedMeans_Summary"].Fill(pedmean);
						_mes[hcaldqm::constants::SUBNAMES[i] + 
							"_PedRMSs_Summary"].Fill(pedrms);
/*						if (hcaldqm::constants::SUBNAMES[i]=="HO")
							_mes["HOD4_PedestalsMap_Summary"].Fill(
								_packager[i].ieta(iieta),
								_packager[i].iphi(iiphi));
						else
							_mes["HBHEHFD" + boost::lexical_cast<std::string>(
								_packager[i].depth(id)) + 
								"_PedestalsMap_Summary"].Fill(
								_packager[i].ieta(iieta),
								_packager[i].iphi(iiphi));
								*/
					}
				}
	}
}

/* virtual */ void HcalPedestalTask::beginLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::beginLuminosityBlock(lb, es);
}

/* virtual */ void HcalPedestalTask::endLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::endLuminosityBlock(lb, es);
}

/* virtual */ void HcalPedestalTask::reset(int const periodflag)
{
	HcalDQSource::reset(periodflag);
	if (periodflag==0)
	{
		// for Event
	}
	else if (periodflag==1)
	{
		//	 for LS
	}
}

/* virtual */ void HcalPedestalTask::doWork(edm::Event const& e,
		edm::EventSetup const& es)
{
	edm::Handle<HBHEDigiCollection>			chbhe;
	edm::Handle<HODigiCollection>			cho;
	edm::Handle<HFDigiCollection>			chf;

	INITCOLL(_labels["HBHEDigi"], chbhe);
	INITCOLL(_labels["HODigi"], cho);
	INITCOLL(_labels["HFDigi"], chf);

	this->process(*chbhe, std::string("HB"));
	this->process(*chbhe, std::string("HE"));
	this->process(*cho, std::string("HO"));
	this->process(*chf, std::string("HF"));

	//	For Online-Only Calib Gap events
	if (_mi.isGlobal &&
		_mi.evsTotal>0 && 
		_mi.evsTotal%hcaldqm::constants::PUBLISH_MIN_CALIBEVENTS==0)
		this->publish();
}

template<typename Hit>
void HcalPedestalTask::specialize(Hit const& hit, std::string const& nameRes,
		int const wtw)
{
	int iphi = hit.id().iphi();
	int ieta = hit.id().ieta();
	int depth = hit.id().depth();
	int subdet = hit.id().subdet()-1;
	int digisize = hit.size();
	//	if digisize is 4(global runs as of 01/06/2015), you get 4
	//	if digisize is 10(local runs as of 01/06/2015), you get 8
	int digisizeToUse = floor(digisize/hcaldqm::constants::STD_NUMCAPS)*
		hcaldqm::constants::STD_NUMCAPS;

	//	Fills up Prioprietary Class for Pedestals Monitoring
	for (int i=0; i<hit.size(); i++)
		_pedData[subdet][_packager[subdet].iieta(ieta)]
			[_packager[subdet].iiphi(iphi)]
			[_packager[subdet].idepth(depth)]
			[hit.sample(i).capid()].push(
					hit.sample(i).adc());

	//	Fill up Online Plots
	double aveP = hcaldqm::math::sum(hit, 0, digisizeToUse-1, 
		0, true)/digisizeToUse;
	_mes[nameRes + "_Pedestals"].Fill(aveP);
	if (subdet==hcaldqm::constants::STD_SUBDET_HO)
		_mes["HOD4_PedestalsMap"].Fill(ieta, iphi, aveP);
	else
		_mes["HBHEHFD" + 
			boost::lexical_cast<std::string>(depth) +
			"_PedestalsMap"].Fill(ieta, iphi, aveP);
}

//	Important!
/* virtual */ bool HcalPedestalTask::isApplicable(edm::Event const& e)
{

	if (!_mi.isGlobal)
	{
		//	For Local
		edm::Handle<HcalTBTriggerData>		ctbt;
		INITCOLL(_labels["HCALTBTrigger"], ctbt);
		return ctbt->wasSpillIgnorantPedestalTrigger();
	}
	else
	{
		//	For Global
		return _mi.currentCalibType==hcaldqm::constants::CT_PED;
	}

	return false;
}

/*
virtual  bool shouldBook()
{
	return !_mi.isGlobal;
}
*/

DEFINE_FWK_MODULE(HcalPedestalTask);



