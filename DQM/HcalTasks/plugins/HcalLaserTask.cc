//	cmssw includes
#include "DQM/HcalTasks/interface/HcalLaserTask.h"

//	system includes
#include <iostream>
#include <string>

HcalLaserTask::HcalLaserTask(edm::ParameterSet const&ps):
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

/* virtual */ HcalLaserTask::~HcalLaserTask()
{
}

/* virtual */ void HcalLaserTask::endRun(edm::Run const& r,
		edm::EventSetup const& es)
{
	this->publish();
}

void HcalLaserTask::publish()
{
	for (int i=0; i<hcaldqm::constants::STD_NUMSUBS; i++)
	{
		_mes[hcaldqm::constants::SUBNAMES[i] + 
			"_SignalMeans_Summary"].Reset();
		_mes[hcaldqm::constants::SUBNAMES[i] + 
			"_SignalRMSs_Summary"].Reset();
		_mes[hcaldqm::constants::SUBNAMES[i] + 
			"_TimingMeans_Summary"].Reset();
		_mes[hcaldqm::constants::SUBNAMES[i] + 
			"_TimingRMSs_Summary"].Reset();
		for (int iieta=0; iieta<hcaldqm::constants::STD_NUMIETAS; iieta++)
			for (int iiphi=0; iiphi<hcaldqm::constants::STD_NUMIPHIS; iiphi++)
				for (int id=0; id<hcaldqm::constants::STD_NUMDEPTHS; id++)
				{
					std::pair<double, double> smeanrms = 
						_laserData[i][iieta][iiphi][id].average();
					std::pair<double, double> tmeanrms = 
						_laserData[i][iieta][iiphi][id].averageTiming();
					double smean = smeanrms.first;
					double srms = smeanrms.second;
					double tmean = tmeanrms.first;
					double trms = tmeanrms.second;
					if (smean==-1 || srms==-1 || tmean==-1 || trms==-1)
						continue;

					_mes[hcaldqm::constants::SUBNAMES[i] + 
						"_SignalMeans_Summary"].Fill(smean);
					_mes[hcaldqm::constants::SUBNAMES[i] + 
						"_SignalRMSs_Summary"].Fill(srms);
					_mes[hcaldqm::constants::SUBNAMES[i] + 
						"_TimingMeans_Summary"].Fill(tmean);
					_mes[hcaldqm::constants::SUBNAMES[i] + 
						"_TimingRMSs_Summary"].Fill(trms);
				}
	}
}

/* virtual */ void HcalLaserTask::beginLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::beginLuminosityBlock(lb, es);
}

/* virtual */ void HcalLaserTask::endLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::endLuminosityBlock(lb, es);
}

/* virtual */ void HcalLaserTask::reset(int const periodflag)
{
	HcalDQSource::reset(periodflag);
	if (periodflag==0)
	{
		//	for Event
	}
	else if (periodflag==1)
	{
		//	for LS
	}
}

/* virtual */ void HcalLaserTask::doWork(edm::Event const& e,
		edm::EventSetup const& es)
{
	edm::Handle<HBHEDigiCollection>		chbhe;
	edm::Handle<HODigiCollection>		cho;
	edm::Handle<HFDigiCollection>		chf;

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
void HcalLaserTask::specialize(Hit const& hit, std::string const& nameRes,
		int const wtw)
{
	//	Get the Info you need
	int iphi = hit.id().iphi();
	int ieta = hit.id().ieta();
	int depth = hit.id().depth();
	int subdet = hit.id().subdet()-1;
	int digisize = hit.size();

	double maxTS = hcaldqm::math::maxTS(hit,
			hcaldqm::constants::PEDESTALS[subdet]);
	double aveT = hcaldqm::math::aveT(hit,
			hcaldqm::constants::PEDESTALS[subdet]);
	double sigQ = hcaldqm::math::sum(hit,
			maxTS-1, maxTS+1, hcaldqm::constants::PEDESTALS[subdet]);

	//	Fill up the prioprietary Class for Laser Monitoring
	for (int i=0; i<digisize; i++)
	{
		_laserData[subdet][_packager[subdet].iieta(ieta)]
			[_packager[subdet].iiphi(iphi)]
			[_packager[subdet].idepth(depth)].push(hit,
					hcaldqm::constants::PEDESTALS[subdet], 
					digisize>4 ? 2 : 1);

		_mes[nameRes + "_Shape"].Fill(i,
			hit.sample(i).nominal_fC()-hcaldqm::constants::PEDESTALS[subdet]);
	}

	_mes[nameRes + "_Signal"].Fill(sigQ);
	_mes[nameRes + "_Timing"].Fill(aveT);
	if (subdet==hcaldqm::constants::STD_SUBDET_HO)
	{
		_mes["HOD4_SignalMap"].Fill(ieta, iphi, sigQ);
		_mes["HOD4_TimingMap"].Fill(ieta, iphi, aveT);
	}
	else
	{
		_mes["HBHEHFD" + 
			boost::lexical_cast<std::string>(depth) +
			"_SignalMap"].Fill(ieta, iphi, sigQ);
		_mes["HBHEHFD" + 
			boost::lexical_cast<std::string>(depth) +
			"_TimingMap"].Fill(ieta, iphi, aveT);
	}
}

/* virtual */ bool HcalLaserTask::isApplicable(edm::Event const& e)
{
	if (!_mi.isGlobal)
	{
		//	For Local
		edm::Handle<HcalTBTriggerData>		ctbt;
		INITCOLL(_labels["HCALTBTrigger"], ctbt);
		return ctbt->wasLaserTrigger();
	}
	else
	{
		//	For Global
		return (_mi.currentCalibType==hcaldqm::constants::CT_RADDAM);
	}

	return false;
}

DEFINE_FWK_MODULE(HcalLaserTask);



