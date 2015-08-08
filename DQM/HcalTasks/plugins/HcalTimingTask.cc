//	cmssw includes
#include "DQM/HcalTasks/interface/HcalTimingTask.h"

//	system includes
#include <iostream>
#include <string>

HcalTimingTask::HcalTimingTask(edm::ParameterSet const&ps):
	hcaldqm::HcalDQSource(ps)
{}

/* virtual */ HcalTimingTask::~HcalTimingTask()
{
}

/* virtual */void HcalTimingTask::beginLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::beginLuminosityBlock(lb, es);
}

/* virtual */void HcalTimingTask::endLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::endLuminosityBlock(lb, es);
}

/* virtual */ void HcalTimingTask::doWork(edm::Event const& e,
		edm::EventSetup const& es)
{
	edm::Handle<HFDigiCollection> chf;
	edm::Handle<HBHEDigiCollection> chbhe;

	INITCOLL(_labels["HFDigi"], chf);
	INITCOLL(_labels["HBHEDigi"], chbhe);

	this->process(*chf, std::string("HF"));
	this->process(*chbhe, std::string("HB"));
	this->process(*chbhe, std::string("HE"));
}

/* virtual */void HcalTimingTask::reset(int const periodflag)
{
	HcalDQSource::reset(periodflag);
	if (periodflag==0)
	{
		//each event
	}
	else if (periodflag==1)
	{
		//each LS
	}
}

template<typename Hit>
void HcalTimingTask::specialize(Hit const& hit, std::string const&nameRes,
		int const wtw)
{
	if (nameRes=="HF")
		hf(hit);
	else if (nameRes=="HB" || nameRes=="HE")
		hbhe(hit);
	return;
}

template<typename HIT>
void HcalTimingTask::hbhe(HIT const& digi)
{
	if (_mi.currentCalibType>0)
		return;

	int iphi = digi.id().iphi();

	double q_TS5 = digi.sample(5).nominal_fC() - hcaldqm::constants::STD_HB_PED;
	double q_TS4 = digi.sample(4).nominal_fC() - hcaldqm::constants::STD_HB_PED;

	//	Apply the cuts - requested by Martin
	if (q_TS4<25 || q_TS5<5)
		return;

	double r_TS5TS4 = q_TS5/q_TS4;

	//	Fill the plots
	if (iphi>=3 && iphi<=26)
		_mes["HBHE_TS5TS4_iphi3to26"].Fill(r_TS5TS4);
	else if (iphi>=27 && iphi<=50)
		_mes["HBHE_TS5TS4_iphi27to50"].Fill(r_TS5TS4);
	else 
		_mes["HBHE_TS5TS4_iphi1to2_iphi51to72"].Fill(r_TS5TS4);
	_mes["HBHE_TS5TS4VSiphi"].Fill(iphi, r_TS5TS4);
}

template<typename HIT>
void HcalTimingTask::hf(HIT const& digi)
{
	//	Do Phase Scan only for a Normal Calibration Mode
	if (_mi.currentCalibType>0)
		return;

	//	Obtain things you need from the digi
	int nTS=digi.size();
	int iphi = digi.id().iphi();
	int ieta = digi.id().ieta();
	int depth = digi.id().depth();
	int maxTS = hcaldqm::math::maxTS(digi, 
			hcaldqm::constants::STD_HF_PED);
	double aveT = hcaldqm::math::aveT(digi, 
			hcaldqm::constants::STD_HF_PED);
	double sumQ_3TS = hcaldqm::math::sum(digi, maxTS-1, maxTS+1, 
			hcaldqm::constants::STD_HF_PED);
	double sumQ_TS12 = hcaldqm::math::sum(digi, 1, 2, 
			hcaldqm::constants::STD_HF_PED);
	double sumQ_TS23 = hcaldqm::math::sum(digi, 2, 3, 
			hcaldqm::constants::STD_HF_PED);
	double q_TS2 = digi.sample(2).nominal_fC()-
			hcaldqm::constants::STD_HF_PED;
	double qTS2QTS12 = q_TS2/sumQ_TS12;
	double qTS2QTS23 = q_TS2/sumQ_TS23;

	//	Fill the plots
	_mes["SumQ_3TS"].Fill(sumQ_3TS);
	for (int i=0; i<nTS; i++)
	{
		if (ieta<0)
		{
			_mes["HFM_Shape"].Fill(i, 
					digi.sample(i).nominal_fC()-
					hcaldqm::constants::STD_HF_PED);
			if (sumQ_3TS>=hcaldqm::constants::STD_HF_DIGI_CUT_3TSQg20)
				_mes["HFM_Shape_3TSQg20"].Fill(i,
						digi.sample(i).nominal_fC()-
						hcaldqm::constants::STD_HF_PED);
		}
		else
		{
			_mes["HFP_Shape"].Fill(i,
					digi.sample(i).nominal_fC()-
					hcaldqm::constants::STD_HF_PED);
			if (sumQ_3TS>=hcaldqm::constants::STD_HF_DIGI_CUT_3TSQg20)
				_mes["HFP_Shape_3TSQg20"].Fill(i,
						digi.sample(i).nominal_fC()-
						hcaldqm::constants::STD_HF_PED);
		}
	}

	//	Exit if you didn't pass the 3TSQg20 Cut
	if (sumQ_3TS<hcaldqm::constants::STD_HF_DIGI_CUT_3TSQg20)
		return;

	//	Fill the plots
	if (ieta<0)
	{
		_mes["HFM_Timing"].Fill(aveT);
		_mes["HFM_TimingVSls"].Fill(_mi.currentLS, aveT);
		_mes["HFM_OccupancyietavsLS"].Fill(_mi.currentLS, ieta);
		_mes["HFM_TimingVSls2D"].Fill(_mi.currentLS, aveT);

		_mes["HFM_QTS2QTS12"].Fill(qTS2QTS12);
		_mes["HFM_QTS2QTS12vsLS"].Fill(_mi.currentLS, qTS2QTS12);
		_mes["HFM_QTS2QTS12vsLS2D"].Fill(_mi.currentLS, qTS2QTS12);

		_mes["HFM_QTS2QTS23"].Fill(qTS2QTS23);
		_mes["HFM_QTS2QTS23vsLS"].Fill(_mi.currentLS, qTS2QTS23);
		_mes["HFM_QTS2QTS23vsLS2D"].Fill(_mi.currentLS, qTS2QTS23);

		if (iphi==43)
		{
			_mes["HFMiphi43_QTS2QTS12vsLS"].Fill(_mi.currentLS, qTS2QTS12);
			_mes["HFMiphi43_QTS2QTS23vsLS"].Fill(_mi.currentLS, qTS2QTS23);
		}
	}
	else 
	{
		_mes["HFP_Timing"].Fill(aveT);
		_mes["HFP_TimingVSls"].Fill(_mi.currentLS, aveT);
		_mes["HFP_OccupancyietavsLS"].Fill(_mi.currentLS, ieta);
		_mes["HFP_TimingVSls2D"].Fill(_mi.currentLS, aveT);

		_mes["HFP_QTS2QTS12"].Fill(qTS2QTS12);
		_mes["HFP_QTS2QTS12vsLS"].Fill(_mi.currentLS, qTS2QTS12);
		_mes["HFP_QTS2QTS12vsLS2D"].Fill(_mi.currentLS, qTS2QTS12);

		_mes["HFP_QTS2QTS23"].Fill(qTS2QTS23);
		_mes["HFP_QTS2QTS23vsLS"].Fill(_mi.currentLS, qTS2QTS23);
		_mes["HFP_QTS2QTS23vsLS2D"].Fill(_mi.currentLS, qTS2QTS23);
	}
	_mes["HF_TimingVSieta"].Fill(ieta, aveT);
	_mes["HF_OccupancyD" + boost::lexical_cast<std::string>(depth)].Fill(
			ieta, iphi);
	//_mes["HF_OccupancyVSieta"].Fill(ieta);
	_mes["HF_TimingVSieta2D"].Fill(ieta, aveT);

}

DEFINE_FWK_MODULE(HcalTimingTask);













