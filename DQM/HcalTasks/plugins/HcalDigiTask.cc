//	cmssw includes
#include "DQM/HcalTasks/interface/HcalDigiTask.h"

//	system includes
#include <iostream>
#include <string>

HcalDigiTask::HcalDigiTask(edm::ParameterSet const&ps):
	hcaldqm::HcalDQSource(ps)
{
	_ornMsgTime = ps.getUntrackedParameter<int>("OrnMsgTime", 3559);
	for (int i=0; i<hcaldqm::constants::STD_NUMSUBS; i++)
	{
		_numDigis_wZSCut[i] = 0;
		_numDigis_NoZSCut[i] = 0;
	}
}

/* virtual */ HcalDigiTask::~HcalDigiTask()
{
}

/* virtual */ void HcalDigiTask::beginLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::beginLuminosityBlock(lb, es);
}

/* virtual */ void HcalDigiTask::endLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::endLuminosityBlock(lb, es);	
}

/* virtual */ void HcalDigiTask::doWork(edm::Event const& e,
		edm::EventSetup const& es)
{
	//	We need this module only for Normal Calibration Mode
	if (_mi.currentCalibType>0)
		return;
	
	edm::Handle<HBHEDigiCollection>			chbhe;
	edm::Handle<HODigiCollection>			cho;
	edm::Handle<HFDigiCollection>			chf;
	edm::Handle<DcsStatusCollection>		cdcs;
	
	//	Initialize collections with try/catch
	INITCOLL(_labels["HBHEDigi"], chbhe);
	INITCOLL(_labels["HODigi"], cho);
	INITCOLL(_labels["HFDigi"], chf);
//	INITCOLL(_labels["DCS"], cdcs);

	//	DCS
//	_dcs.initDcs(*cdcs);

	//	Run the Processors
	this->process(*chbhe, std::string("HB"));
	this->process(*chbhe, std::string("HE"));
	this->process(*cho, std::string("HO"));
	this->process(*chf, std::string("HF"));	

	//	Fill Plots
	_mes["HB_OccupancyVSls_wZSCut"].Fill(_mi.currentLS, _numDigis_wZSCut[0]);
	_mes["HE_OccupancyVSls_wZSCut"].Fill(_mi.currentLS, _numDigis_wZSCut[1]);
	_mes["HO_OccupancyVSls_wZSCut"].Fill(_mi.currentLS, _numDigis_wZSCut[2]);
	_mes["HF_OccupancyVSls_wZSCut"].Fill(_mi.currentLS, _numDigis_wZSCut[3]);
	_mes["HB_OccupancyVSls_NoZSCut"].Fill(_mi.currentLS, _numDigis_NoZSCut[0]);
	_mes["HE_OccupancyVSls_NoZSCut"].Fill(_mi.currentLS, _numDigis_NoZSCut[1]);
	_mes["HO_OccupancyVSls_NoZSCut"].Fill(_mi.currentLS, _numDigis_NoZSCut[2]);
	_mes["HF_OccupancyVSls_NoZSCut"].Fill(_mi.currentLS, _numDigis_NoZSCut[3]);
}

/* virtual */ void HcalDigiTask::reset(int const periodflag)
{
	HcalDQSource::reset(periodflag);
	if (periodflag==0)
	{
		// each events reset
		for (unsigned int i=0; i<hcaldqm::constants::STD_NUMSUBS; i++)
		{
			_numDigis_wZSCut[i]=0;
			_numDigis_NoZSCut[i]=0;
		}
	}
	else if (periodflag==1)
	{
		// each LS reset
	}
}

//	specializer
template<typename Hit>
void HcalDigiTask::specialize(Hit const& hit, std::string const& nameRes,
		int const wtw)
{
	//	offset of BCN for this channel(digi) relative to the nominal 
	//	in the unpacker. range(-7, +7), -1000 indicates invalid data
	int offset = hit.fiberIdleOffset();
	if (offset!=hcaldqm::constants::INVALID_FIBER_IDLEOFFSET 
			&& _ornMsgTime>-1)
		_mes[nameRes+"_bcnOffset"].Fill(offset);

	int iphi = hit.id().iphi();
	int ieta = hit.id().ieta();
	int depth = hit.id().depth();
	int subdet = hit.id().subdet()-1;
	int digisize = hcaldqm::constants::DIGISIZE_GLOBAL[subdet];
	int maxTS = hcaldqm::math::maxTS(hit, (subdet<4) ? 
			hcaldqm::constants::PEDESTALS[subdet]: 0);
	double aveT = hcaldqm::math::aveT(hit, (subdet<4) ? 
			hcaldqm::constants::PEDESTALS[subdet]:0);
	double sumQ_3TS = hcaldqm::math::sum(hit, maxTS-1, maxTS+1, (subdet<4) ? 
			hcaldqm::constants::PEDESTALS[subdet]:0);

//	std::cout << nameRes << "  " << maxTS << "  " << aveT << "  " 
//		<< sumQ_3TS << "  " << hit.size()
//		<< std::endl;

	_mes["DigiSize"].Fill(subdet, hit.size());
	_mes["DigiSizeExp"].Fill(subdet, digisize);
	if (subdet<4)
		_numDigis_NoZSCut[subdet]++;

	//	Fill some plots before ZS Cut
	if (subdet==hcaldqm::constants::STD_SUBDET_HO)
	{
		_mes["HO_OccupancyMapD" + 
			boost::lexical_cast<std::string>(depth)].Fill(ieta, iphi, 1);
		_mes["HO_OccupancyMapD" + 
			boost::lexical_cast<std::string>(depth) + 
			"_LS"].Fill(ieta, iphi, 1);
	}
	else if (subdet==hcaldqm::constants::STD_SUBDET_HB  || 
			subdet==hcaldqm::constants::STD_SUBDET_HE || 
			subdet==hcaldqm::constants::STD_SUBDET_HF)
	{
		_mes["HBHEHF_OccupancyMapD" + 
			boost::lexical_cast<std::string>(depth)].Fill(ieta, iphi, 1);
		_mes["HBHEHF_OccupancyMapD" + 
			boost::lexical_cast<std::string>(depth) + 
			"_LS"].Fill(ieta, iphi, 1);
	}
	

	//	Extract per 1TS Information
	for (int i=0; i<hit.size(); i++)
	{
		_mes[nameRes+"_DigiShape"].Fill(i, 
				(subdet<4) ? hit.sample(i).nominal_fC() - 
				hcaldqm::constants::PEDESTALS[subdet]: 0);
		_mes[nameRes+"_ADCCountPerTS"].Fill(hit.sample(i).adc());
		_mes[nameRes+"_fCPerTS"].Fill(hit.sample(i).nominal_fC());
		_mes[nameRes+"_Presamples"].Fill(hit.presamples());
		_mes[nameRes+"_CapId"].Fill(hit.sample(i).capid());

		if (sumQ_3TS>hcaldqm::constants::DIGI_ZSCUT[(subdet<4) ? subdet : 0])
			_mes[nameRes+"_DigiShape_ZSCut"].Fill(i, 
				(subdet<4) ? hit.sample(i).nominal_fC() - 
				hcaldqm::constants::PEDESTALS[subdet]: 0);
	}

	//	Fill Plots if > ZSCut
	if (sumQ_3TS<hcaldqm::constants::DIGI_ZSCUT[(subdet<4)?subdet:0])
		return;

	_mes[nameRes + "_Timing_ZSCut"].Fill(aveT);
	_mes[nameRes + "_TimingVSieta_ZSCut"].Fill(ieta, aveT);
	_mes[nameRes + "_TimingVSiphi_ZSCut"].Fill(iphi, aveT);
	if (subdet<4)
		_numDigis_wZSCut[subdet]++;
	if (subdet==hcaldqm::constants::STD_SUBDET_HO)
	{
		_mes["HO_OccupancyMapD" + 
			boost::lexical_cast<std::string>(depth) + 
			"_ZSCut"].Fill(ieta, iphi, 1);
		_mes["HOD4_TimingMap_ZSCut"].Fill(ieta, iphi, aveT);
	}
	else if (subdet==hcaldqm::constants::STD_SUBDET_HB  || 
			subdet==hcaldqm::constants::STD_SUBDET_HE || 
			subdet==hcaldqm::constants::STD_SUBDET_HF)
	{
		_mes["HBHEHF_OccupancyMapD" + 
			boost::lexical_cast<std::string>(depth) + 
			"_ZSCut"].Fill(ieta, iphi, 1);
		_mes["HBHEHFD" + 
			boost::lexical_cast<std::string>(depth) + 
			"_TimingMap_ZSCut"].Fill(ieta, iphi, aveT);
	}
	if (subdet==hcaldqm::constants::STD_SUBDET_HF)
	{
		std::string name_tmp;
		if (ieta<0)
			name_tmp = "M";
		else
			name_tmp = "P";

		_mes["HF" + name_tmp + "_OccupancyVSiphi"].Fill(iphi);
		_mes["HF" + name_tmp + "_OccupancyiphiVSLS"].Fill(_mi.currentLS, iphi);
	}

}

DEFINE_FWK_MODULE(HcalDigiTask);



