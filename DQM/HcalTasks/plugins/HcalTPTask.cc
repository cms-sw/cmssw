//	cmssw includes
#include "DQM/HcalTasks/interface/HcalTPTask.h"

//	system includes
#include <iostream>
#include <string>

HcalTPTask::HcalTPTask(edm::ParameterSet const&ps):
	hcaldqm::HcalDQSource(ps)
{}

/* virtual */ HcalTPTask::~HcalTPTask()
{
}

/* virtual */ void HcalTPTask::beginLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::beginLuminosityBlock(lb, es);
}

/* virtual */ void HcalTPTask::endLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::endLuminosityBlock(lb, es);
}

/* virtual */ void HcalTPTask::reset(int const periodflag)
{
	HcalDQSource::reset(periodflag);
	if (periodflag==0)
	{
		// Event
	}
	else if (periodflag==1)
	{
		// LS
	}
}

/* virtual */ void HcalTPTask::doWork(edm::Event const& e,
		edm::EventSetup const& es)
{
	edm::Handle<HcalTrigPrimDigiCollection>		ctpd;
	edm::Handle<HcalTrigPrimDigiCollection>		ctpe;

	INITCOLL(_labels["HCALTPD"], ctpd);
	INITCOLL(_labels["HCALTPE"], ctpe);

	this->debug_(_mi.name +	" Comparing Data vs Emulator");

	//	Convention is
	//	wtw=1 => iterate over ctpd and look iter_ctpd in ctpe and vice versa
	this->process(*ctpd, *ctpe, std::string("HBHE"), 1);
	this->process(*ctpe, *ctpd, std::string("HBHE"), 2);
	this->process(*ctpd, *ctpe, std::string("HF"), 1);
	this->process(*ctpe, *ctpd, std::string("HF"), 2);
}

template<typename Hit>
void HcalTPTask::specialize(Hit const& hit, std::string const& nameRes, 
		int const wtw)
{
	//	Data or Emul
	std::string tpRes = wtw==1 ? "Data" : "Emul";
	this->debug_(_mi.name + tpRes + " specializer");

	//	get all the info we need
	int iphi = hit.id().iphi();
	int ieta = hit.id().ieta();
	int soi_cEt = hit.SOI_compressedEt();
	int ps = hit.presamples();

	//	SOI plots + presamples
	_mes[nameRes + "_SOI_Et_" + tpRes].Fill(soi_cEt);
	_mes[nameRes + "_Presamples_" + tpRes].Fill(ps);

	for (int i=0; i<hit.size(); i++)
	{
		int cEt = hit.sample(i).compressedEt();
		_mes[nameRes + "_EtShape_" + tpRes].Fill(i, cEt);	
	}

	//	Occupancy Maps
	_mes[nameRes + "_TPOccupancyVSiphi_" + tpRes].Fill(iphi);
	_mes["HBHEHF_TPOccupancyVSiphi_" + tpRes].Fill(iphi);
	_mes["HBHEHF_TPOccupancyVSieta_" + tpRes].Fill(ieta);
	_mes["HBHEHF_TPOccupancy_" + tpRes].Fill(ieta, iphi);
		
}

template<typename Hit>
void HcalTPTask::specialize(Hit const& hit1, Hit const& hit2, 
		std::string const& nameRes, int const wtw)
{
	this->debug_(_mi.name + " specializer");

	//	 wtw=2 is comparing Data to Emulator, upon matching should be skipped
	//	 as this has been processed already when comparing emulator to data
	if (wtw==2)
		return;

	//	get all the info we need
	int iphi = hit1.id().iphi();
	int ieta = hit1.id().ieta();
	int subdet = hcaldqm::packaging::isHFTrigTower(hit1.id().ietaAbs()) ? 1 : 0;
	int soi_cEt_1 = hit1.SOI_compressedEt();
	int soi_cEt_2 = hit2.SOI_compressedEt();
	int soi_fg_1 = hit1.SOI_fineGrain() ? 1 : 0;
	int soi_fg_2 = hit2.SOI_fineGrain() ? 1 : 0;
	int ps_1 = hit1.presamples();
	int ps_2 = hit2.presamples();
	int digisize_1 = hit1.size();
	int digisize_2 = hit2.size();

	//	Fill the SOI Et and FG Mismatch Summary Plots
	bool matched_SOI_Et = abs(soi_cEt_1-soi_cEt_2)==0 ? true : false;
	if (!matched_SOI_Et)
	{
		_mes["Summary_Flags"].Fill(hcaldqm::flags::tpbMMEt_SOI, subdet);
		_mes["Summary_"+nameRes+"_FlagsVsLS"].Fill(_mi.currentLS, 
				hcaldqm::flags::tpbMMEt_SOI);
	}
	bool matched_SOI_FG = soi_fg_1 == soi_fg_2;
	if (!matched_SOI_FG)
	{
		_mes["Summary_Flags"].Fill(hcaldqm::flags::tpbMMFG_SOI, subdet);
		_mes["Summary_"+nameRes+"_FlagsVsLS"].Fill(_mi.currentLS, 
			hcaldqm::flags::tpbMMFG_SOI);
	}

	//	SOI plots + presamples
	_mes[nameRes + "_SOI_Et_Data"].Fill(soi_cEt_1);
	_mes[nameRes + "_SOI_Et_Emul"].Fill(soi_cEt_2);
	_mes[nameRes + "_SOI_Et_Correlation"].Fill(soi_cEt_1, soi_cEt_2);
	_mes[nameRes + "_SOI_FG_Correlation"].Fill(soi_fg_1, soi_fg_2);
	_mes[nameRes + "_Presamples_Data"].Fill(ps_1);
	_mes[nameRes + "_Presamples_Emul"].Fill(ps_2);
	_mes["HBHEHF_TPDigiSize_Data"].Fill(subdet, digisize_1);
	_mes["HBHEHF_TPDigiSize_Emul"].Fill(subdet, digisize_2);

	for (int i=0; i<hit1.size(); i++)
	{
		int cEt_1 = hit1.sample(i).compressedEt();
		int cEt_2 = hit2.sample(i).compressedEt();
		bool fg_1 = hit1.sample(i).fineGrain();
		bool fg_2 = hit2.sample(i).fineGrain();
//		bool matched_nonSOI_Et = abs(cEt_1-cEt_2)==0 ? true : false;
//		bool matched_nonSOI_FG = fg_1==fg_2;

		_mes[nameRes + "_EtShape_Data"].Fill(i, cEt_1);
		_mes[nameRes + "_EtShape_Emul"].Fill(i, cEt_2);

		//	If this is not a Sample of Interest
		if (i!=ps_1)
		{
			_mes[nameRes + "_nonSOI_Et_Correlation"].Fill(cEt_1, cEt_2);
			_mes[nameRes + "_nonSOI_FG_Correlation"].Fill(fg_1, fg_2);
		}
	}
	
	//	Occupancy Maps
	_mes[nameRes + "_TPOccupancyVSiphi_Data"].Fill(iphi);
	_mes[nameRes + "_TPOccupancyVSieta_Data"].Fill(ieta);
	_mes["HBHEHF_TPOccupancyVSiphi_Data"].Fill(iphi);
	_mes["HBHEHF_TPOccupancyVSieta_Data"].Fill(ieta);
	_mes["HBHEHF_TPOccupancy_Data"].Fill(ieta, iphi);
	
	_mes[nameRes + "_TPOccupancyVSiphi_Emul"].Fill(iphi);
	_mes[nameRes + "_TPOccupancyVSieta_Emul"].Fill(ieta);
	_mes["HBHEHF_TPOccupancyVSiphi_Emul"].Fill(iphi);
	_mes["HBHEHF_TPOccupancyVSieta_Emul"].Fill(ieta);
	_mes["HBHEHF_TPOccupancy_Emul"].Fill(ieta, iphi);
}

//	Performed if hit2(hit1.id()) isn't found in the collection.
template<typename Hit>
void HcalTPTask::check(Hit const& hit, std::string const& nameRes, int const wtw)
{
	
	//	Data or Emul
	std::string tpRes = wtw==1 ? "Data" : "Emul";
	std::string misRes = wtw==1 ? "Emul" : "Data";
	int subdet = hcaldqm::packaging::isHFTrigTower(hit.id().ietaAbs()) ? 0 : 1;

	//	line below clarifies the convention...
	int missWhat = wtw==1 ? hcaldqm::flags::tpbMissingEmul : 
		hcaldqm::flags::tpbMissingData;
	this->debug_(_mi.name + tpRes + " Checking");

	//	get all the info we need
	int iphi = hit.id().iphi();
	int ieta = hit.id().ieta();
	int soi_cEt = hit.SOI_compressedEt();
	int ps = hit.presamples();

	//	SOI plots + presamples
	_mes[nameRes + "_SOI_Et_" + tpRes].Fill(soi_cEt);
	_mes[nameRes + "_Presamples_" + tpRes].Fill(ps);

	for (int i=0; i<hit.size(); i++)
	{
		int cEt = hit.sample(i).compressedEt();
		_mes[nameRes + "_EtShape_" + tpRes].Fill(i, cEt);	
	}

	//	Occupancy Maps
	_mes[nameRes + "_TPOccupancyVSiphi_" + tpRes].Fill(iphi);
	_mes["HBHEHF_TPOccupancyVSiphi_" + tpRes].Fill(iphi);
	_mes["HBHEHF_TPOccupancyVSieta_" + tpRes].Fill(ieta);
	_mes[nameRes + "_TPOccupancyVSieta_" + tpRes].Fill(ieta);
	_mes["HBHEHF_TPOccupancy_" + tpRes].Fill(ieta, iphi);

	//	Fill out maps to show which detid is missing
	_mes["HBHEHF_Missing_" + misRes].Fill(ieta, iphi);

	//	Fill the Summar Flags
	_mes["Summary_Flags"].Fill(missWhat, subdet);
	_mes["Summary_"+nameRes+"_FlagsVsLS"].Fill(_mi.currentLS, missWhat);
}

DEFINE_FWK_MODULE(HcalTPTask);



















