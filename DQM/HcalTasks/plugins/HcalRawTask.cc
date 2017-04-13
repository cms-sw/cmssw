//	cmssw includes
#include "DQM/HcalTasks/interface/HcalRawTask.h"

//	system includes
#include <iostream>
#include <string>

HcalRawTask::HcalRawTask(edm::ParameterSet const&ps):
	hcaldqm::HcalDQSource(ps)
{}

/* virtual */ HcalRawTask::~HcalRawTask()
{
}

/* virtual */ void HcalRawTask::beginLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::beginLuminosityBlock(lb, es);
}

/* virtual */ void HcalRawTask::endLuminosityBlock(
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	HcalDQSource::endLuminosityBlock(lb, es);
}

/* virtual */ void HcalRawTask::doWork(edm::Event const& e,
		edm::EventSetup const& es)
{
	//	We need this module only for the Normal Calibration Mode
	if (_mi.currentCalibType>0)
		return;

	edm::Handle<FEDRawDataCollection> craw;
	edm::Handle<HcalUnpackerReport> report;

	INITCOLL(_labels["RAW"], craw);
	INITCOLL(_labels["UnpackerReport"], report);

	this->process(*craw);
	this->dumpUnpackerReport(*report);

	//	Per Event Fills
	_mes["NumFEDsUnpackedvsLS"].Fill(_mi.currentLS, _numFEDsUnpackedPerEvent);
}

//	dump all the unpacker report information into the MEs
void HcalRawTask::dumpUnpackerReport(HcalUnpackerReport const& report)
{
	//	Get All the Info you need
	bool errorFree			= report.errorFree();
//	bool anyValidHcal		= report.anyValidHCAL();
	int unmappedDigis		= report.unmappedDigis();
	int unmappedTPDigis		= report.unmappedTPDigis();
	int spigotFormatErrors	= report.spigotFormatErrors();
	int badQualityDigis		= report.badQualityDigis();
	int totalDigis			= report.totalDigis();
	int totalTPDigis		= report.totalTPDigis();
//	int totalHOTPDigis		=
	int emptyEventSpigots	= report.emptyEventSpigots();
	int ofwSpigots			= report.OFWSpigots();
	int busySpigots			= report.busySpigots();

	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubErrorFree, errorFree ? 0 : 1);
	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubUnmappedDigis, unmappedDigis);
	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubUnmappedTPDigis, unmappedTPDigis);
	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubSpigotFormatErrors, spigotFormatErrors);
	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubBadQualityDigis, badQualityDigis);
	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubTotalDigis, totalDigis);
	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubTotalTPDigis, totalTPDigis);
	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubEmptyEventSpigots, emptyEventSpigots);
	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubOFWSpigots, ofwSpigots);
	_mes["Summary_UnpackerReportFlagsVsLS"].Fill(_mi.currentLS,
		hcaldqm::flags::ubBusySpigots, busySpigots);
}

/* virtyal */ void HcalRawTask::reset(int const periodflag)
{
	HcalDQSource::reset(periodflag);
	if (periodflag==0)
	{
		//	per event
		_numFEDsUnpackedPerEvent=0;
	}
	else if (periodflag==1)
	{
		//	per LS
	}
}

void HcalRawTask::specialize(FEDRawData const& raw, int ifed)
{
	if (this->isuTCA(ifed))
	{
		hcal::AMC13Header const* amc13h = (hcal::AMC13Header const*)(raw.data());
		if (!amc13h)
			return;
		amc13(amc13h, raw.size(), ifed);
		_numFEDsUnpackedPerEvent++;
		_mes["uTCA_FEDsUnpacked"].Fill(ifed);
	}
	else
	{
		HcalDCCHeader const* dcch = (HcalDCCHeader const*)(raw.data());
		if (!dcch)
			return;
		dcc(dcch, raw.size(), ifed);
		_numFEDsUnpackedPerEvent++;
		_mes["VME_FEDsUnpacked"].Fill(ifed);
	}
}

//	Some private functions
bool HcalRawTask::isuTCA(int const ifed) const
{
	if (ifed<FEDNumbering::MINHCALuTCAFEDID || ifed>FEDNumbering::MAXHCALuTCAFEDID)
		return false;
	else 
		return true;

	return false;
}

//	For AMC13/uTCA 
void HcalRawTask::amc13(hcal::AMC13Header const* amc13h, 
		unsigned int const size, int const ifed)
{
	//	Get The Info you need 
//	int				sourceId			= amc13h->sourceId();
	int				bx					= amc13h->bunchId();
	unsigned int	orn					= amc13h->orbitNumber();
	int				l1a					= amc13h->l1aNumber();
	int				namc				= amc13h->NAMC();
//	int				amc13version		= amc13h->AMC13FormatVersion();
	
	//	Iterate over all AMCs
	for (int iamc=0; iamc<namc; iamc++)
	{
		//	Get the info for that AMC13
		int slot		= amc13h->AMCSlot(iamc);
		int crate		= amc13h->AMCId(iamc)&0xFF;
//		int amcsize		= amc13h->AMCSize(iamc)/1000;

		_mes["uTCA_CratesVSslots"].Fill(slot, crate);
		HcalUHTRData uhtr(amc13h->AMCPayload(iamc), amc13h->AMCSize(iamc));
		for (HcalUHTRData::const_iterator iuhtr=uhtr.begin(); iuhtr!=uhtr.end();
				++iuhtr)
		{
			if (!iuhtr.isHeader())
				continue;

			//	Flavor determines what kind of data this uhtr contains
			//	tp, regular digi, upgrade qie digis, etc..
			if (iuhtr.flavor()==hcaldqm::constants::UTCA_DATAFLAVOR)
			{
				//	get the Info you need
				int fiber = (iuhtr.channelid()>>2)&0x1F;
				int fibchannel = iuhtr.channelid()&0x3;
				uint32_t	l1a_uhtr	= uhtr.l1ANumber();
				uint32_t	bx_uhtr		= uhtr.bunchNumber();
				uint32_t	orn_uhtr	= uhtr.orbitNumber();
				int32_t	dbcn = bx_uhtr - bx;
				int32_t	dorn = orn_uhtr - orn;
				int32_t devn = l1a_uhtr-l1a;

				//	Fill 
				_mes["uTCA_C" + 
					boost::lexical_cast<std::string>(crate) + "S" + 
					boost::lexical_cast<std::string>(slot) + 
					"_Channels"].Fill(fiber, fibchannel);
//				_mes["uTCA_DataSize"].Fill(amcsize);
				_mes["uTCA_C" + 
					boost::lexical_cast<std::string>(crate) + "S" + 
					boost::lexical_cast<std::string>(slot) + 
					"_EvNComp"].Fill(devn);
				_mes["uTCA_C" + 
					boost::lexical_cast<std::string>(crate) + "S" + 
					boost::lexical_cast<std::string>(slot) + 
					"_ORNComp"].Fill(dorn);
				_mes["uTCA_C" + 
					boost::lexical_cast<std::string>(crate) + "S" + 
					boost::lexical_cast<std::string>(slot) + 
					"_BcNComp"].Fill(dbcn);

				_mes["uTCA_CratesVSslots_dBcN"].Fill(slot, crate, abs(dbcn));
				_mes["uTCA_CratesVSslots_dOrN"].Fill(slot, crate, abs(dorn));
				_mes["uTCA_CratesVSslots_dEvN"].Fill(slot, crate, abs(devn));

				//	Fill Summary Plots
				if (dbcn!=0)
				{
					_mes["Summary_Flags"].Fill(hcaldqm::flags::rbMMBcN, 0);
					_mes["Summary_FlagsVsLS"].Fill(_mi.currentLS, 
						hcaldqm::flags::rbMMBcN);
				}
				if (dorn!=0)
				{
					_mes["Summary_Flags"].Fill(hcaldqm::flags::rbMMOrN, 0);
					_mes["Summary_FlagsVsLS"].Fill(_mi.currentLS, 
						hcaldqm::flags::rbMMOrN);
				}
				if (devn!=0)
				{
					_mes["Summary_Flags"].Fill(hcaldqm::flags::rbMMEvN, 0);
					_mes["Summary_FlagsVsLS"].Fill(_mi.currentLS, 
						hcaldqm::flags::rbMMEvN);
				}
			}
		}
	}
}

//	For DCC/VME
void HcalRawTask::dcc(HcalDCCHeader const* dcch, unsigned int const size, int const ifed)
{
	//	Get the Info you need
//	unsigned int		bytes			= dcch->getTotalLengthBytes();
	int					sourceId		= dcch->getSourceId();
	int					bx				= dcch->getBunchId();
	unsigned int		orn				= dcch->getOrbitNumber();
	unsigned long		evn				= dcch->getDCCEventNumber();
	int					dccid			= sourceId-hcaldqm::constants::VME_DCC_OFFSET;
	HcalHTRData			htr;

	//	Iterate over all spigots
	for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++)
	{
		//	Get the info you need
		int ret = dcch->getSpigotData(spigot, htr, size);
		if (ret!=0)
		{
			if (ret==-1)
				//	Set Invalid Data Flag
			continue;
		}
		unsigned int htr_evn				= htr.getL1ANumber();
		unsigned int htr_orn				= htr.getOrbitNumber();
		unsigned int htr_bx					= htr.getBunchNumber();
		int32_t devn = htr_evn - evn;
		int32_t dorn = htr_orn - orn;
		int32_t dbcn = htr_bx - bx;

		//	Fill the plots
		_mes["VME_DCCvsSpigots"].Fill(spigot, dccid);
//		_mes["VME_D" + 
//			boost::lexical_cast<std::string>(crate) + "S" + 
//			boost::lexical_cast<std::string>(slot) + 
//			"_Channels"].Fill(fiber, fibchannel);
//				_mes["uTCA_DataSize"].Fill(amcsize);
		_mes["VME_D" + 
			boost::lexical_cast<std::string>(dccid) + "S" + 
			boost::lexical_cast<std::string>(spigot) + 
			"_EvNComp"].Fill(devn);
		_mes["VME_D" + 
			boost::lexical_cast<std::string>(dccid) + "S" + 
			boost::lexical_cast<std::string>(spigot) + 
			"_ORNComp"].Fill(dorn);
		_mes["VME_D" + 
			boost::lexical_cast<std::string>(dccid) + "S" + 
			boost::lexical_cast<std::string>(spigot) + 
			"_BcNComp"].Fill(dbcn);

		_mes["VME_DCCvsSpigots_dBcN"].Fill(spigot, dccid, abs(dbcn));
		_mes["VME_DCCvsSpigots_dOrN"].Fill(spigot, dccid, abs(dorn));
		_mes["VME_DCCvsSpigots_dEvN"].Fill(spigot, dccid, abs(devn));
		
		//	Fill Summary Plots
		if (dbcn!=0)
		{
			_mes["Summary_Flags"].Fill(hcaldqm::flags::rbMMBcN, 1);
			_mes["Summary_FlagsVsLS"].Fill(_mi.currentLS, 
				hcaldqm::flags::rbMMBcN);
		}
		if (dorn!=0)
		{
			_mes["Summary_Flags"].Fill(hcaldqm::flags::rbMMOrN, 1);
			_mes["Summary_FlagsVsLS"].Fill(_mi.currentLS, 
				hcaldqm::flags::rbMMOrN);
		}
		if (devn!=0)
		{
			_mes["Summary_Flags"].Fill(hcaldqm::flags::rbMMEvN, 1);
			_mes["Summary_FlagsVsLS"].Fill(_mi.currentLS, 
				hcaldqm::flags::rbMMEvN);
		}
	}
}

DEFINE_FWK_MODULE(HcalRawTask);



