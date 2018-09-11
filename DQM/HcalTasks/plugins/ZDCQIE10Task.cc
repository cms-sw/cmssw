
#include "DQM/HcalTasks/interface/ZDCQIE10Task.h"
#include <map>

ZDCQIE10Task::ZDCQIE10Task(edm::ParameterSet const& ps)
{
	//	tags
	_tagQIE10 = ps.getUntrackedParameter<edm::InputTag>("tagQIE10",
		edm::InputTag("hcalDigis", "ZDC"));
	_tokQIE10 = consumes<QIE10DigiCollection>(_tagQIE10);
}

/* virtual */ void ZDCQIE10Task::bookHistograms(DQMStore::IBooker &ib,
	edm::Run const& r, edm::EventSetup const& es)
{

	ib.cd();

	//book histos per channel
	char histoname[300];
	for ( int channel = 1; channel < 6; channel++ ) {
		// EM Pos
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/Shape_perChannel");
		sprintf(histoname,"EM_P_%i_1", channel);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
		_cADC_EChannel[histoname] = ib.book1D( histoname, histoname, 256, 0, 256);
		_cADC_EChannel[histoname]->setAxisTitle("ADC", 1);
		_cADC_EChannel[histoname]->setAxisTitle("N", 2);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
		_cADC_vs_TS_EChannel[histoname] = ib.book1D( histoname, histoname, 10, 0, 10);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("TS");
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("sum ADC");

		// EM Minus
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/Shape_perChannel");
		sprintf(histoname,"EM_M_%i_1", channel);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
		_cADC_EChannel[histoname] = ib.book1D( histoname, histoname, 256, 0, 256);
		_cADC_EChannel[histoname]->setAxisTitle("ADC", 1);
		_cADC_EChannel[histoname]->setAxisTitle("N", 2);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
		_cADC_vs_TS_EChannel[histoname] = ib.book1D( histoname, histoname, 10, 0, 10);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("TS", 1);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("sum ADC", 2);
	}

	for ( int channel = 1; channel < 5; channel++ ) {
		// HAD Pos
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/Shape_perChannel");
		sprintf(histoname,"HAD_P_%i_%i", channel, channel+2);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
		_cADC_EChannel[histoname] = ib.book1D( histoname, histoname, 256, 0, 256);
		_cADC_EChannel[histoname]->setAxisTitle("ADC", 1);
		_cADC_EChannel[histoname]->setAxisTitle("N", 2);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
		_cADC_vs_TS_EChannel[histoname] = ib.book1D( histoname, histoname, 10, 0, 10);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("TS", 1);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("sum ADC", 2);

		// HAD Minus
		sprintf(histoname,"HAD_M_%i_%i", channel, channel+2);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
		_cADC_EChannel[histoname] = ib.book1D( histoname, histoname, 256, 0, 256);
		_cADC_EChannel[histoname]->setAxisTitle("ADC", 1);
		_cADC_EChannel[histoname]->setAxisTitle("N", 2);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
		_cADC_vs_TS_EChannel[histoname] = ib.book1D( histoname, histoname, 10, 0, 10);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("TS", 1);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("sum ADC", 2);
	}

	for ( int channel = 1; channel < 17; channel++ ) {
		// RPD Pos
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/Shape_perChannel");
		sprintf(histoname,"RPD_P_%i_2", channel);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
		_cADC_EChannel[histoname] = ib.book1D( histoname, histoname, 256, 0, 256);
		_cADC_EChannel[histoname]->setAxisTitle("ADC");
		_cADC_EChannel[histoname]->setAxisTitle("N");
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
		_cADC_vs_TS_EChannel[histoname] = ib.book1D( histoname, histoname, 10, 0, 10);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("TS", 1);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("sum ADC", 2);

		// RPD Minus
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/Shape_perChannel");
		sprintf(histoname,"RPD_M_%i_2", channel);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
		_cADC_EChannel[histoname] = ib.book1D( histoname, histoname, 256, 0, 256);
		_cADC_EChannel[histoname]->setAxisTitle("ADC", 1);
		_cADC_EChannel[histoname]->setAxisTitle("N", 2);
		ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
		_cADC_vs_TS_EChannel[histoname] = ib.book1D( histoname, histoname, 10, 0, 10);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("TS", 1);
		_cADC_vs_TS_EChannel[histoname]->setAxisTitle("sum ADC", 2);
	}

}


/* virtual */ void ZDCQIE10Task::analyze(edm::Event const& e, edm::EventSetup const&)
{
	edm::Handle<QIE10DigiCollection> digis;
	if (!e.getByToken(_tokQIE10, digis))
		edm::LogError("Collection QIE10DigiCollection for ZDC isn't available"
				+ _tagQIE10.label() + " " + _tagQIE10.instance());

	for ( auto it = digis->begin(); it != digis->end(); it++ ) {
		const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);
		HcalZDCDetId const& did = digi.detid();

		std::string section;
		switch (did.section()) {
			case(HcalZDCDetId::EM)  : section = "EM_";  break;
			case(HcalZDCDetId::HAD) : section = "HAD_"; break;
			case(HcalZDCDetId::LUM) : section = "LUM_"; break;
			case(HcalZDCDetId::RPD) : section = "RPD_"; break;
			default : section = "UNKNOWN_";
		}
		std::string zside = (did.zside()==1)?("P_"):("M_");

		std::string histoname = section + zside + std::to_string(did.channel()) + "_" + std::to_string(did.depth());

		for ( int i = 0; i < digi.samples(); i++ ) {
			// iter over all samples
			if ( _cADC_EChannel.find( histoname ) != _cADC_EChannel.end() ) {
				_cADC_EChannel[histoname ]->Fill(digi[i].adc());
			}
			if ( _cADC_vs_TS_EChannel.find( histoname ) != _cADC_vs_TS_EChannel.end() ) {
				_cADC_vs_TS_EChannel[histoname ]->Fill(i, digi[i].adc());
			}

		}
	}
}


DEFINE_FWK_MODULE(ZDCQIE10Task);
