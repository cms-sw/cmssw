
#include "DQM/HcalTasks/interface/FCDTask.h"

bool operator==(const FCDTask::FCDChannel& lhs, const FCDTask::FCDChannel& rhs) {
    return ((lhs.crate == rhs.crate) && (lhs.slot == rhs.slot) && (lhs.fiber == rhs.fiber) && (lhs.fiberChannel == rhs.fiberChannel));
}


FCDTask::FCDTask(edm::ParameterSet const& ps)
{
	//	tags
	_tagQIE10 = ps.getUntrackedParameter<edm::InputTag>("tagQIE10",
		edm::InputTag("hcalDigis", "ZDC"));
	_tokQIE10 = consumes<QIE10DigiCollection>(_tagQIE10);

	// channels
	edm::ParameterSet channelPSet = ps.getParameter<edm::ParameterSet>("fcdChannels");
	std::vector<int32_t> crates = channelPSet.getUntrackedParameter<std::vector<int32_t> >("crate");
	std::vector<int32_t> slots = channelPSet.getUntrackedParameter<std::vector<int32_t> >("slot");
	std::vector<int32_t> fibers = channelPSet.getUntrackedParameter<std::vector<int32_t> >("fiber");
	std::vector<int32_t> fiberChannels = channelPSet.getUntrackedParameter<std::vector<int32_t> >("fiber_channel");
	for (unsigned int i = 0; i < crates.size(); ++i) {
		_channels.push_back({crates[i], slots[i], fibers[i], fiberChannels[i]});
	}
}

/* virtual */ void FCDTask::bookHistograms(DQMStore::IBooker &ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	edm::ESHandle<HcalDbService> dbService;
	es.get<HcalDbRecord>().get(dbService);
	_emap = dbService->getHcalMapping();
	_ehashmap.initialize(_emap, hcaldqm::electronicsmap::fD2EHashMap);

	ib.cd();

	//book histos per channel
	std::string histoname;
	std::vector<HcalGenericDetId> gids = _emap->allPrecisionId();
	for (auto& it_gid : gids) {
		if (it_gid.genericSubdet() != HcalGenericDetId::HcalGenZDC) {
			continue;
		}
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(it_gid));
		for (auto& it_channel : _channels) {
			if ((eid.crateId() == it_channel.crate) && (eid.slot() == it_channel.slot) && (eid.fiberIndex() == it_channel.fiber) && (eid.fiberChanId() == it_channel.fiberChannel)) {
				_fcd_eids.push_back(eid);
			}
		}
	}
	for (auto& it_eid : _fcd_eids) {
		// EM Pos
		histoname = std::to_string(it_eid.crateId()) + "-" + std::to_string(it_eid.slot()) + "-" + std::to_string(it_eid.fiberIndex()) + "-" + std::to_string(it_eid.fiberChanId());
		ib.setCurrentFolder("Hcal/FCDTask/ADC");
		_cADC[it_eid] = ib.book1D( histoname.c_str(), histoname.c_str(), 256, 0, 256);
		_cADC[it_eid]->setAxisTitle("ADC", 1);
		_cADC[it_eid]->setAxisTitle("N", 2);

		ib.setCurrentFolder("Hcal/FCDTask/ADC_vs_TS"),
		_cADC_vs_TS[it_eid] = ib.book2D( histoname.c_str(), histoname.c_str(), 10, 0, 10, 64, 0, 256);
		_cADC_vs_TS[it_eid]->setAxisTitle("TS", 1);
		_cADC_vs_TS[it_eid]->setAxisTitle("ADC", 2);

		ib.setCurrentFolder("Hcal/FCDTask/TDCTime");
		_cTDCTime[it_eid] = ib.book1D( histoname.c_str(), histoname.c_str(), 500, 0., 250.);
		_cTDCTime[it_eid]->setAxisTitle("TDC time [ns]", 1);


		ib.setCurrentFolder("Hcal/FCDTask/TDC");
		_cTDC[it_eid] = ib.book1D( histoname.c_str(), histoname.c_str(), 64, -0.5, 63.5);
		_cTDC[it_eid]->setAxisTitle("TDC", 1);

	}
}


/* virtual */ void FCDTask::analyze(edm::Event const& e, edm::EventSetup const&)
{
	edm::Handle<QIE10DigiCollection> digis;
	if (!e.getByToken(_tokQIE10, digis))
		edm::LogError("Collection QIE10DigiCollection for ZDC isn't available"
				+ _tagQIE10.label() + " " + _tagQIE10.instance());

	for ( auto it = digis->begin(); it != digis->end(); it++ ) {
		const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);
		HcalGenericDetId const& gdid = digi.detid();
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(gdid));
		if (std::find(_fcd_eids.begin(), _fcd_eids.end(), eid) == _fcd_eids.end()) {
			continue;
		}

		for ( int i = 0; i < digi.samples(); i++ ) {
			// iter over all samples
			_cADC[eid]->Fill(digi[i].adc());
			_cADC_vs_TS[eid]->Fill(i, digi[i].adc());
			_cTDC[eid]->Fill(digi[i].le_tdc());
			if (digi[i].le_tdc() <= 50.) {
				double tdctime = 25. * i + 0.5 * digi[i].le_tdc();
				_cTDCTime[eid]->Fill(tdctime);
			}
		}
	}
}


DEFINE_FWK_MODULE(FCDTask);
