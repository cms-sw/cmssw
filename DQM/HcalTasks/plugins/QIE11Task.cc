#include "DQM/HcalTasks/interface/QIE11Task.h"
#include "FWCore/Framework/interface/Run.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
QIE11Task::QIE11Task(edm::ParameterSet const& ps) : DQTask(ps) {
  //	tags
  _tagQIE11 = ps.getUntrackedParameter<edm::InputTag>("tagQIE11", edm::InputTag("hcalDigis"));
  _tokQIE11 = consumes<QIE11DigiCollection>(_tagQIE11);

  _taguMN = ps.getUntrackedParameter<edm::InputTag>("taguMN", edm::InputTag("hcalDigis"));
  _tokuMN = consumes<HcalUMNioDigi>(_taguMN);

  //	cuts
  _cut = ps.getUntrackedParameter<double>("cut", 50.0);
  _ped = ps.getUntrackedParameter<int>("ped", 4);
  _laserType = ps.getUntrackedParameter<int32_t>("laserType", -1);
  _eventType = ps.getUntrackedParameter<int32_t>("eventType", -1);
}
/* virtual */ void QIE11Task::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  if (_ptype == fLocal)
    if (r.runAuxiliary().run() == 1)
      return;

  DQTask::bookHistograms(ib, r, es);

  //	GET WHAT YOU NEED
  edm::ESHandle<HcalDbService> dbs;
  es.get<HcalDbRecord>().get(dbs);
  _emap = dbs->getHcalMapping();
  std::vector<uint32_t> vhashC34;
  vhashC34.push_back(HcalElectronicsId(34, 11, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
  vhashC34.push_back(HcalElectronicsId(34, 12, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
  _filter_C34.initialize(filter::fPreserver, hcaldqm::hashfunctions::fCrateSlot, vhashC34);

  std::vector<std::pair<int, int> > timingChannels;
  timingChannels.push_back(std::pair<int, int>(28, 63));
  timingChannels.push_back(std::pair<int, int>(28, 65));
  timingChannels.push_back(std::pair<int, int>(20, 63));
  timingChannels.push_back(std::pair<int, int>(20, 65));
  for (int iChan = 0; iChan < 4; ++iChan) {
    std::vector<uint32_t> vhashTimingChannel;
    for (int depth = 1; depth <= 7; ++depth) {
      vhashTimingChannel.push_back(
          HcalDetId(HcalEndcap, timingChannels[iChan].first, timingChannels[iChan].second, depth));
    }
    _filter_timingChannels[iChan].initialize(filter::fPreserver, hcaldqm::hashfunctions::fDChannel, vhashTimingChannel);
  }

  //	INITIALIZE what you need

  // EChannel plots, online+local only
  if (_ptype != fOffline) {
    unsigned int itr = 0;
    for (unsigned int crate = 34; crate <= 34; ++crate) {
      for (unsigned int slot = 11; slot <= 12; ++slot) {
        std::vector<uint32_t> vhashSlot;
        vhashSlot.push_back(HcalElectronicsId(crate, slot, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
        _filter_slot[itr].initialize(filter::fPreserver, hashfunctions::fCrateSlot, vhashSlot);
        _cShapeCut_EChannel[itr].initialize(_name,
                                            "ShapeCut",
                                            hcaldqm::hashfunctions::fEChannel,
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_400000));
        _cLETDCvsTS_EChannel[itr].initialize(_name,
                                             "LETDCvsTS",
                                             hcaldqm::hashfunctions::fEChannel,
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                             0);
        _cLETDCTime_EChannel[itr].initialize(_name,
                                             "LETDCTime",
                                             hcaldqm::hashfunctions::fEChannel,
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250),
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                             0);
        for (unsigned int j = 0; j < 10; j++) {
          _cLETDCvsADC_EChannel[j][itr].initialize(
              _name,
              "LETDCvsADC",
              hcaldqm::hashfunctions::fEChannel,
              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
              0);
          _cADC_EChannel[j][itr].initialize(_name,
                                            "ADC",
                                            hcaldqm::hashfunctions::fEChannel,
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                            0);
          _cLETDC_EChannel[j][itr].initialize(_name,
                                              "LETDC",
                                              hcaldqm::hashfunctions::fEChannel,
                                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
                                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                              0);
        }
        ++itr;
      }
    }
  }
  _cShapeCut.initialize(_name,
                        "ShapeCut",
                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_400000));
  _cLETDCvsADC.initialize(_name,
                          "LETDCvsADC",
                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                          0);
  _cLETDCTimevsADC.initialize(_name,
                              "LETDCTimevsADC",
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                              0);
  _cLETDC.initialize(_name,
                     "LETDC",
                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                     0);
  _cADC.initialize(_name,
                   "ADC",
                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                   0);

  if (_ptype != fOffline) {
    unsigned int itr = 0;
    std::map<std::pair<unsigned int, unsigned int>, unsigned int> itr_map;
    for (unsigned int crate = 34; crate <= 34; ++crate) {
      for (unsigned int slot = 11; slot <= 12; ++slot) {
        char aux[100];
        sprintf(aux, "/Crate%d_Slot%d", crate, slot);
        _cShapeCut_EChannel[itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
        _cLETDCvsTS_EChannel[itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
        _cLETDCTime_EChannel[itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
        for (unsigned int j = 0; j < 10; j++) {
          char aux2[100];
          sprintf(aux2, "/Crate%d_Slot%d/TS%d", crate, slot, j);
          _cLETDCvsADC_EChannel[j][itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux2);
          _cLETDC_EChannel[j][itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux2);
          _cADC_EChannel[j][itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux2);
        }
        itr_map[std::make_pair(crate, slot)] = itr;
        ++itr;
      }
    }
  }
  _cShapeCut.book(ib, _subsystem);
  _cLETDCvsADC.book(ib, _subsystem);
  _cLETDCTimevsADC.book(ib, _subsystem);
  _cLETDC.book(ib, _subsystem);
  _cADC.book(ib, _subsystem);

  _ehashmap.initialize(_emap, electronicsmap::fD2EHashMap, _filter_C34);
}

/* virtual */ void QIE11Task::dqmEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  //	finish
  DQTask::dqmEndLuminosityBlock(lb, es);
}

/* virtual */ void QIE11Task::_process(edm::Event const& e, edm::EventSetup const&) {
  edm::Handle<QIE11DigiCollection> cqie11;
  if (!e.getByToken(_tokQIE11, cqie11))
    return;

  for (uint32_t i = 0; i < cqie11->size(); i++) {
    QIE11DataFrame frame = static_cast<QIE11DataFrame>((*cqie11)[i]);
    DetId did = frame.detid();
    if (HcalDetId(frame.detid()).subdet() != HcalEndcap) {
      continue;
    }

    HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));
    if (!eid.rawId()) {
      continue;
    }
    int fakecrate = -1;
    if (eid.crateId() == 34)
      fakecrate = 0;
    int index = fakecrate * 12 + (eid.slot() - 10) - 1;

    //	compute the signal, ped subracted
    CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE11DataFrame>(_dbService, did, frame);
    //		double q = hcaldqm::utilities::aveTS_v10<QIE11DataFrame>(frame,
    //			constants::adc2fC[_ped], 0, frame.samples()-1);

    //	iterate thru all TS and fill
    for (int j = 0; j < frame.samples(); j++) {
      double q_pedsub = hcaldqm::utilities::adc2fCDBMinusPedestal<QIE11DataFrame>(_dbService, digi_fC, did, frame, j);
      if (_ptype != fOffline) {
        if (index == 0 || index == 1) {
          //	shapes are after the cut
          _cShapeCut_EChannel[index].fill(eid, j, q_pedsub);
          _cLETDCvsTS_EChannel[index].fill(eid, j, frame[j].tdc());

          //	w/o a cut
          _cLETDCvsADC_EChannel[j][index].fill(eid, frame[j].adc(), frame[j].tdc());
          _cLETDC_EChannel[j][index].fill(eid, frame[j].tdc());
          if (frame[j].tdc() < 50) {
            // Each TDC count is 0.5 ns.
            // tdc == 62 or 63 means value was below or above threshold for whole time slice.
            _cLETDCTime_EChannel[index].fill(eid, j * 25. + (frame[j].tdc() / 2.));
          }
          _cADC_EChannel[j][index].fill(eid, frame[j].adc());
        }
      }
      _cShapeCut.fill(eid, j, q_pedsub);

      _cLETDCvsADC.fill(frame[j].adc(), frame[j].tdc());
      if (frame[j].tdc() < 50) {
        _cLETDCTimevsADC.fill(frame[j].adc(), j * 25. + (frame[j].tdc() / 2.));
      }
      _cLETDC.fill(eid, frame[j].tdc());

      _cADC.fill(eid, frame[j].adc());
    }
  }
}

/* virtual */ bool QIE11Task::_isApplicable(edm::Event const& e) {
  if (_ptype != fOnline || (_laserType < 0 && _eventType < 0))
    return true;
  else {
    //      fOnline mode
    edm::Handle<HcalUMNioDigi> cumn;
    if (!e.getByToken(_tokuMN, cumn))
      return false;

    //      event type check first
    int eventType = cumn->eventType();
    if (eventType == _eventType)
      return true;

    //      check if this analysis task is of the right laser type
    int laserType = cumn->valueUserWord(0);
    if (laserType == _laserType)
      return true;
  }

  return false;
}

/* virtual */ void QIE11Task::_resetMonitors(hcaldqm::UpdateFreq) {}

DEFINE_FWK_MODULE(QIE11Task);
