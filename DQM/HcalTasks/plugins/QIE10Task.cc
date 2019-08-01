
#include "DQM/HcalTasks/interface/QIE10Task.h"
#include "FWCore/Framework/interface/Run.h"
#include <map>

using namespace hcaldqm;
using namespace hcaldqm::constants;
QIE10Task::QIE10Task(edm::ParameterSet const& ps) : DQTask(ps) {
  //	tags
  _tagQIE10 = ps.getUntrackedParameter<edm::InputTag>("tagQIE10", edm::InputTag("hcalDigis"));
  _tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF", edm::InputTag("hcalDigis"));
  _tokQIE10 = consumes<QIE10DigiCollection>(_tagQIE10);
  _tokHF = consumes<HFDigiCollection>(_tagHF);

  //	cuts
  _cut = ps.getUntrackedParameter<double>("cut", 50.0);
  _ped = ps.getUntrackedParameter<int>("ped", 4);
}
/* virtual */ void QIE10Task::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  if (_ptype == fLocal)
    if (r.runAuxiliary().run() == 1)
      return;

  DQTask::bookHistograms(ib, r, es);

  //	GET WHAT YOU NEED
  edm::ESHandle<HcalDbService> dbs;
  es.get<HcalDbRecord>().get(dbs);
  _emap = dbs->getHcalMapping();

  unsigned int nTS = _ptype == fLocal ? 10 : 6;

  // create a slot filter and initialize what you need
  unsigned int itr = 0;
  for (auto& crate : constants::crateListHF) {
    for (unsigned int slot = SLOT_uTCA_MIN; slot <= SLOT_uTCA_MAX; ++slot) {
      std::vector<uint32_t> vhashSlot;
      vhashSlot.push_back(HcalElectronicsId(crate, slot, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
      _filter_slot[itr].initialize(filter::fPreserver, hashfunctions::fCrateSlot, vhashSlot);

      _cShapeCut_EChannel[itr].initialize(_name,
                                          "ShapeCut",
                                          hcaldqm::hashfunctions::fEChannel,
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_400000));
      _cLETDCTime_EChannel[itr].initialize(_name,
                                           "LETDCTime",
                                           hcaldqm::hashfunctions::fEChannel,
                                           new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250),
                                           new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
      _cLETDCvsTS_EChannel[itr].initialize(_name,
                                           "TDCvsTS",
                                           hcaldqm::hashfunctions::fEChannel,
                                           new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
                                           new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
                                           new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));

      for (unsigned int j = 0; j < nTS; j++) {
        _cLETDCvsADC_EChannel[j][itr].initialize(_name,
                                                 "LETDCvsADC",
                                                 hcaldqm::hashfunctions::fEChannel,
                                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
                                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
                                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
        _cADC_EChannel[j][itr].initialize(_name,
                                          "ADC",
                                          hcaldqm::hashfunctions::fEChannel,
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
        _cLETDC_EChannel[j][itr].initialize(_name,
                                            "LETDC",
                                            hcaldqm::hashfunctions::fEChannel,
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
      }
      ++itr;
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
                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
  _cLETDCTimevsADC.initialize(_name,
                              "LETDCTimevsADC",
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
  _cLETDC.initialize(_name,
                     "LETDC",
                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
  _cADC.initialize(_name,
                   "ADC",
                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));

  //OCCUPANCY PER CRATE/SLOT
  _cOccupancy_Crate.initialize(_name,
                               "Occupancy",
                               hashfunctions::fCrate,
                               new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
                               new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
                               new quantity::ValueQuantity(quantity::fN));
  _cOccupancy_CrateSlot.initialize(_name,
                                   "Occupancy",
                                   hashfunctions::fCrateSlot,
                                   new quantity::ElectronicsQuantity(quantity::fFiberuTCA),
                                   new quantity::ElectronicsQuantity(quantity::fFiberCh),
                                   new quantity::ValueQuantity(quantity::fN));

  // OCCUPANCY IN DETECTOR COORDINATES
  _cOccupancy_depth.initialize(_name,
                               "Occupancy",
                               hcaldqm::hashfunctions::fdepth,
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));

  itr = 0;
  for (auto& crate : constants::crateListHF) {
    for (unsigned int slot = SLOT_uTCA_MIN; slot <= SLOT_uTCA_MAX; ++slot) {
      char aux[100];
      sprintf(aux, "/Crate%d_Slot%d", crate, slot);
      _cShapeCut_EChannel[itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
      _cLETDCTime_EChannel[itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
      _cLETDCvsTS_EChannel[itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
      for (unsigned int i = 0; i < nTS; i++) {
        char aux2[100];
        sprintf(aux2, "/Crate%d_Slot%d/TS%d", crate, slot, i);
        _cLETDCvsADC_EChannel[i][itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
        _cLETDC_EChannel[i][itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
        _cADC_EChannel[i][itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
      }
      ++itr;
    }
  }

  _cShapeCut.book(ib, _subsystem);
  _cLETDCvsADC.book(ib, _subsystem);
  _cLETDCTimevsADC.book(ib, _subsystem);
  _cLETDC.book(ib, _subsystem);
  _cADC.book(ib, _subsystem);

  _cOccupancy_Crate.book(ib, _emap, _subsystem);
  _cOccupancy_CrateSlot.book(ib, _emap, _subsystem);
  _cOccupancy_depth.book(ib, _emap, _subsystem);

  _ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
}

/* virtual */ void QIE10Task::dqmEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  //	finish
  DQTask::dqmEndLuminosityBlock(lb, es);
}

/* virtual */ void QIE10Task::_process(edm::Event const& e, edm::EventSetup const&) {
  edm::Handle<QIE10DigiCollection> cqie10;
  edm::Handle<HFDigiCollection> chf;
  if (!e.getByToken(_tokQIE10, cqie10))
    return;
  if (!e.getByToken(_tokHF, chf))
    _logger.dqmthrow("Collection HFDigiCollection isn't available" + _tagHF.label() + " " + _tagHF.instance());

  std::map<uint32_t, QIE10DataFrame> mqie10;
  for (uint32_t i = 0; i < cqie10->size(); i++) {
    QIE10DataFrame frame = static_cast<QIE10DataFrame>((*cqie10)[i]);
    HcalDetId did = frame.detid();
    HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));
    if (did.subdet() != HcalForward) {
      continue;
    }

    // Compute index for EChannel plots
    int fakecrate = -1;
    if (eid.crateId() == 22) {
      fakecrate = 0;
    } else if (eid.crateId() == 29) {
      fakecrate = 1;
    } else if (eid.crateId() == 32) {
      fakecrate = 2;
    } else {
      // Unknown crate, skip digi
      continue;
    }
    int index = fakecrate * 12 + eid.slot() - 1;

    //	compute the signal, ped subracted
    CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE10DataFrame>(_dbService, did, frame);
    double sumQ = hcaldqm::utilities::sumQDB<QIE10DataFrame>(_dbService, digi_fC, did, frame, 0, frame.samples() - 1);
    // double sumQ = hcaldqm::utilities::sumQ_v10<QIE10DataFrame>(frame, constants::adc2fC[_ped], 0, frame.samples()-1);

    _cOccupancy_Crate.fill(eid);
    _cOccupancy_CrateSlot.fill(eid);
    _cOccupancy_depth.fill(did);

    //	iterate thru all TS and fill
    for (int j = 0; j < frame.samples(); j++) {
      //	shapes are after the cut
      if (sumQ > _cut) {
        double q = hcaldqm::utilities::adc2fCDBMinusPedestal<QIE10DataFrame>(_dbService, digi_fC, did, frame, j);
        _cShapeCut_EChannel[index].fill(eid, j, q);
        _cShapeCut.fill(j, q);
      }
      //	w/o a cut
      _cLETDCvsADC_EChannel[j][index].fill(eid, frame[j].adc(), frame[j].le_tdc());
      _cLETDCvsADC.fill(frame[j].adc(), frame[j].le_tdc());
      _cLETDC_EChannel[j][index].fill(eid, frame[j].le_tdc());
      _cLETDC.fill(frame[j].le_tdc());
      _cADC_EChannel[j][index].fill(eid, frame[j].adc());
      _cADC.fill(frame[j].adc());
      _cLETDCvsTS_EChannel[index].fill(eid, j, frame[j].le_tdc());

      // TDC conversion to time
      if (frame[j].le_tdc() < 50) {
        // Each TDC count is 0.5 ns.
        // tdc == 62 or 63 means value was below or above threshold for whole time slice.
        double time = j * 25. + (frame[j].le_tdc() / 2.);
        _cLETDCTime_EChannel[index].fill(eid, time);
        _cLETDCTimevsADC.fill(frame[j].adc(), time);
      }
    }

    mqie10[did.rawId()] = frame;
  }
}

/* virtual */ void QIE10Task::_resetMonitors(hcaldqm::UpdateFreq) {}

DEFINE_FWK_MODULE(QIE10Task);
