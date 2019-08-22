//-------------------------------------------------
//
//   Class: DTTrig
//
//   Description: Steering routine for L1 trigger simulation in
//                a muon barrel station
//
//
//   Author List:
//   C. Grandi
//   Modifications:
//   S Vanini, S. Marcellini, D. Bonacorsi,  C.Battilana
//
//   07/03/30 : configuration now through DTConfigManager SV
//-------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTrigger/interface/DTTrig.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Geometry/DTGeometry/interface/DTChamber.h"

//----------------
// Constructors --
//----------------

DTTrig::DTTrig(const edm::ParameterSet& params, edm::ConsumesCollector&& iC)
    : _inputexist(true), _configid(0), _geomid(0) {
  // Set configuration parameters
  _debug = params.getUntrackedParameter<bool>("debug");

  if (_debug) {
    std::cout << std::endl;
    std::cout << "**** Initialization of DTTrigger ****" << std::endl;
  }

  _digitag = params.getParameter<edm::InputTag>("digiTag");
  iC.consumes<DTDigiCollection>(_digitag);
}

void DTTrig::createTUs(const edm::EventSetup& iSetup) {
  // build up Sector Collectors and then
  // build the trrigger units (one for each chamber)
  for (int iwh = -2; iwh <= 2; iwh++) {
    for (int ise = 1; ise <= 12; ise++) {
      if (_debug) {
        std::cout << "calling sectcollid wh sc " << iwh << " " << ise << std::endl;
      }
      DTSectCollId scid(iwh, ise);
      {
        SC_iterator it = _cache1.find(scid);
        if (it != _cache1.end()) {
          if (_debug) {
            std::cout << "DTTrig::createTUs: Sector Collector unit already exists" << std::endl;
          }
          continue;
        }
      }
      {
        auto element = _cache1.emplace(scid, scid);
        if (_debug) {
          std::cout << " DTTrig::createTUs new SC sc = " << &(element.first->second) << " at scid.sector() "
                    << scid.sector() << " at scid.wheel() " << scid.wheel() << std::endl;
        }
      }
    }
  }

  edm::ESHandle<DTGeometry> dtGeom;
  iSetup.get<MuonGeometryRecord>().get(dtGeom);
  for (std::vector<const DTChamber*>::const_iterator ich = dtGeom->chambers().begin(); ich != dtGeom->chambers().end();
       ich++) {
    const DTChamber* chamb = (*ich);
    DTChamberId chid = chamb->id();
    TU_iterator it = _cache.find(chid);
    if (it != _cache.end()) {
      if (_debug)
        std::cout << "DTTrig::init: Trigger unit already exists" << std::endl;
      continue;
    }

    auto info = _cache.emplace(chid, chamb);
    auto tru = &(info.first->second);

    //----------- add TU to corresponding SC
    // returning correspondent SC id
    DTSectCollId scid;
    if (chid.sector() == 13) {
      scid = DTSectCollId(chid.wheel(), 4);
    } else if (chid.sector() == 14) {
      scid = DTSectCollId(chid.wheel(), 10);
    } else {
      scid = DTSectCollId(chid.wheel(), chid.sector());
    }

    SC_iterator it1 = _cache1.find(scid);

    if (it1 != _cache1.end()) {
      auto& sc = (*it1).second;
      if (_debug) {
        std::cout << "DTTrig::init:  adding TU in SC << "
                  << " sector = " << scid.sector() << " wheel = " << scid.wheel() << std::endl;
      }
      sc.addTU(tru);
    } else {
      std::cout << "DTTrig::createTUs: Trigger Unit not in the map: ";
    }
  }
}

void DTTrig::triggerReco(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  updateES(iSetup);
  if (!_inputexist)
    return;

  DTDigiMap digiMap;
  //Sort digis by chamber so they can be used by BTIs
  edm::Handle<DTDigiCollection> dtDigis;
  iEvent.getByLabel(_digitag, dtDigis);

  if (!dtDigis.isValid()) {
    LogDebug("DTTrig") << "DTTrig::triggerReco DTDigiCollection  with input tag " << _digitag
                       << "requested in configuration, but not found in the event." << std::endl;
    _inputexist = false;
    return;
  }

  DTDigiCollection::DigiRangeIterator detUnitIt;

  for (detUnitIt = dtDigis->begin(); detUnitIt != dtDigis->end(); ++detUnitIt) {
    const DTLayerId& layId = (*detUnitIt).first;
    const DTChamberId chambId = layId.superlayerId().chamberId();
    const DTDigiCollection::Range& range = (*detUnitIt).second;
    digiMap[chambId].put(range, layId);
  }

  //Run reconstruct for single trigger subsystem (Bti, Traco TS)
  for (TU_iterator it = _cache.begin(); it != _cache.end(); it++) {
    DTSCTrigUnit& thisTU = (*it).second;
    if (thisTU.BtiTrigs()->size() > 0) {
      thisTU.BtiTrigs()->clearCache();
      thisTU.TSThTrigs()->clearCache();
      thisTU.TracoTrigs()->clearCache();
      thisTU.TSPhTrigs()->clearCache();
    }
    DTChamberId chid = thisTU.statId();
    DTDigiMap_iterator dmit = digiMap.find(chid);
    if (dmit != digiMap.end()) {
      thisTU.BtiTrigs()->reconstruct((*dmit).second);
      if (thisTU.BtiTrigs()->size() > 0) {
        thisTU.TSThTrigs()->reconstruct();
        thisTU.TracoTrigs()->reconstruct();
        if (thisTU.TracoTrigs()->size() > 0)
          thisTU.TSPhTrigs()->reconstruct();
      }
    }
  }
  //Run reconstruct for Sector Collector
  for (SC_iterator it = _cache1.begin(); it != _cache1.end(); it++) {
    DTSectColl& sectcoll = (*it).second;
    DTSectCollId scid = (*it).first;
    if (sectcoll.sizePh() > 0 || sectcoll.sizeTh() > 0)
      sectcoll.clearCache();
    bool mustreco = false;
    for (int i = 1; i < 5; i++) {
      if (sectcoll.getTSPhi(i)->size() > 0) {
        mustreco = true;
        break;
      }
    }
    for (int i = 1; i < 4; i++) {
      if (sectcoll.getTSTheta(i)->size() > 0) {
        mustreco = true;
        break;
      }
    }
    if (scid.sector() == 4 || scid.sector() == 10) {
      if (sectcoll.getTSPhi(5)->size() > 0)
        mustreco = true;
    }
    if (mustreco)
      sectcoll.reconstruct();
  }
}

void DTTrig::updateES(const edm::EventSetup& iSetup) {
  // Check for updatets in config
  edm::ESHandle<DTConfigManager> confHandle;
  edm::ESHandle<DTGeometry> geomHandle;

  if (iSetup.get<DTConfigManagerRcd>().cacheIdentifier() != _configid) {
    if (_debug)
      std::cout << "DTTrig::updateES updating DTTPG configuration" << std::endl;

    _configid = iSetup.get<DTConfigManagerRcd>().cacheIdentifier();
    iSetup.get<DTConfigManagerRcd>().get(confHandle);
    _conf_manager = confHandle.product();
    for (TU_iterator it = _cache.begin(); it != _cache.end(); it++) {
      (*it).second.setConfig(_conf_manager);
    }
    for (SC_iterator it = _cache1.begin(); it != _cache1.end(); it++) {
      (*it).second.setConfig(_conf_manager);
    }
  }

  if (iSetup.get<MuonGeometryRecord>().cacheIdentifier() != _configid) {
    if (_debug)
      std::cout << "DTTrig::updateES updating muon geometry" << std::endl;

    _geomid = iSetup.get<MuonGeometryRecord>().cacheIdentifier();
    iSetup.get<MuonGeometryRecord>().get(geomHandle);
    for (TU_iterator it = _cache.begin(); it != _cache.end(); it++) {
      (*it).second.setGeom(geomHandle->chamber((*it).second.statId()));
    }
  }
}

void DTTrig::clear() {
  // Delete the map
  _cache.clear();
  _cache1.clear();
}

DTSCTrigUnit* DTTrig::trigUnit(DTChamberId chid) { /*check();*/
  return const_cast<DTSCTrigUnit*>(constTrigUnit(chid));
}

DTSCTrigUnit const* DTTrig::constTrigUnit(DTChamberId chid) const {
  //    std::cout << " SC: running DTTrig::constTrigUnit(DTChamberId chid)" << std::endl;
  TU_const_iterator it = _cache.find(chid);
  if (it == _cache.end()) {
    std::cout << "DTTrig::trigUnit: Trigger Unit not in the map: ";
    std::cout << " wheel=" << chid.wheel();
    std::cout << ", station=" << chid.station();
    std::cout << ", sector=" << chid.sector();
    std::cout << std::endl;
    return nullptr;
  }

  return &(*it).second;
}

DTSectColl const* DTTrig::SCUnit(DTSectCollId scid) const {
  SC_const_iterator it = _cache1.find(scid);
  if (it == _cache1.end()) {
    std::cout << "DTTrig::SCUnit: Trigger Unit not in the map: ";
    std::cout << " wheel=" << scid.wheel();
    std::cout << ", sector=" << scid.sector();
    std::cout << std::endl;
    return nullptr;
  }

  return &(*it).second;
}

DTSCTrigUnit* DTTrig::trigUnit(int wheel, int stat, int sect) {
  return const_cast<DTSCTrigUnit*>(constTrigUnit(wheel, stat, sect));
}

DTSectColl const* DTTrig::SCUnit(int wheel, int sect) const {
  sect++;
  return SCUnit(DTSectCollId(wheel, sect));
}

DTSCTrigUnit const* DTTrig::constTrigUnit(int wheel, int stat, int sect) const {
  sect++;  // offset 1 for sector number ([0,11] --> [1,12])
  return constTrigUnit(DTChamberId(wheel, stat, sect));
}

DTChambPhSegm* DTTrig::chPhiSegm1(DTSCTrigUnit* unit, int step) {
  if (unit == nullptr)
    return nullptr;
  if (unit->nPhiSegm(step) < 1)
    return nullptr;
  return const_cast<DTChambPhSegm*>(unit->phiSegment(step, 1));
}

DTChambPhSegm* DTTrig::chPhiSegm2(DTSCTrigUnit* unit, int step) {
  if (unit == nullptr)
    return nullptr;
  if (unit->nPhiSegm(step) < 2)
    return nullptr;
  return const_cast<DTChambPhSegm*>(unit->phiSegment(step, 2));
}

DTChambThSegm* DTTrig::chThetaSegm(DTSCTrigUnit* unit, int step) {
  if (unit == nullptr)
    return nullptr;
  if (unit->nThetaSegm(step) < 1)
    return nullptr;
  return const_cast<DTChambThSegm*>(unit->thetaSegment(step, 1));
}

DTChambPhSegm* DTTrig::chPhiSegm1(DTChamberId sid, int step) { return chPhiSegm1(trigUnit(sid), step); }

DTChambPhSegm* DTTrig::chPhiSegm2(DTChamberId sid, int step) { return chPhiSegm2(trigUnit(sid), step); }

DTChambThSegm* DTTrig::chThetaSegm(DTChamberId sid, int step) {
  if (sid.station() == 4)
    return nullptr;
  return chThetaSegm(trigUnit(sid), step);
}

DTChambPhSegm* DTTrig::chPhiSegm1(int wheel, int stat, int sect, int step) {
  return chPhiSegm1(trigUnit(wheel, stat, sect), step);
  // to make it transparent to the outside world
  //  return chSectCollSegm1(wheel,stat,sect,step);
}

DTChambPhSegm* DTTrig::chPhiSegm2(int wheel, int stat, int sect, int step) {
  //  if(stat==4&&(sect==3||sect==9)) {
  // if hrizontal chambers of MB4 get first track of twin chamber (flag=1)
  //   return chPhiSegm1(trigUnit(wheel,stat,sect,1),step);
  //  } else {
  return chPhiSegm2(trigUnit(wheel, stat, sect), step);
  // to make it transparent to the outside world
  // return chSectCollSegm2(wheel,stat,sect,step);
  //}
}

DTChambThSegm* DTTrig::chThetaSegm(int wheel, int stat, int sect, int step) {
  if (stat == 4)
    return nullptr;
  return chThetaSegm(trigUnit(wheel, stat, sect), step);
}

// SM sector collector section
DTSectCollPhSegm* DTTrig::chSectCollPhSegm1(DTSectColl* unit, int step) {
  if (unit == nullptr)
    return nullptr;
  if (unit->nSegmPh(step) < 1)
    return nullptr;
  return const_cast<DTSectCollPhSegm*>(unit->SectCollPhSegment(step, 1));
}

DTSectCollPhSegm* DTTrig::chSectCollPhSegm2(DTSectColl* unit, int step) {
  if (unit == nullptr)
    return nullptr;
  if (unit->nSegmPh(step) < 2)
    return nullptr;
  return const_cast<DTSectCollPhSegm*>(unit->SectCollPhSegment(step, 2));
}

DTSectCollPhSegm* DTTrig::chSectCollPhSegm1(int wheel, int sect, int step) {
  return chSectCollPhSegm1(const_cast<DTSectColl*>(SCUnit(wheel, sect)), step);
}

DTSectCollPhSegm* DTTrig::chSectCollPhSegm2(int wheel, int sect, int step) {
  //  if(stat==4&&(sect==3||sect==9)) {
  // if hrizontal chambers of MB4 get first track of twin chamber (flag=1)
  //return chSectCollSegm1(trigUnit(wheel,stat,sect,1),step);
  //} else {
  return chSectCollPhSegm2(const_cast<DTSectColl*>(SCUnit(wheel, sect)), step);
  //}
}

DTSectCollThSegm* DTTrig::chSectCollThSegm(DTSectColl* unit, int step) {
  if (unit == nullptr)
    return nullptr;
  if (unit->nSegmTh(step) < 1)
    return nullptr;
  return const_cast<DTSectCollThSegm*>(unit->SectCollThSegment(step));
}

DTSectCollThSegm* DTTrig::chSectCollThSegm(int wheel, int sect, int step) {
  return chSectCollThSegm(const_cast<DTSectColl*>(SCUnit(wheel, sect)), step);
}

// end SM

void DTTrig::dumpGeom() const {
  /*check();*/
  for (TU_const_iterator it = _cache.begin(); it != _cache.end(); it++) {
    ((*it).second).dumpGeom();
  }
}

void DTTrig::dumpLuts(short int lut_btic, const DTConfigManager* conf) const {
  for (TU_const_iterator it = _cache.begin(); it != _cache.end(); it++) {
    const DTSCTrigUnit& thisTU = (*it).second;

    // dump lut command file from geometry
    thisTU.dumpLUT(lut_btic);

    // dump lut command file from parameters (DB or CMSSW)
    DTChamberId chid = thisTU.statId();
    conf->dumpLUTParam(chid);
  }

  return;
}

std::vector<DTBtiTrigData> DTTrig::BtiTrigs() const {
  /*check();*/
  std::vector<DTBtiTrigData> trigs;
  for (auto ptu = _cache.begin(); ptu != _cache.end(); ptu++) {
    const DTSCTrigUnit& tu = (*ptu).second;
    auto peb = tu.BtiTrigs()->end();
    for (auto p = tu.BtiTrigs()->begin(); p != peb; p++) {
      trigs.push_back(*p);
    }
  }
  return trigs;
}

std::vector<DTTracoTrigData> DTTrig::TracoTrigs() const {
  std::vector<DTTracoTrigData> trigs;
  /*check();*/
  for (auto ptu = _cache.begin(); ptu != _cache.end(); ptu++) {
    const DTSCTrigUnit& tu = (*ptu).second;
    auto peb = tu.TracoTrigs()->end();
    for (auto p = tu.TracoTrigs()->begin(); p != peb; p++) {
      trigs.push_back(*p);
    }
  }
  return trigs;
}

std::vector<DTChambPhSegm> DTTrig::TSPhTrigs() const {
  /*check();*/
  std::vector<DTChambPhSegm> trigs;
  for (auto ptu = _cache.begin(); ptu != _cache.end(); ptu++) {
    const DTSCTrigUnit& tu = (*ptu).second;
    auto peb = tu.TSPhTrigs()->end();
    for (auto p = tu.TSPhTrigs()->begin(); p != peb; p++) {
      trigs.push_back(*p);
    }
  }
  return trigs;
}

std::vector<DTChambThSegm> DTTrig::TSThTrigs() const {
  /*check();*/
  std::vector<DTChambThSegm> trigs;
  for (auto ptu = _cache.begin(); ptu != _cache.end(); ptu++) {
    const DTSCTrigUnit& tu = (*ptu).second;
    auto peb = tu.TSThTrigs()->end();
    for (auto p = tu.TSThTrigs()->begin(); p != peb; p++) {
      trigs.push_back(*p);
    }
  }
  return trigs;
}

std::vector<DTSectCollPhSegm> DTTrig::SCPhTrigs() const {
  /*check();*/
  std::vector<DTSectCollPhSegm> trigs;
  for (auto psc = _cache1.begin(); psc != _cache1.end(); psc++) {
    //    DTSCTrigUnit* tu = (*ptu).second;
    //
    // old SMDB:
    //      DTSectColl* tu = (*ptu).second;
    //      std::vector<DTChambPhSegm>::const_iterator p=0;
    //      std::vector<DTChambPhSegm>::const_iterator peb=tu->SCTrigs()->end();
    //      for(p=tu->SCTrigs()->begin();p!=peb;p++){
    //        trigs.push_back(*p);
    //      }

    const DTSectColl& sc = (*psc).second;
    auto peb = sc.endPh();
    for (auto p = sc.beginPh(); p != peb; p++) {
      trigs.push_back(*p);
    }
  }
  return trigs;
}

std::vector<DTSectCollThSegm> DTTrig::SCThTrigs() const {
  /*check();*/
  std::vector<DTSectCollThSegm> trigs;
  for (auto psc = _cache1.begin(); psc != _cache1.end(); psc++) {
    const DTSectColl& sc = (*psc).second;
    auto peb = sc.endTh();
    for (auto p = sc.beginTh(); p != peb; p++) {
      trigs.push_back(*p);
    }
  }
  return trigs;
}
