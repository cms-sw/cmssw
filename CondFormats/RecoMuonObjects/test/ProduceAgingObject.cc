// -*- C++ -*-
//
// Class:      ProduceAgingObject
//
//
// Original Author:  Sunil Bansal
//         Created:  Wed, 29 Jun 2016 16:27:31 GMT
//
//

// system include files
#include <memory>
#include <regex>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include "CondFormats/RecoMuonObjects/interface/MuonSystemAging.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

//
// Class declaration
//

class ProduceAgingObject : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit ProduceAgingObject(const edm::ParameterSet&);
  ~ProduceAgingObject() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override{};
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override{};
  void endJob() override{};

  void createRpcAgingMap();
  void createDtAgingMap(const edm::ESHandle<DTGeometry>& dtGeom);
  void createCscAgingMap(const edm::ESHandle<CSCGeometry>& cscGeom);
  void printAgingMap(const std::map<uint32_t, float>& map, const std::string& type) const;

  // -- member data --

  std::vector<std::string> m_RPCRegEx;
  std::map<uint32_t, float> m_RPCChambEffs;

  std::vector<std::string> m_DTRegEx;
  std::map<uint32_t, float> m_DTChambEffs;

  std::vector<std::string> m_CSCRegEx;
  std::map<uint32_t, std::pair<uint32_t, float>> m_CSCChambEffs;

  std::map<uint32_t, float> m_GEMChambEffs;
  std::map<uint32_t, float> m_ME0ChambEffs;
};

//
// Constructors and destructor
//

ProduceAgingObject::ProduceAgingObject(const edm::ParameterSet& iConfig)

{
  m_DTRegEx = iConfig.getParameter<std::vector<std::string>>("dtRegEx");
  m_RPCRegEx = iConfig.getParameter<std::vector<std::string>>("rpcRegEx");
  m_CSCRegEx = iConfig.getParameter<std::vector<std::string>>("cscRegEx");

  for (auto gemId : iConfig.getParameter<std::vector<int>>("maskedGEMIDs")) {
    m_GEMChambEffs[gemId] = 0.;
  }

  for (auto gemId : iConfig.getParameter<std::vector<int>>("maskedME0IDs")) {
    m_ME0ChambEffs[gemId] = 0.;
  }
}

ProduceAgingObject::~ProduceAgingObject() {}

//
// Member Functions
//

// -- Called for each event --
void ProduceAgingObject::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  MuonSystemAging* muonAgingObject = new MuonSystemAging();

  muonAgingObject->m_DTChambEffs = m_DTChambEffs;
  muonAgingObject->m_RPCChambEffs = m_RPCChambEffs;
  muonAgingObject->m_CSCChambEffs = m_CSCChambEffs;

  muonAgingObject->m_GEMChambEffs = m_GEMChambEffs;
  muonAgingObject->m_ME0ChambEffs = m_ME0ChambEffs;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable())
    poolDbService->writeOne(muonAgingObject, poolDbService->currentTime(), "MuonSystemAgingRcd");
}

// -- Called at the beginning of each run --
void ProduceAgingObject::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  edm::ESHandle<DTGeometry> dtGeom;
  iSetup.get<MuonGeometryRecord>().get(dtGeom);

  edm::ESHandle<CSCGeometry> cscGeom;
  iSetup.get<MuonGeometryRecord>().get(cscGeom);

  createDtAgingMap(dtGeom);
  createCscAgingMap(cscGeom);
  createRpcAgingMap();

  printAgingMap(m_GEMChambEffs, "GEM");
  printAgingMap(m_ME0ChambEffs, "ME0");
}

/// -- Create RPC aging map --
void ProduceAgingObject::createRpcAgingMap() {
  std::cout << "[ProduceAgingObject] List of aged RPC objects (ID, efficiency)" << std::endl;
  for (auto& chRegExStr : m_RPCRegEx) {
    std::string id = chRegExStr.substr(0, chRegExStr.find(':'));
    std::string eff = chRegExStr.substr(id.size() + 1, chRegExStr.find(':'));

    std::cout << "\t( " << id << " , " << eff << " )" << std::endl;
    m_RPCChambEffs[std::atoi(id.c_str())] = std::atof(eff.c_str());
  }
}

/// -- Create DT aging map ------------
void ProduceAgingObject::createDtAgingMap(const edm::ESHandle<DTGeometry>& dtGeom) {
  const std::vector<const DTChamber*> chambers = dtGeom->chambers();

  std::cout << "[ProduceAgingObject] List of aged DT chambers (ChamberID, efficiency)" << std::endl;
  for (const DTChamber* ch : chambers) {
    DTChamberId chId = ch->id();

    std::string chTag = "WH" + std::to_string(chId.wheel()) + "_ST" + std::to_string(chId.station()) + "_SEC" +
                        std::to_string(chId.sector());

    float eff = 1.;

    for (auto& chRegExStr : m_DTRegEx) {
      std::string effTag(chRegExStr.substr(chRegExStr.find(':')));

      const std::regex chRegEx(chRegExStr.substr(0, chRegExStr.find(':')));
      const std::regex effRegEx("(\\d*\\.\\d*)");

      std::smatch effMatch;

      if (std::regex_search(chTag, chRegEx) && std::regex_search(effTag, effMatch, effRegEx)) {
        std::string effStr = effMatch.str();
        eff = std::atof(effStr.c_str());
      }
    }

    if (eff < 1.) {
      std::cout << "\t(" << chId << ", " << eff << " )" << std::endl;
      m_DTChambEffs[chId.rawId()] = eff;
    }
  }
}

/// -- Create CSC aging map ------------
void ProduceAgingObject::createCscAgingMap(const edm::ESHandle<CSCGeometry>& cscGeom) {
  const auto chambers = cscGeom->chambers();

  std::cout << "[ProduceAgingObject] List of aged CSC chambers (ChamberID, efficiency, type)" << std::endl;

  for (const auto* ch : chambers) {
    CSCDetId chId = ch->id();

    std::string chTag = (chId.zendcap() == 1 ? "ME+" : "ME-") + std::to_string(chId.station()) + "/" +
                        std::to_string(chId.ring()) + "/" + std::to_string(chId.chamber());

    int type = 0;
    float eff = 1.;

    for (auto& chRegExStr : m_CSCRegEx) {
      int loc = chRegExStr.find(':');
      // if there's no :, then we don't have to correct format
      if (loc < 0)
        continue;

      std::string effTag(chRegExStr.substr(loc));

      const std::regex chRegEx(chRegExStr.substr(0, chRegExStr.find(':')));
      const std::regex predicateRegEx("(\\d*,\\d*\\.\\d*)");

      std::smatch predicate;

      if (std::regex_search(chTag, chRegEx) && std::regex_search(effTag, predicate, predicateRegEx)) {
        std::string predicateStr = predicate.str();
        std::string typeStr = predicateStr.substr(0, predicateStr.find(','));
        std::string effStr = predicateStr.substr(predicateStr.find(',') + 1);
        type = std::atoi(typeStr.c_str());
        eff = std::atof(effStr.c_str());

        std::cout << "\t( " << chTag << " , " << eff << " , " << type << " )" << std::endl;
      }
    }

    // Note, layer 0 for chamber specification
    int rawId = chId.rawIdMaker(chId.endcap(), chId.station(), chId.ring(), chId.chamber(), 0);
    m_CSCChambEffs[rawId] = std::make_pair(type, eff);
  }
}
void ProduceAgingObject::printAgingMap(const std::map<uint32_t, float>& map, const std::string& type) const {
  std::cout << "[ProduceAgingObject] List of aged " << type << " objects (ID, efficiency)" << std::endl;

  std::map<uint32_t, float>::const_iterator mapObj = map.begin();
  std::map<uint32_t, float>::const_iterator mapEnd = map.end();

  for (; mapObj != mapEnd; ++mapObj) {
    std::cout << "\t( " << mapObj->first << " , " << mapObj->second << " )" << std::endl;
  }
}

/// -- Fill 'descriptions' with the allowed parameters for the module  --
void ProduceAgingObject::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("dtRegEx", {});
  desc.add<std::vector<std::string>>("rpcRegEx", {});
  desc.add<std::vector<std::string>>("cscRegEx", {});
  desc.add<std::vector<int>>("maskedGEMIDs", {});
  desc.add<std::vector<int>>("maskedME0IDs", {});

  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ProduceAgingObject);
