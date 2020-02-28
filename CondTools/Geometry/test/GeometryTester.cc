#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDetExtra.h"
#include "CondFormats/GeometryObjects/interface/PCaloGeometry.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"

#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PGeometricDetExtraRcd.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "Geometry/Records/interface/CSCRecoGeometryRcd.h"
#include "Geometry/Records/interface/CSCRecoDigiParametersRcd.h"
#include "Geometry/Records/interface/PEcalBarrelRcd.h"
#include "Geometry/Records/interface/PEcalEndcapRcd.h"
#include "Geometry/Records/interface/PEcalPreshowerRcd.h"
#include "Geometry/Records/interface/PHcalRcd.h"
#include "Geometry/Records/interface/PHGCalRcd.h"
#include "Geometry/Records/interface/PCaloTowerRcd.h"
#include "Geometry/Records/interface/PCastorRcd.h"
#include "Geometry/Records/interface/PZdcRcd.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

  class GeometryTester : public edm::one::EDAnalyzer<> {
  public:
    GeometryTester(edm::ParameterSet const&);
    void beginJob() override {}
    void analyze(edm::Event const&, edm::EventSetup const&) override;
    void endJob() override {}

  private:
    bool m_xmltest, m_tktest, m_ecaltest;
    bool m_hcaltest, m_hgcaltest, m_calotowertest;
    bool m_castortest, m_zdctest, m_csctest;
    bool m_dttest, m_rpctest;
    std::string m_geomLabel;
  };
}  // namespace

GeometryTester::GeometryTester(const edm::ParameterSet& iConfig) {
  m_xmltest = iConfig.getUntrackedParameter<bool>("XMLTest", true);
  m_tktest = iConfig.getUntrackedParameter<bool>("TrackerTest", true);
  m_ecaltest = iConfig.getUntrackedParameter<bool>("EcalTest", true);
  m_hcaltest = iConfig.getUntrackedParameter<bool>("HcalTest", true);
  m_hgcaltest = iConfig.getUntrackedParameter<bool>("HGCalTest", true);
  m_calotowertest = iConfig.getUntrackedParameter<bool>("CaloTowerTest", true);
  m_castortest = iConfig.getUntrackedParameter<bool>("CastorTest", true);
  m_zdctest = iConfig.getUntrackedParameter<bool>("ZDCTest", true);
  m_csctest = iConfig.getUntrackedParameter<bool>("CSCTest", true);
  m_dttest = iConfig.getUntrackedParameter<bool>("DTTest", true);
  m_rpctest = iConfig.getUntrackedParameter<bool>("RPCTest", true);
  m_geomLabel = iConfig.getUntrackedParameter<std::string>("geomLabel", "Extended");
}

void GeometryTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  if (m_xmltest) {
    edm::ESHandle<FileBlob> xmlgeo;
    iSetup.get<GeometryFileRcd>().get(m_geomLabel, xmlgeo);
    std::cout << "XML FILE\n";
    std::unique_ptr<std::vector<unsigned char> > tb = (*xmlgeo).getUncompressedBlob();
    std::cout << "SIZE FILE = " << tb->size() << "\n";
    for (auto it : *tb) {
      std::cout << it;
    }
    std::cout << "\n";
  }

  if (m_tktest) {
    edm::ESHandle<PGeometricDet> tkGeo;
    edm::ESHandle<PGeometricDetExtra> tkExtra;
    iSetup.get<IdealGeometryRecord>().get(tkGeo);
    iSetup.get<PGeometricDetExtraRcd>().get(tkExtra);
    std::cout << "TRACKER\n";

    //helper map
    std::map<uint32_t, uint32_t> diTogde;
    for (uint32_t g = 0; g < tkExtra->pgdes_.size(); ++g) {
      diTogde[tkExtra->pgdes_[g]._geographicalId] = g;
    }
    uint32_t tkeInd;
    for (auto it : tkGeo->pgeomdets_) {
      std::cout << it._params0 << it._params1 << it._params2 << it._params3 << it._params4 << it._params5 << it._params6
                << it._params7 << it._params8 << it._params9 << it._params10 << it._x << it._y << it._z << it._phi
                << it._rho << it._a11 << it._a12 << it._a13 << it._a21 << it._a22 << it._a23 << it._a31 << it._a32
                << it._a33 << it._shape << it._name << it._ns;
      tkeInd = diTogde[it._geographicalID];
      std::cout << tkExtra->pgdes_[tkeInd]._volume << tkExtra->pgdes_[tkeInd]._density
                << tkExtra->pgdes_[tkeInd]._weight << tkExtra->pgdes_[tkeInd]._copy
                << tkExtra->pgdes_[tkeInd]._material;
      std::cout << it._radLength << it._xi << it._pixROCRows << it._pixROCCols << it._pixROCx << it._pixROCy
                << it._stereo << it._siliconAPVNum << it._geographicalID << it._nt0 << it._nt1 << it._nt2 << it._nt3
                << it._nt4 << it._nt5 << it._nt6 << it._nt7 << it._nt8 << it._nt9 << it._nt10 << "\n";
    }
  }

  if (m_ecaltest) {
    edm::ESHandle<PCaloGeometry> ebgeo;
    iSetup.get<PEcalBarrelRcd>().get(ebgeo);
    std::cout << "ECAL BARREL\n";
    auto tseb = ebgeo->getTranslation();
    auto dimeb = ebgeo->getDimension();
    auto indeb = ebgeo->getIndexes();
    std::cout << "Translations " << tseb.size() << "\n";
    std::cout << "Dimensions " << dimeb.size() << "\n";
    std::cout << "Indices " << indeb.size() << "\n";
    for (auto it : tseb)
      std::cout << it;
    std::cout << "\n";
    for (auto it : dimeb)
      std::cout << it;
    std::cout << "\n";
    for (auto it : indeb)
      std::cout << it;
    std::cout << "\n";

    edm::ESHandle<PCaloGeometry> eegeo;
    iSetup.get<PEcalEndcapRcd>().get(eegeo);
    std::cout << "ECAL ENDCAP\n";
    auto tsee = eegeo->getTranslation();
    auto dimee = eegeo->getDimension();
    auto indee = eegeo->getIndexes();
    std::cout << "Translations " << tsee.size() << "\n";
    std::cout << "Dimensions " << dimee.size() << "\n";
    std::cout << "Indices " << indee.size() << "\n";
    for (auto it : tsee)
      std::cout << it;
    std::cout << "\n";
    for (auto it : dimee)
      std::cout << it;
    std::cout << "\n";
    for (auto it : indee)
      std::cout << it;
    std::cout << "\n";

    edm::ESHandle<PCaloGeometry> epgeo;
    iSetup.get<PEcalPreshowerRcd>().get(epgeo);
    std::cout << "ECAL PRESHOWER\n";
    auto tsep = epgeo->getTranslation();
    auto dimep = epgeo->getDimension();
    auto indep = epgeo->getIndexes();
    std::cout << "Translations " << tsep.size() << "\n";
    std::cout << "Dimensions " << dimep.size() << "\n";
    std::cout << "Indices " << indep.size() << "\n";
    for (auto it : tsep)
      std::cout << it;
    std::cout << "\n";
    for (auto it : dimep)
      std::cout << it;
    std::cout << "\n";
    for (auto it : indep)
      std::cout << it;
    std::cout << "\n";
  }

  if (m_hcaltest) {
    edm::ESHandle<PCaloGeometry> hgeo;
    iSetup.get<PHcalRcd>().get(hgeo);
    std::cout << "HCAL\n";
    auto tsh = hgeo->getTranslation();
    auto dimh = hgeo->getDimension();
    auto indh = hgeo->getIndexes();
    auto dindh = hgeo->getDenseIndices();
    std::cout << "Translations " << tsh.size() << "\n";
    std::cout << "Dimensions " << dimh.size() << "\n";
    std::cout << "Indices " << indh.size() << "\n";
    std::cout << "Dense Indices " << dindh.size() << "\n";
    for (auto it : tsh)
      std::cout << it;
    std::cout << "\n";
    for (auto it : dimh)
      std::cout << it;
    std::cout << "\n";
    for (auto it : indh)
      std::cout << it;
    for (auto it : dindh)
      std::cout << it;
    std::cout << "\n";
  }

  if (m_hgcaltest) {
    edm::ESHandle<PCaloGeometry> hgcgeo;
    iSetup.get<PHGCalRcd>().get(hgcgeo);
    std::cout << "HGCAL\n";
    auto tsh = hgcgeo->getTranslation();
    auto dimh = hgcgeo->getDimension();
    auto indh = hgcgeo->getIndexes();
    auto dindh = hgcgeo->getDenseIndices();
    std::cout << "Translations " << tsh.size() << "\n";
    std::cout << "Dimensions " << dimh.size() << "\n";
    std::cout << "Indices " << indh.size() << "\n";
    std::cout << "Dense Indices " << dindh.size() << "\n";
    for (auto it : tsh)
      std::cout << it;
    std::cout << "\n";
    for (auto it : dimh)
      std::cout << it;
    std::cout << "\n";
    for (auto it : indh)
      std::cout << it;
    for (auto it : dindh)
      std::cout << it;
    std::cout << "\n";
  }

  if (m_calotowertest) {
    edm::ESHandle<PCaloGeometry> ctgeo;
    iSetup.get<PCaloTowerRcd>().get(ctgeo);
    std::cout << "CALO TOWER:\n";
    auto tsct = ctgeo->getTranslation();
    auto dimct = ctgeo->getDimension();
    auto indct = ctgeo->getIndexes();
    std::cout << "Translations " << tsct.size() << "\n";
    std::cout << "Dimensions " << dimct.size() << "\n";
    std::cout << "Indices " << indct.size() << "\n";
    for (auto it : tsct)
      std::cout << it;
    std::cout << "\n";
    for (auto it : dimct)
      std::cout << it;
    std::cout << "\n";
    for (auto it : indct)
      std::cout << it;
    std::cout << "\n";
  }

  if (m_castortest) {
    edm::ESHandle<PCaloGeometry> castgeo;
    iSetup.get<PCastorRcd>().get(castgeo);
    std::cout << "CASTOR\n";
    for (auto it : castgeo->getTranslation())
      std::cout << it;
    std::cout << "\n";
    for (auto it : castgeo->getDimension())
      std::cout << it;
    std::cout << "\n";
    for (auto it : castgeo->getIndexes())
      std::cout << it;
    std::cout << "\n";
  }

  if (m_zdctest) {
    edm::ESHandle<PCaloGeometry> zdcgeo;
    iSetup.get<PZdcRcd>().get(zdcgeo);
    std::cout << "ZDC\n";
    for (auto it : zdcgeo->getTranslation())
      std::cout << it;
    std::cout << "\n";
    for (auto it : zdcgeo->getDimension())
      std::cout << it;
    std::cout << "\n";
    for (auto it : zdcgeo->getIndexes())
      std::cout << it;
    std::cout << "\n";
  }

  if (m_csctest) {
    edm::ESHandle<RecoIdealGeometry> cscgeo;
    iSetup.get<CSCRecoGeometryRcd>().get(cscgeo);

    edm::ESHandle<CSCRecoDigiParameters> cscdigigeo;
    iSetup.get<CSCRecoDigiParametersRcd>().get(cscdigigeo);
    std::cout << "CSC\n";

    std::vector<int> obj1(cscdigigeo->pUserParOffset);
    for (auto it : obj1)
      std::cout << it;
    std::cout << "\n";

    std::vector<int> obj2(cscdigigeo->pUserParSize);
    for (auto it : obj2)
      std::cout << it;
    std::cout << "\n";

    std::vector<int> obj3(cscdigigeo->pChamberType);
    for (auto it : obj3)
      std::cout << it;
    std::cout << "\n";

    std::vector<float> obj4(cscdigigeo->pfupars);
    for (auto it : obj4)
      std::cout << it;
    std::cout << "\n";

    std::vector<DetId> myIdcsc(cscgeo->detIds());
    for (auto it : myIdcsc)
      std::cout << it;
    std::cout << "\n";

    uint32_t cscsize = myIdcsc.size();
    for (uint32_t i = 0; i < cscsize; i++) {
      std::vector<double> trcsc(cscgeo->tranStart(i), cscgeo->tranEnd(i));
      for (auto it : trcsc)
        std::cout << it;
      std::cout << "\n";

      std::vector<double> rotcsc(cscgeo->rotStart(i), cscgeo->rotEnd(i));
      for (auto it : rotcsc)
        std::cout << it;
      std::cout << "\n";

      std::vector<double> shapecsc(cscgeo->shapeStart(i), cscgeo->shapeEnd(i));
      for (auto it : shapecsc)
        std::cout << it;
      std::cout << "\n";
    }
  }

  if (m_dttest) {
    edm::ESHandle<RecoIdealGeometry> dtgeo;
    iSetup.get<DTRecoGeometryRcd>().get(dtgeo);
    std::cout << "DT\n";
    std::vector<DetId> myIddt(dtgeo->detIds());
    for (auto it : myIddt)
      std::cout << it;
    std::cout << "\n";

    uint32_t dtsize = myIddt.size();
    for (uint32_t i = 0; i < dtsize; i++) {
      std::vector<double> trdt(dtgeo->tranStart(i), dtgeo->tranEnd(i));
      for (auto it : trdt)
        std::cout << it;
      std::cout << "\n";

      std::vector<double> rotdt(dtgeo->rotStart(i), dtgeo->rotEnd(i));
      for (auto it : rotdt)
        std::cout << it;
      std::cout << "\n";

      std::vector<double> shapedt(dtgeo->shapeStart(i), dtgeo->shapeEnd(i));
      for (auto it : shapedt)
        std::cout << it;
      std::cout << "\n";
    }
  }

  if (m_rpctest) {
    edm::ESHandle<RecoIdealGeometry> rpcgeo;
    iSetup.get<RPCRecoGeometryRcd>().get(rpcgeo);
    std::cout << "RPC\n";

    std::vector<DetId> myIdrpc(rpcgeo->detIds());
    for (auto it : myIdrpc)
      std::cout << it;
    std::cout << "\n";
    uint32_t rpcsize = myIdrpc.size();
    for (uint32_t i = 0; i < rpcsize; i++) {
      std::vector<double> trrpc(rpcgeo->tranStart(i), rpcgeo->tranEnd(i));
      for (auto it : trrpc)
        std::cout << it;
      std::cout << "\n";

      std::vector<double> rotrpc(rpcgeo->rotStart(i), rpcgeo->rotEnd(i));
      for (auto it : rotrpc)
        std::cout << it;
      std::cout << "\n";

      std::vector<double> shaperpc(rpcgeo->shapeStart(i), rpcgeo->shapeEnd(i));
      for (auto it : shaperpc)
        std::cout << it;
      std::cout << "\n";

      std::vector<std::string> strrpc(rpcgeo->strStart(i), rpcgeo->strEnd(i));
      for (auto it : strrpc)
        std::cout << it;
      std::cout << "\n";
    }
  }
}

DEFINE_FWK_MODULE(GeometryTester);
