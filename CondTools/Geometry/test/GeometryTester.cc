#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "CondFormats/GeometryObjects/interface/PCaloGeometry.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"
#include "DataFormats/Math/interface/Rounding.h"

#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
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

  class FmtOstream {
  public:
    FmtOstream() = default;

    friend FmtOstream &operator<<(FmtOstream &output, const double val) {
      std::cout << " " << cms_rounding::roundIfNear0(val);
      return (output);
    }

    friend FmtOstream &operator<<(FmtOstream &output, const int val) {
      std::cout << " " << val;
      return (output);
    }

    friend FmtOstream &operator<<(FmtOstream &output, const unsigned int val) {
      std::cout << " " << val;
      return (output);
    }

    friend FmtOstream &operator<<(FmtOstream &output, const bool val) {
      std::cout << " " << (val ? "true" : "false");
      return (output);
    }

    friend FmtOstream &operator<<(FmtOstream &output, const std::string &strVal) {
      std::cout << " " << strVal;
      return (output);
    }

    friend FmtOstream &operator<<(FmtOstream &output, const char *cstr) {
      std::cout << " " << cstr;
      return (output);
    }
  };

  class GeometryTester : public edm::one::EDAnalyzer<> {
  public:
    GeometryTester(edm::ParameterSet const &);
    void beginJob() override {}
    void analyze(edm::Event const &, edm::EventSetup const &) override;
    void endJob() override {}

  private:
    bool m_xmltest, m_tktest, m_ecaltest;
    bool m_hcaltest, m_hgcaltest, m_calotowertest;
    bool m_castortest, m_zdctest, m_csctest;
    bool m_dttest, m_rpctest, m_round;
    std::string m_geomLabel;
    edm::ESGetToken<FileBlob, GeometryFileRcd> m_xmlGeoToken;
    edm::ESGetToken<PGeometricDet, IdealGeometryRecord> m_tkGeoToken;
    edm::ESGetToken<PCaloGeometry, PEcalBarrelRcd> m_ebGeoToken;
    edm::ESGetToken<PCaloGeometry, PEcalEndcapRcd> m_eeGeoToken;
    edm::ESGetToken<PCaloGeometry, PEcalPreshowerRcd> m_epGeoToken;
    edm::ESGetToken<PCaloGeometry, PHcalRcd> m_hGeoToken;
    edm::ESGetToken<PCaloGeometry, PHGCalRcd> m_hgcGeoToken;
    edm::ESGetToken<PCaloGeometry, PCaloTowerRcd> m_ctGeoToken;
    edm::ESGetToken<PCaloGeometry, PCastorRcd> m_castGeoToken;
    edm::ESGetToken<PCaloGeometry, PZdcRcd> m_zdcGeoToken;
    edm::ESGetToken<CSCRecoDigiParameters, CSCRecoDigiParametersRcd> m_cscDigiGeoToken;
    edm::ESGetToken<RecoIdealGeometry, CSCRecoGeometryRcd> m_cscGeoToken;
    edm::ESGetToken<RecoIdealGeometry, DTRecoGeometryRcd> m_dtGeoToken;
    edm::ESGetToken<RecoIdealGeometry, RPCRecoGeometryRcd> m_rpcGeoToken;
  };
}  // namespace

GeometryTester::GeometryTester(const edm::ParameterSet &iConfig)
    : m_xmltest(iConfig.getUntrackedParameter<bool>("XMLTest", true)),
      m_tktest(iConfig.getUntrackedParameter<bool>("TrackerTest", true)),
      m_ecaltest(iConfig.getUntrackedParameter<bool>("EcalTest", true)),
      m_hcaltest(iConfig.getUntrackedParameter<bool>("HcalTest", true)),
      m_hgcaltest(iConfig.getUntrackedParameter<bool>("HGCalTest", true)),
      m_calotowertest(iConfig.getUntrackedParameter<bool>("CaloTowerTest", true)),
      m_castortest(iConfig.getUntrackedParameter<bool>("CastorTest", true)),
      m_zdctest(iConfig.getUntrackedParameter<bool>("ZDCTest", true)),
      m_csctest(iConfig.getUntrackedParameter<bool>("CSCTest", true)),
      m_dttest(iConfig.getUntrackedParameter<bool>("DTTest", true)),
      m_rpctest(iConfig.getUntrackedParameter<bool>("RPCTest", true)),
      m_round(iConfig.getUntrackedParameter<bool>("roundValues", false)),
      m_geomLabel(iConfig.getUntrackedParameter<std::string>("geomLabel", "Extended")),
      m_xmlGeoToken(esConsumes(edm::ESInputTag("", m_geomLabel))),
      m_tkGeoToken(esConsumes()),
      m_ebGeoToken(esConsumes()),
      m_eeGeoToken(esConsumes()),
      m_epGeoToken(esConsumes()),
      m_hGeoToken(esConsumes()),
      m_hgcGeoToken(esConsumes()),
      m_ctGeoToken(esConsumes()),
      m_castGeoToken(esConsumes()),
      m_zdcGeoToken(esConsumes()),
      m_cscDigiGeoToken(esConsumes()),
      m_cscGeoToken(esConsumes()),
      m_dtGeoToken(esConsumes()),
      m_rpcGeoToken(esConsumes()) {}

void GeometryTester::analyze(const edm::Event &, const edm::EventSetup &iSetup) {
  if (m_xmltest) {
    auto const &xmlGeo = iSetup.getData(m_xmlGeoToken);

    std::cout << "XML FILE\n";
    std::unique_ptr<std::vector<unsigned char> > tb = xmlGeo.getUncompressedBlob();
    std::cout << "SIZE FILE = " << tb->size() << "\n";
    for (auto it : *tb) {
      std::cout << it;
    }
    std::cout << "\n";
  }

  FmtOstream outStream;
  if (m_tktest) {
    auto const &tkGeo = iSetup.getData(m_tkGeoToken);
    std::cout << "TRACKER\n";

    for (auto it : tkGeo.pgeomdets_) {
      std::cout << "trk ";
      outStream << it._params0 << it._params1 << it._params2 << it._params3 << it._params4 << it._params5 << it._params6
                << it._params7 << it._params8 << it._params9 << it._params10 << it._x << it._y << it._z << it._phi
                << it._rho << it._a11 << it._a12 << it._a13 << it._a21 << it._a22 << it._a23 << it._a31 << it._a32
                << it._a33 << it._shape << it._name << it._ns;
      outStream << it._radLength << it._xi << it._pixROCRows << it._pixROCCols << it._pixROCx << it._pixROCy
                << it._stereo << it._siliconAPVNum << it._geographicalID << it._nt0 << it._nt1 << it._nt2 << it._nt3
                << it._nt4 << it._nt5 << it._nt6 << it._nt7 << it._nt8 << it._nt9 << it._nt10 << "\n";
    }
  }

  if (m_ecaltest) {
    auto const &ebGeo = iSetup.getData(m_ebGeoToken);
    std::cout << "ECAL BARREL\n";
    auto tseb = ebGeo.getTranslation();
    auto dimeb = ebGeo.getDimension();
    auto indeb = ebGeo.getIndexes();
    std::cout << "Translations " << tseb.size() << "\n";
    std::cout << "Dimensions " << dimeb.size() << "\n";
    std::cout << "Indices " << indeb.size() << "\n";
    std::cout << "ecalb ";
    for (auto it : tseb)
      outStream << it;
    std::cout << "\n";
    std::cout << "ecalb ";
    for (auto it : dimeb)
      outStream << it;
    std::cout << "\n";
    std::cout << "ecalb ";
    for (auto it : indeb)
      outStream << it;
    std::cout << "\n";

    auto const &eeGeo = iSetup.getData(m_eeGeoToken);
    std::cout << "ECAL ENDCAP\n";
    auto tsee = eeGeo.getTranslation();
    auto dimee = eeGeo.getDimension();
    auto indee = eeGeo.getIndexes();
    std::cout << "Translations " << tsee.size() << "\n";
    std::cout << "Dimensions " << dimee.size() << "\n";
    std::cout << "Indices " << indee.size() << "\n";
    std::cout << "ecale " << std::endl;
    int cnt = 1;
    for (auto it : tsee) {
      if (cnt % 6 == 1) {
        outStream << "ecale xyz ";
        if (m_round)
          std::cout << std::defaultfloat << std::setprecision(6);
      } else if (cnt % 3 == 1) {
        outStream << "ecale phi/theta/psi ";
        if (m_round) {
          std::cout << std::setprecision(4);
          // Angles show fluctuations beyond four digits
          if (std::abs(it) < 1.e-4) {
            it = 0.0;
            // Show tiny phi values as 0 to avoid fluctuations
          }
        }
      }
      outStream << it;
      if ((cnt++) % 3 == 0)
        std::cout << "\n";
    }
    std::cout << "\n";
    if (m_round)
      std::cout << std::setprecision(6);
    std::cout << "ecale ";
    for (auto it : dimee)
      outStream << it;
    std::cout << "\n";
    std::cout << "ecale ";
    for (auto it : indee)
      outStream << it;
    std::cout << "\n";

    auto const &epGeo = iSetup.getData(m_epGeoToken);
    std::cout << "ECAL PRESHOWER\n";
    auto tsep = epGeo.getTranslation();
    auto dimep = epGeo.getDimension();
    auto indep = epGeo.getIndexes();
    std::cout << "Translations " << tsep.size() << "\n";
    std::cout << "Dimensions " << dimep.size() << "\n";
    std::cout << "Indices " << indep.size() << "\n";
    std::cout << "ecalpre ";
    for (auto it : tsep)
      outStream << it;
    std::cout << "\n";
    std::cout << "ecalpre ";
    for (auto it : dimep)
      outStream << it;
    std::cout << "\n";
    std::cout << "ecalpre ";
    for (auto it : indep)
      outStream << it;
    std::cout << "\n";
  }

  if (m_hcaltest) {
    auto const &hGeo = iSetup.getData(m_hGeoToken);
    std::cout << "HCAL\n";
    auto tsh = hGeo.getTranslation();
    auto dimh = hGeo.getDimension();
    auto indh = hGeo.getIndexes();
    auto dindh = hGeo.getDenseIndices();
    std::cout << "Translations " << tsh.size() << "\n";
    std::cout << "Dimensions " << dimh.size() << "\n";
    std::cout << "Indices " << indh.size() << "\n";
    std::cout << "Dense Indices " << dindh.size() << "\n";
    std::cout << "hcal";
    for (auto it : tsh)
      outStream << it;
    std::cout << "\n";
    std::cout << "hcal";
    for (auto it : dimh)
      outStream << it;
    std::cout << "\n";
    std::cout << "hcal";
    for (auto it : indh)
      outStream << it;
    std::cout << "hcal";
    for (auto it : dindh)
      outStream << it;
    std::cout << "\n";
  }

  if (m_hgcaltest) {
    auto const &hgcGeo = iSetup.getData(m_hgcGeoToken);
    std::cout << "HGCAL\n";
    auto tsh = hgcGeo.getTranslation();
    auto dimh = hgcGeo.getDimension();
    auto indh = hgcGeo.getIndexes();
    auto dindh = hgcGeo.getDenseIndices();
    std::cout << "Translations " << tsh.size() << "\n";
    std::cout << "Dimensions " << dimh.size() << "\n";
    std::cout << "Indices " << indh.size() << "\n";
    std::cout << "Dense Indices " << dindh.size() << "\n";
    std::cout << "hgcal ";
    for (auto it : tsh)
      outStream << it;
    std::cout << "\n";
    std::cout << "hgcal ";
    for (auto it : dimh)
      outStream << it;
    std::cout << "\n";
    std::cout << "hgcal ";
    for (auto it : indh)
      outStream << it;
    std::cout << "\n";
    std::cout << "hgcal ";
    for (auto it : dindh)
      outStream << it;
    std::cout << "\n";
  }

  if (m_calotowertest) {
    auto const &ctGeo = iSetup.getData(m_ctGeoToken);
    std::cout << "CALO TOWER:\n";
    auto tsct = ctGeo.getTranslation();
    auto dimct = ctGeo.getDimension();
    auto indct = ctGeo.getIndexes();
    std::cout << "Translations " << tsct.size() << "\n";
    std::cout << "Dimensions " << dimct.size() << "\n";
    std::cout << "Indices " << indct.size() << "\n";
    std::cout << "calotow ";
    for (auto it : tsct)
      outStream << it;
    std::cout << "\n";
    std::cout << "calotow ";
    for (auto it : dimct)
      outStream << it;
    std::cout << "\n";
    std::cout << "calotow ";
    for (auto it : indct)
      outStream << it;
    std::cout << "\n";
  }

  if (m_castortest) {
    auto const &castGeo = iSetup.getData(m_castGeoToken);
    std::cout << "CASTOR\n";
    std::cout << "castor ";
    for (auto it : castGeo.getTranslation())
      outStream << it;
    std::cout << "\n";
    std::cout << "castor ";
    for (auto it : castGeo.getDimension())
      outStream << it;
    std::cout << "\n";
    std::cout << "castor ";
    for (auto it : castGeo.getIndexes())
      outStream << it;
    std::cout << "\n";
  }

  if (m_zdctest) {
    auto const &zdcGeo = iSetup.getData(m_zdcGeoToken);
    std::cout << "ZDC\n";
    std::cout << "zdc ";
    for (auto it : zdcGeo.getTranslation())
      outStream << it;
    std::cout << "\n";
    std::cout << "zdc ";
    for (auto it : zdcGeo.getDimension())
      outStream << it;
    std::cout << "\n";
    std::cout << "zdc ";
    for (auto it : zdcGeo.getIndexes())
      outStream << it;
    std::cout << "\n";
  }

  if (m_csctest) {
    auto const &cscGeo = iSetup.getData(m_cscGeoToken);

    auto const &cscDigiGeo = iSetup.getData(m_cscDigiGeoToken);
    std::cout << "CSC\n";

    std::vector<int> obj1(cscDigiGeo.pUserParOffset);
    std::cout << "csc ";
    for (auto it : obj1)
      outStream << it;
    std::cout << "\n";

    std::vector<int> obj2(cscDigiGeo.pUserParSize);
    std::cout << "csc ";
    for (auto it : obj2)
      outStream << it;
    std::cout << "\n";

    std::vector<int> obj3(cscDigiGeo.pChamberType);
    std::cout << "csc ";
    for (auto it : obj3)
      outStream << it;
    std::cout << "\n";

    std::vector<float> obj4(cscDigiGeo.pfupars);
    std::cout << "csc ";
    for (auto it : obj4)
      outStream << it;
    std::cout << "\n";

    std::vector<DetId> myIdcsc(cscGeo.detIds());
    std::cout << "csc ";
    for (auto it : myIdcsc)
      outStream << it;
    std::cout << "\n";

    uint32_t cscsize = myIdcsc.size();
    for (uint32_t i = 0; i < cscsize; i++) {
      std::vector<double> trcsc(cscGeo.tranStart(i), cscGeo.tranEnd(i));
      std::cout << "csc ";
      for (auto it : trcsc)
        outStream << it;
      std::cout << "\n";

      std::vector<double> rotcsc(cscGeo.rotStart(i), cscGeo.rotEnd(i));
      std::cout << "csc ";
      for (auto it : rotcsc)
        outStream << it;
      std::cout << "\n";

      std::vector<double> shapecsc(cscGeo.shapeStart(i), cscGeo.shapeEnd(i));
      std::cout << "csc ";
      for (auto it : shapecsc)
        outStream << it;
      std::cout << "\n";
    }
  }

  if (m_dttest) {
    auto const &dtGeo = iSetup.getData(m_dtGeoToken);
    std::cout << "DT\n";
    std::vector<DetId> myIddt(dtGeo.detIds());
    std::cout << "dt ";
    for (auto it : myIddt)
      std::cout << " " << it;  // DetId
    std::cout << "\n";

    uint32_t dtsize = myIddt.size();
    std::cout << "dt ";
    for (uint32_t i = 0; i < dtsize; i++) {
      std::vector<double> trdt(dtGeo.tranStart(i), dtGeo.tranEnd(i));
      for (auto it : trdt)
        outStream << it;
      std::cout << "\n";

      std::vector<double> rotdt(dtGeo.rotStart(i), dtGeo.rotEnd(i));
      std::cout << "dt ";
      for (auto it : rotdt)
        outStream << it;
      std::cout << "\n";

      std::vector<double> shapedt(dtGeo.shapeStart(i), dtGeo.shapeEnd(i));
      std::cout << "dt ";
      for (auto it : shapedt)
        outStream << it;
      std::cout << "\n";
    }
  }

  if (m_rpctest) {
    auto const &rpcGeo = iSetup.getData(m_rpcGeoToken);
    std::cout << "RPC\n";

    std::vector<DetId> myIdrpc(rpcGeo.detIds());
    std::cout << "rpc ";
    for (auto it : myIdrpc)
      std::cout << " " << it;  // DetId
    std::cout << "\n";
    uint32_t rpcsize = myIdrpc.size();
    for (uint32_t i = 0; i < rpcsize; i++) {
      std::vector<double> trrpc(rpcGeo.tranStart(i), rpcGeo.tranEnd(i));
      std::cout << "rpc ";
      for (auto it : trrpc)
        outStream << it;
      std::cout << "\n";

      std::vector<double> rotrpc(rpcGeo.rotStart(i), rpcGeo.rotEnd(i));
      std::cout << "rpc ";
      for (auto it : rotrpc)
        outStream << it;
      std::cout << "\n";

      std::vector<double> shaperpc(rpcGeo.shapeStart(i), rpcGeo.shapeEnd(i));
      std::cout << "rpc ";
      for (auto it : shaperpc)
        outStream << it;
      std::cout << "\n";

      std::vector<std::string> strrpc(rpcGeo.strStart(i), rpcGeo.strEnd(i));
      std::cout << "rpc ";
      for (auto it : strrpc)
        outStream << it;
      std::cout << "\n";
    }
  }
}

DEFINE_FWK_MODULE(GeometryTester);
