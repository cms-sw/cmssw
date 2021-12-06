/** Derived from DTGeometryAnalyzer by Nicola Amapane
 *
 *  \author M. Maggi - INFN Bari
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <memory>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

class ME0GeometryAnalyzer : public edm::one::EDAnalyzer<> {
public:
  ME0GeometryAnalyzer(const edm::ParameterSet& pset);

  ~ME0GeometryAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string& myName() { return myName_; }

  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  const edm::ESGetToken<ME0Geometry, MuonGeometryRecord> tokGeom_;
  std::ofstream ofos;
};

using namespace std;

ME0GeometryAnalyzer::ME0GeometryAnalyzer(const edm::ParameterSet& /*iConfig*/)
    : dashedLineWidth_(104),
      dashedLine_(string(dashedLineWidth_, '-')),
      myName_("ME0GeometryAnalyzer"),
      tokGeom_{esConsumes<ME0Geometry, MuonGeometryRecord>(edm::ESInputTag{})} {
  ofos.open("ME0testOutput.out");
  ofos << "======================== Opening output file" << endl;
}

ME0GeometryAnalyzer::~ME0GeometryAnalyzer() {
  ofos.close();
  ofos << "======================== Closing output file" << endl;
}

void ME0GeometryAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const auto& pDD = iSetup.getData(tokGeom_);

  ofos << myName() << ": Analyzer..." << endl;
  ofos << "start " << dashedLine_ << endl;

  ofos << " Geometry node for ME0Geom is  " << &pDD << endl;
  ofos << " detTypes       \t" << pDD.detTypes().size() << endl;
  ofos << " GeomDetUnit       \t" << pDD.detUnits().size() << endl;
  ofos << " GeomDet           \t" << pDD.dets().size() << endl;
  ofos << " GeomDetUnit DetIds\t" << pDD.detUnitIds().size() << endl;
  ofos << " eta partitions \t" << pDD.etaPartitions().size() << endl;
  ofos << " layers         \t" << pDD.layers().size() << endl;
  ofos << " chambers       \t" << pDD.chambers().size() << endl;
  // ofos << " regions        \t"              <<pDD.regions().size() << endl;

  // checking uniqueness of roll detIds
  bool flagNonUniqueRollID = false;
  bool flagNonUniqueRollRawID = false;
  int nstrips = 0;
  int npads = 0;
  for (auto roll1 : pDD.etaPartitions()) {
    nstrips += roll1->nstrips();
    npads += roll1->npads();
    for (auto roll2 : pDD.etaPartitions()) {
      if (roll1 != roll2) {
        if (roll1->id() == roll2->id())
          flagNonUniqueRollID = true;
        if (roll1->id().rawId() == roll2->id().rawId())
          flagNonUniqueRollRawID = true;
      }
    }
  }
  if (flagNonUniqueRollID or flagNonUniqueRollRawID)
    ofos << " -- WARNING: non unique roll Ids!!!" << endl;

  // checking uniqueness of layer detIds
  bool flagNonUniqueLaID = false;
  bool flagNonUniqueLaRawID = false;
  for (auto la1 : pDD.layers()) {
    for (auto la2 : pDD.layers()) {
      if (la1 != la2) {
        if (la1->id() == la2->id())
          flagNonUniqueLaID = true;
        if (la1->id().rawId() == la2->id().rawId())
          flagNonUniqueLaRawID = true;
      }
    }
  }
  if (flagNonUniqueLaID or flagNonUniqueLaRawID)
    ofos << " -- WARNING: non unique layer Ids!!!" << endl;

  // checking uniqueness of chamber detIds
  bool flagNonUniqueChID = false;
  bool flagNonUniqueChRawID = false;
  for (auto ch1 : pDD.chambers()) {
    for (auto ch2 : pDD.chambers()) {
      if (ch1 != ch2) {
        if (ch1->id() == ch2->id())
          flagNonUniqueChID = true;
        if (ch1->id().rawId() == ch2->id().rawId())
          flagNonUniqueChRawID = true;
      }
    }
  }
  if (flagNonUniqueChID or flagNonUniqueChRawID)
    ofos << " -- WARNING: non unique chamber Ids!!!" << endl;

  // print out number of strips and pads
  ofos << " total number of strips\t" << nstrips << endl;
  ofos << " total number of pads  \t" << npads << endl;

  ofos << myName() << ": Begin iteration over geometry..." << endl;
  ofos << "iter " << dashedLine_ << endl;

  ofos << myName() << "Begin ME0Geometry TEST" << endl;

  /*
   * possible checklist for an eta partition:
   *   base_bottom, base_top, height, strips, pads
   *   cx, cy, cz, ceta, cphi
   *   tx, ty, tz, teta, tphi
   *   bx, by, bz, beta, bphi
   *   pitch center, pitch bottom, pitch top
   *   deta, dphi
   *   gap thicess
   *   sum of all dx + gap = chamber height
   */

  int i = 1;
  for (auto ch : pDD.chambers()) {
    ME0DetId chId(ch->id());
    int nLayers(ch->nLayers());
    ofos << "\tME0Chamber " << i << ", ME0DetId = " << chId.rawId() << ", " << chId << " has " << nLayers << " layers."
         << endl;
    int j = 1;
    for (auto la : ch->layers()) {
      ME0DetId laId(la->id());
      int nRolls(la->nEtaPartitions());
      ofos << "\t\tME0Layer " << j << ", ME0DetId = " << laId.rawId() << ", " << laId << " has " << nRolls
           << " eta partitions." << endl;

      int k = 1;
      auto& rolls(la->etaPartitions());
      for (auto roll : rolls) {
        // for (auto roll : pDD.etaPartitions()){
        ME0DetId rId(roll->id());
        ofos << "\t\t\tME0EtaPartition " << k << " , ME0DetId = " << rId.rawId() << ", " << rId << endl;

        const BoundPlane& bSurface(roll->surface());
        const StripTopology* topology(&(roll->specificTopology()));

        // base_bottom, base_top, height, strips, pads (all half length)
        auto& parameters(roll->specs()->parameters());
        float bottomEdge(parameters[0]);
        float topEdge(parameters[1]);
        float height(parameters[2]);
        float nStrips(parameters[3]);
        float nPads(parameters[4]);

        LocalPoint lCentre(0., 0., 0.);
        GlobalPoint gCentre(bSurface.toGlobal(lCentre));

        LocalPoint lTop(0., height, 0.);
        GlobalPoint gTop(bSurface.toGlobal(lTop));

        LocalPoint lBottom(0., -height, 0.);
        GlobalPoint gBottom(bSurface.toGlobal(lBottom));

        //   gx, gy, gz, geta, gphi (center)
        double cx(gCentre.x());
        double cy(gCentre.y());
        double cz(gCentre.z());
        double ceta(gCentre.eta());
        int cphi(static_cast<int>(gCentre.phi().degrees()));
        if (cphi < 0)
          cphi += 360;

        double tx(gTop.x());
        double ty(gTop.y());
        double tz(gTop.z());
        double teta(gTop.eta());
        int tphi(static_cast<int>(gTop.phi().degrees()));
        if (tphi < 0)
          tphi += 360;

        double bx(gBottom.x());
        double by(gBottom.y());
        double bz(gBottom.z());
        double beta(gBottom.eta());
        int bphi(static_cast<int>(gBottom.phi().degrees()));
        if (bphi < 0)
          bphi += 360;

        // pitch bottom, pitch top, pitch centre
        float pitch(roll->pitch());
        float topPitch(roll->localPitch(lTop));
        float bottomPitch(roll->localPitch(lBottom));

        // Type - should be GHA0[1-nRolls]
        string type(roll->type().name());

        // print info about edges
        LocalPoint lEdge1(topology->localPosition(0.));
        LocalPoint lEdgeN(topology->localPosition((float)nStrips));

        double cstrip1(roll->toGlobal(lEdge1).phi().degrees());
        double cstripN(roll->toGlobal(lEdgeN).phi().degrees());
        double dphi(cstripN - cstrip1);
        if (dphi < 0.)
          dphi += 360.;

        double deta(abs(beta - teta));
        const bool printDetails(true);
        if (printDetails)
          ofos << "\t\t\t\tType: " << type << endl
               << "\t\t\t\tDimensions[cm]: b = " << bottomEdge * 2 << ", B = " << topEdge * 2 << ", H  = " << height * 2
               << endl
               << "\t\t\t\tnStrips = " << nStrips << ", nPads =  " << nPads << endl
               << "\t\t\t\ttop(x,y,z)[cm] = (" << tx << ", " << ty << ", " << tz << "), top (eta,phi) = (" << teta
               << ", " << tphi << ")" << endl
               << "\t\t\t\tcenter(x,y,z) = (" << cx << ", " << cy << ", " << cz << "), center(eta,phi) = (" << ceta
               << ", " << cphi << ")" << endl
               << "\t\t\t\tbottom(x,y,z) = (" << bx << ", " << by << ", " << bz << "), bottom(eta,phi) = (" << beta
               << ", " << bphi << ")" << endl
               << "\t\t\t\tpitch (top,center,bottom) = " << topPitch << " " << pitch << " " << bottomPitch
               << ", dEta = " << deta << ", dPhi = " << dphi << endl
               << "\t\t\t\tlocal pos at strip 1 " << lEdge1 << " strip N " << lEdgeN << endl;
        ++k;
      }
      ++j;
    }
    ++i;
  }
  ofos << dashedLine_ << " end" << endl;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ME0GeometryAnalyzer);
