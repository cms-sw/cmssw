// Test CSCDetId & CSCIndexer 13.11.2007 ptc

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <DataFormats/MuonDetId/interface/CSCIndexer.h>
#include <DataFormats/GeometryVector/interface/Pi.h>

#include <cmath>
#include <iomanip>  // for setw() etc.
#include <string>
#include <vector>

class CSCDetIdAnalyzer : public edm::EDAnalyzer {
public:
  explicit CSCDetIdAnalyzer(const edm::ParameterSet&);
  ~CSCDetIdAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  const std::string& myName() { return myName_; }

private:
  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
};

CSCDetIdAnalyzer::CSCDetIdAnalyzer(const edm::ParameterSet& iConfig)
    : dashedLineWidth_(140), dashedLine_(std::string(dashedLineWidth_, '-')), myName_("CSCDetIdAnalyzer") {
  std::cout << dashedLine_ << std::endl;
  std::cout << "Welcome to " << myName_ << std::endl;
  std::cout << dashedLine_ << std::endl;
  std::cout << "I will build the CSC geometry, then iterate over all layers." << std::endl;
  std::cout << "From each CSCDetId I will build the associated linear index, skipping ME1a layers." << std::endl;
  std::cout << "I will build this index once from the layer labels and once from the CSCDetId, and check they agree."
            << std::endl;
  std::cout << "I will output one line per layer, listing the CSCDetId, the labels, and these two indexes."
            << std::endl;
  std::cout << "I will append the strip-channel indexes for the two edge strips and the central strip." << std::endl;
  std::cout << "Finally, I will rebuild a CSCDetId from the layer index, and check it is the same as the original,"
            << std::endl;
  std::cout << "and rebuild a CSCDetId from the final strip-channel index, and check it is the same as the original."
            << std::endl;
  std::cout << "If any of these tests fail, you will see assert failure messages in the output." << std::endl;
  std::cout << "If there are no such failures then the tests passed." << std::endl;
  std::cout << dashedLine_ << std::endl;
}

CSCDetIdAnalyzer::~CSCDetIdAnalyzer() {}

void CSCDetIdAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const double dPi = Geom::pi();
  const double radToDeg = 180. / dPi;

  std::cout << myName() << ": Analyzer..." << std::endl;
  std::cout << "start " << dashedLine_ << std::endl;
  std::cout << "pi = " << dPi << ", radToDeg = " << radToDeg << std::endl;

  edm::ESHandle<CSCGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get(pDD);
  std::cout << " Geometry node for CSCGeom is  " << &(*pDD) << std::endl;
  std::cout << " I have " << pDD->detTypes().size() << " detTypes" << std::endl;
  std::cout << " I have " << pDD->detUnits().size() << " detUnits" << std::endl;
  std::cout << " I have " << pDD->dets().size() << " dets" << std::endl;
  std::cout << " I have " << pDD->layers().size() << " layers" << std::endl;
  std::cout << " I have " << pDD->chambers().size() << " chambers" << std::endl;

  std::cout << myName() << ": Begin iteration over geometry..." << std::endl;
  std::cout << "iter " << dashedLine_ << std::endl;

  std::cout << "\n  #     id(dec)      id(oct)                        "
               "lindex     lindex2      cindex       label       strip  sindex   strip  sindex   strip  sindex"
            << std::endl;

  // Construct an indexer object
  CSCIndexer* theIndexer = new CSCIndexer;

  int icount = 0;
  int icountAll = 0;

  // Iterate over the DetUnits in the CSCGeometry
  for (const auto& it : pDD->detUnits()) {
    // Check each DetUnit really is a CSC layer
    auto layer = dynamic_cast<CSCLayer const*>(it);

    if (layer) {
      ++icountAll;  // how many layers we see

      // What's its DetId?

      DetId detId = layer->geographicalId();
      int id = detId();  // or detId.rawId()
      CSCDetId cscDetId = layer->id();

      // There's going to be a lot of messing with field width (and precision) so
      // save input values...
      int iw = std::cout.width();      // save current width
      int ip = std::cout.precision();  // save current precision

      short ie = CSCDetId::endcap(id);
      short is = CSCDetId::station(id);
      short ir = CSCDetId::ring(id);
      short ic = CSCDetId::chamber(id);
      short il = CSCDetId::layer(id);

      if (ir == 4)
        continue;  // DOES NOT HANDLE ME1a SEPARATELY FROM ME11
      ++icount;    // count ONLY non-ME1a layers!

      std::cout << std::setw(4) << icount << std::setw(12) << id << std::oct << std::setw(12) << id << std::dec
                << std::setw(iw) << "   E" << ie << " S" << is << " R" << ir << " C" << std::setw(2) << ic
                << std::setw(iw) << " L" << il;

      unsigned lind = theIndexer->layerIndex(ie, is, ir, ic, il);
      unsigned cind = theIndexer->startChamberIndexInEndcap(ie, is, ir) + ic - 1;
      unsigned lind2 = theIndexer->layerIndex(cscDetId);

      //	   std::cout << std::setw(12) << std::setw(12) << lind << std::setw(12) << lind2 << "     " << std::endl;
      std::cout << std::setw(12) << lind << std::setw(12) << lind2 << std::setw(12) << cind << std::setw(12)
                << theIndexer->chamberLabelFromChamberIndex(cind) << "     ";

      // Index a few strips
      unsigned short nstrips = theIndexer->stripChannelsPerLayer(is, ir);
      unsigned int sc1 = theIndexer->stripChannelIndex(ie, is, ir, ic, il, 1);
      unsigned int scm = theIndexer->stripChannelIndex(ie, is, ir, ic, il, nstrips / 2);
      unsigned int scn = theIndexer->stripChannelIndex(ie, is, ir, ic, il, nstrips);

      std::cout << "      1  " << std::setw(6) << sc1 << "      " << nstrips / 2 << "  " << std::setw(6) << scm
                << "      " << nstrips << "  " << std::setw(6) << scn << std::endl;

      // Reset the values we changed
      std::cout << std::setprecision(ip) << std::setw(iw);

      // ASSERTS
      // =======

      // Check layer indices are consistent
      //	   std::cout << "lind2 = " << lind2 << ", lind=" << lind << std::endl;
      assert(lind2 == lind);

      // Build CSCDetId from this index and check it's same as original
      CSCDetId cscDetId2 = theIndexer->detIdFromLayerIndex(lind);
      // std::cout << "cscDetId2 = E" << cscDetId2.endcap() << " S" << cscDetId2.station() << " R" << cscDetId2.ring() << " C" << cscDetId2.chamber() << " L" << cscDetId2.layer() << std::endl;
      assert(cscDetId2 == cscDetId);

      // Build CSCDetId from the strip-channel index for strip 'nstrips' and check it matches
      std::pair<CSCDetId, unsigned short int> p = theIndexer->detIdFromStripChannelIndex(scn);
      CSCDetId cscDetId3 = p.first;
      unsigned short iscn = p.second;
      // std::cout << "scn=" << scn << "  iscn=" << iscn << std::endl;
      // std::cout << "cscDetId3 = E" << cscDetId3.endcap() << " S" << cscDetId3.station() << " R" << cscDetId3.ring() << " C" << cscDetId3.chamber() << " L" << cscDetId3.layer() << std::endl;
      assert(iscn == nstrips);
      assert(cscDetId3 == cscDetId);

      // Check idToDetUnit
      const GeomDetUnit* gdu = pDD->idToDetUnit(detId);
      assert(gdu == layer);
      // Check idToDet
      const GeomDet* gd = pDD->idToDet(detId);
      assert(gd == layer);
    } else {
      std::cout << "Something wrong ... could not dynamic_cast Det* to CSCLayer* " << std::endl;
    }
  }

  delete theIndexer;

  std::cout << dashedLine_ << " end" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCDetIdAnalyzer);
