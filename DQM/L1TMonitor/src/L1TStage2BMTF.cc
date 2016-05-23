#include "DQM/L1TMonitor/interface/L1TStage2BMTF.h"


L1TStage2BMTF::L1TStage2BMTF(const edm::ParameterSet& ps)
    : bmtfToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("bmtfSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TStage2BMTF::~L1TStage2BMTF() {}

void L1TStage2BMTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2BMTF::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2BMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  ibooker.setCurrentFolder(monitorDir);
 
  bmtfBX = ibooker.book2D("bmtfBX", "BMTF BX", 12, -0.5, 11.5, 7, -3.5, 3.5);
  bmtfBX->setAxisTitle("Wedge", 1);
  for (int bin = 1; bin < 13; ++bin) {
    bmtfBX->setBinLabel(bin, std::to_string(bin), 1);
  }
  bmtfBX->setAxisTitle("BX", 2);

  bmtfhwPt = ibooker.book1D("bmtfhwPt", "BMTF p_{T}", 512, -0.5, 511.5);
  bmtfhwPt->setAxisTitle("Hardware p_{T}", 1);

  bmtfhwEta = ibooker.book1D("bmtfhwEta", "BMTF #eta", 461, -230.5, 230.5);
  bmtfhwEta->setAxisTitle("Hardware #eta", 1);

  bmtfhwPhi = ibooker.book1D("bmtfhwPhi", "BMTF #phi", 201, -100.5, 100.5);
  bmtfhwPhi->setAxisTitle("Hardware #phi", 1);

  bmtfhwPtvshwEta = ibooker.book2D("bmtfhwPtvshwEta", "BMTF p_{T} vs #eta", 461, -230.5, 230.5, 512, -0.5, 511.5);
  bmtfhwPtvshwEta->setAxisTitle("Hardware #eta", 1);
  bmtfhwPtvshwEta->setAxisTitle("Hardware p_{T}", 2);
  
  bmtfhwPtvshwPhi = ibooker.book2D("bmtfhwPtvshwPhi", "BMTF p_{T} vs #phi", 201, -100.5, 100.5, 512, -0.5, 511.5);
  bmtfhwPtvshwPhi->setAxisTitle("Hardware #phi", 1);
  bmtfhwPtvshwPhi->setAxisTitle("Hardware p_{T}", 2);

  bmtfhwPhivshwEta = ibooker.book2D("bmtfhwPhivshwEta", "BMTF #phi vs #eta", 461, -230.5, 230.5, 201, -100.5, 100.5);
  bmtfhwPhivshwEta->setAxisTitle("Hardware #eta", 1);
  bmtfhwPhivshwEta->setAxisTitle("Hardware #phi", 2);

  bmtfBXvshwPt = ibooker.book2D("bmtfBXvshwPt", "BMTF BX vs p_{T}", 512, -0.5, 511.5, 7, -3.5, 3.5);
  bmtfBXvshwPt->setAxisTitle("Hardware p_{T}", 1);
  bmtfBXvshwPt->setAxisTitle("BX", 2);

  bmtfBXvshwEta = ibooker.book2D("bmtfBXvshwEta", "BMTF BX vs #eta", 461, -230.5, 230.5, 7, -3.5, 3.5);
  bmtfBXvshwEta->setAxisTitle("Hardware #eta", 1);
  bmtfBXvshwEta->setAxisTitle("BX", 2);

  bmtfBXvshwPhi = ibooker.book2D("bmtfBXvshwPhi", "BMTF BX vs #phi", 201, -100.5, 100.5, 7, -3.5, 3.5);
  bmtfBXvshwPhi->setAxisTitle("Hardware #phi", 1);
  bmtfBXvshwPhi->setAxisTitle("BX", 2);
}

void L1TStage2BMTF::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2BMTF") << "L1TStage2BMTF: analyze..." << std::endl;

  edm::Handle<l1t::RegionalMuonCandBxCollection> BMTFBxCollection;
  e.getByToken(bmtfToken, BMTFBxCollection);

  for (int itBX = BMTFBxCollection->getFirstBX(); itBX <= BMTFBxCollection->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator BMTF = BMTFBxCollection->begin(itBX); BMTF != BMTFBxCollection->end(itBX); ++BMTF) {
      bmtfBX->Fill(BMTF->processor(), itBX);
      bmtfhwPt->Fill(BMTF->hwPt());
      bmtfhwEta->Fill(BMTF->hwEta());
      bmtfhwPhi->Fill(BMTF->hwPhi());
      
      bmtfhwPtvshwEta->Fill(BMTF->hwEta(), BMTF->hwPt());
      bmtfhwPtvshwPhi->Fill(BMTF->hwPhi(), BMTF->hwPt());
      bmtfhwPhivshwEta->Fill(BMTF->hwEta(), BMTF->hwPhi());

      bmtfBXvshwPt->Fill(BMTF->hwPt(), itBX);
      bmtfBXvshwEta->Fill(BMTF->hwEta(), itBX);
      bmtfBXvshwPhi->Fill(BMTF->hwPhi(), itBX);
    }
  }
}

