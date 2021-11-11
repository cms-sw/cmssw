/**
 * \file EcalShowerContainmentAnalyzer
 * 
 * Analyzer to test Shower Containment Corrections
 *   
 * \author S. Argiro'
 *
*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalCorrections/interface/EcalShowerContainmentCorrections.h"
#include "CondFormats/DataRecord/interface/EcalShowerContainmentCorrectionsRcd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "RecoTBCalo/EcalTBAnalysisCoreTools/interface/TBPositionCalc.h"

// geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <map>

class EcalShowerContainmentAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalShowerContainmentAnalyzer(const edm::ParameterSet& ps);
  ~EcalShowerContainmentAnalyzer() override;

  void analyze(edm::Event const& iEvent, const edm::EventSetup& iSetup) override;
  void endJob() override;

protected:
  void readIntercalibrationConstants();

  std::vector<EBDetId> Xtals3x3(const edm::Event& iEvent, EBDetId centerXtal);
  std::vector<EBDetId> Xtals5x5(const edm::Event& iEvent, EBDetId centerXtal);

  double energy3x3(const edm::Event& iEvent, EBDetId centerXtal);
  double energy5x5(const edm::Event& iEvent, EBDetId centerXtal);
  std::pair<double, double> contCorrection(const edm::Event& iEvent,
                                           const edm::EventSetup& iSetup,
                                           const EBDetId& centerXtal);

  /// fill a map of (detid,amplitude)
  void fillxtalmap(edm::Event const& iEvent);

  double getAdcToGevConstant(const edm::EventSetup& eSetup);

  TFile* file_;
  TTree* tree_;

  // map of < xtal, (coefficient, goodness)>
  typedef std::map<EBDetId, std::pair<double, double> > CalibMap;
  CalibMap calibmap_;

  // map of <xtal, amplitude>
  std::map<EBDetId, double> xtalmap_;

  const edm::InputTag inputTag_;
  const std::string intercalibrationCoeffFile_;

  const std::string recHitProducer_;
  const std::string ebRecHitCollection_;

  const edm::EDGetTokenT<EcalTBEventHeader> tbEvtHeaderToken_;
  const edm::EDGetTokenT<EcalTBHodoscopeRecInfo> tbHodoRecInfoToken_;
  const edm::EDGetTokenT<EcalUncalibratedRecHitCollection> uncalibRecHitsToken_;
  const edm::EDGetTokenT<EBRecHitCollection> ebRecHitsToken_;

  const edm::ESGetToken<EcalShowerContainmentCorrections, EcalShowerContainmentCorrectionsRcd> showerContCorrToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  const edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> adcToGeVConstantToken_;

  struct Ntuple {
    int run;
    int event;
    double energy1x1;
    double energy3x3;
    double energy5x5;
    double corrected3x3;
    double corrected5x5;
    double posx;
    double posy;
    double hodo_posx;
    double hodo_posy;

    void zero() {
      run = event = 0;
      energy1x1 = energy3x3 = energy5x5 = corrected3x3 = corrected5x5 = posx = posy = hodo_posx = hodo_posy = 0;
    }

  } ntuple_;
};

DEFINE_FWK_MODULE(EcalShowerContainmentAnalyzer);

EcalShowerContainmentAnalyzer::EcalShowerContainmentAnalyzer(const edm::ParameterSet& ps)
    : inputTag_(ps.getUntrackedParameter<edm::InputTag>("EcalUnCalibratedRecHitsLabel")),
      intercalibrationCoeffFile_(ps.getUntrackedParameter<std::string>("IntercalibFile")),
      recHitProducer_("ecal2006TBRecHit"),
      ebRecHitCollection_("EcalRecHitsEB"),
      tbEvtHeaderToken_(consumes<EcalTBEventHeader>(edm::InputTag("ecalTBunpack"))),
      tbHodoRecInfoToken_(consumes<EcalTBHodoscopeRecInfo>(
          edm::InputTag("ecal2006TBHodoscopeReconstructor", "EcalTBHodoscopeRecInfo"))),
      uncalibRecHitsToken_(consumes<EcalUncalibratedRecHitCollection>(inputTag_)),
      ebRecHitsToken_(consumes<EBRecHitCollection>(edm::InputTag(recHitProducer_, ebRecHitCollection_))),
      geometryToken_(esConsumes()),
      adcToGeVConstantToken_(esConsumes()) {
  std::string outfilename = ps.getUntrackedParameter<std::string>("OutputFile", "testContCorrect.root");

  file_ = new TFile(outfilename.c_str(), "recreate");
  tree_ = new TTree("test", "test");

  tree_->Branch("tb",
                &ntuple_,
                "run/I:"
                "event/I:"
                "energy1x1/D:"
                "energy3x3/D:"
                "energy5x5/D:"
                "corrected3x3/D:"
                "corrected5x5/D:"
                "posx/D:"
                "posy/D:"
                "hodo_posx/D:"
                "hodo_posy/D");

  readIntercalibrationConstants();
}

EcalShowerContainmentAnalyzer::~EcalShowerContainmentAnalyzer() {
  delete file_;
  delete tree_;
}

void EcalShowerContainmentAnalyzer::endJob() {
  file_->cd();
  tree_->Write();
}

void EcalShowerContainmentAnalyzer::analyze(edm::Event const& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  fillxtalmap(iEvent);

  Handle<EcalTBEventHeader> pHeader;
  iEvent.getByToken(tbEvtHeaderToken_, pHeader);

  ntuple_.run = pHeader->runNumber();
  ntuple_.event = pHeader->eventNumber();

  Handle<EcalTBHodoscopeRecInfo> pHodoscope;
  iEvent.getByToken(tbHodoRecInfoToken_, pHodoscope);

  ntuple_.hodo_posx = pHodoscope->posX();
  ntuple_.hodo_posy = pHodoscope->posY();

  Handle<EcalUncalibratedRecHitCollection> pIn;
  iEvent.getByToken(uncalibRecHitsToken_, pIn);

  // pick the hottest xtal

  EcalUncalibratedRecHitCollection::const_iterator iter;
  double maxampl = 0;
  EBDetId maxid;

  for (iter = pIn->begin(); iter != pIn->end(); ++iter) {
    EBDetId detid = iter->id();

    if (iter->amplitude() > maxampl) {
      maxampl = iter->amplitude() * calibmap_[detid].first;
      maxid = iter->id();
    }
  }  // for

  double adctogev = getAdcToGevConstant(iSetup);

  ntuple_.energy1x1 = maxampl * calibmap_[maxid].first * adctogev;
  ntuple_.energy3x3 = energy3x3(iEvent, maxid) * adctogev;
  ntuple_.energy5x5 = energy5x5(iEvent, maxid) * adctogev;

  std::pair<double, double> correction = contCorrection(iEvent, iSetup, maxid);
  ntuple_.corrected3x3 = ntuple_.energy3x3 / correction.first;
  ntuple_.corrected5x5 = ntuple_.energy5x5 / correction.second;

  tree_->Fill();

  ntuple_.zero();
}

/// return vector of detids of the xtals around centerXtal (3x3)
std::vector<EBDetId> EcalShowerContainmentAnalyzer::Xtals3x3(const edm::Event& iEvent, EBDetId centerXtal) {
  using namespace edm;
  using namespace std;

  Handle<EcalUncalibratedRecHitCollection> pIn;
  iEvent.getByToken(uncalibRecHitsToken_, pIn);

  // find the ids of the 3x3 matrix
  vector<EBDetId> Xtals3x3;

  for (unsigned int icry = 0; icry < 9; icry++) {
    unsigned int row = icry / 3;
    unsigned int column = icry % 3;

    try {
      Xtals3x3.push_back(EBDetId(centerXtal.ieta() + column - 1, centerXtal.iphi() + row - 1, EBDetId::ETAPHIMODE));
    } catch (cms::Exception& e) {
      Xtals3x3.clear();
      return Xtals3x3;
    }  //catch
  }    // for
  return Xtals3x3;
}

/// return vector of detids of the xtals around centerXtal (5x5)
std::vector<EBDetId> EcalShowerContainmentAnalyzer::Xtals5x5(const edm::Event& iEvent, EBDetId centerXtal) {
  using namespace edm;
  using namespace std;

  Handle<EcalUncalibratedRecHitCollection> pIn;
  iEvent.getByToken(uncalibRecHitsToken_, pIn);

  // find the ids of the 3x3 matrix
  vector<EBDetId> Xtals5x5;

  for (unsigned int icry = 0; icry < 25; icry++) {
    unsigned int row = icry / 5;
    unsigned int column = icry % 5;

    try {
      Xtals5x5.push_back(EBDetId(centerXtal.ieta() + column - 2, centerXtal.iphi() + row - 2, EBDetId::ETAPHIMODE));
    } catch (cms::Exception& e) {
      Xtals5x5.clear();
      return Xtals5x5;
    }  //catch
  }    // for

  return Xtals5x5;
}

void EcalShowerContainmentAnalyzer::readIntercalibrationConstants() {
  using namespace std;

  ifstream calibrationfile(intercalibrationCoeffFile_.c_str());

  // skip header lines
  int kHeaderLines = 6;
  for (int i = 0; i < kHeaderLines; i++) {
    char line[64];
    calibrationfile.getline(line, 64);
  }

  int xtal;
  double coeff, sigma;
  int nev, good;

  while (calibrationfile >> xtal >> coeff >> sigma >> nev >> good) {
    EBDetId detid(1, xtal, EBDetId::SMCRYSTALMODE);
    calibmap_[detid] = pair<double, double>(coeff, good);
  }
}

void EcalShowerContainmentAnalyzer::fillxtalmap(const edm::Event& iEvent) {
  using namespace edm;

  Handle<EcalUncalibratedRecHitCollection> pIn;
  iEvent.getByToken(uncalibRecHitsToken_, pIn);

  EcalUncalibratedRecHitCollection::const_iterator iter;

  for (iter = pIn->begin(); iter != pIn->end(); ++iter) {
    EBDetId detid = iter->id();
    double amplitude = iter->amplitude();
    xtalmap_[detid] = amplitude;
  }  // for
}

/** return a pair with correction coeff for 3x3 in first place 
    and correction 5x5 in second   */
std::pair<double, double> EcalShowerContainmentAnalyzer::contCorrection(const edm::Event& iEvent,
                                                                        const edm::EventSetup& iESetup,
                                                                        const EBDetId& centerXtal) {
  using namespace std;
  using namespace edm;

  const auto& gapCorr = iESetup.getData(showerContCorrToken_);

  Handle<EBRecHitCollection> pEBRecHits;
  iEvent.getByToken(ebRecHitsToken_, pEBRecHits);
  const EBRecHitCollection* EBRecHits = pEBRecHits.product();

  map<string, double> posparam;
  posparam["LogWeighted"] = 1.0;
  posparam["X0"] = 0.89;
  posparam["T0"] = 6.2;
  posparam["W0"] = 4.0;

  const auto& geometry = iESetup.getData(geometryToken_);
  const CaloSubdetectorGeometry* geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);

  edm::FileInPath mapfile("Geometry/EcalTestBeam/data/BarrelSM1CrystalCenterElectron120GeV.dat");
  TBPositionCalc pos(posparam, mapfile.fullPath(), geometry_p);

  vector<EBDetId> myDets = Xtals3x3(iEvent, centerXtal);

  CLHEP::Hep3Vector clusterPos;
  try {
    clusterPos = pos.CalculateTBPos(myDets, centerXtal.ic(), EBRecHits);
  } catch (cms::Exception& e) {
    return make_pair(-99., -99.);
  }

  ntuple_.posx = clusterPos.x() * 10.;
  ntuple_.posy = clusterPos.y() * 10.;

  math::XYZPoint mathpoint(clusterPos.x(), clusterPos.y(), clusterPos.z());

  double correction3x3 = gapCorr.correction3x3(centerXtal, mathpoint);
  double correction5x5 = gapCorr.correction5x5(centerXtal, mathpoint);

  return std::make_pair(correction3x3, correction5x5);
}

// retrieve adctogev constant for xtalid from the database
double EcalShowerContainmentAnalyzer::getAdcToGevConstant(const edm::EventSetup& eSetup) {
  const auto& ical = eSetup.getData(adcToGeVConstantToken_);
  return ical.getEBValue();
}

double EcalShowerContainmentAnalyzer::energy3x3(const edm::Event& iEvent, EBDetId centerXtal) {
  using namespace edm;
  using namespace std;

  vector<EBDetId> xtals3x3 = Xtals3x3(iEvent, centerXtal);

  // get the energy
  double energy3x3 = 0;
  vector<EBDetId>::iterator detIt;
  for (detIt = xtals3x3.begin(); detIt != xtals3x3.end(); ++detIt) {
    energy3x3 += xtalmap_[*detIt] * calibmap_[*detIt].first;
  }

  return energy3x3;
}

double EcalShowerContainmentAnalyzer::energy5x5(const edm::Event& iEvent, EBDetId centerXtal) {
  using namespace edm;
  using namespace std;

  vector<EBDetId> xtals5x5 = Xtals5x5(iEvent, centerXtal);

  // get the energy
  double energy5x5 = 0;
  vector<EBDetId>::iterator detIt;
  for (detIt = xtals5x5.begin(); detIt != xtals5x5.end(); ++detIt) {
    energy5x5 += xtalmap_[*detIt] * calibmap_[*detIt].first;
  }

  return energy5x5;
}
