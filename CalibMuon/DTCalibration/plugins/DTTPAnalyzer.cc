/** \class DTTPAnalyzer
 *
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <string>
#include <map>

//class DTT0;
class DTLayerId;
class DTWireId;
class DTGeometry;
class DTTTrigBaseSync;
class TFile;

class DTTPAnalyzer : public edm::one::EDAnalyzer<> {
public:
  DTTPAnalyzer(const edm::ParameterSet&);
  ~DTTPAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  std::string getHistoName(const DTLayerId&);

  const bool subtractT0_;
  const edm::EDGetTokenT<DTDigiCollection> digiToken_;

  TFile* rootFile_;
  edm::ESHandle<DTGeometry> dtGeom_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
  std::unique_ptr<DTTTrigBaseSync> tTrigSync_;

  // Map of the t0 and sigma histos by layer
  std::map<DTWireId, int> nDigisPerWire_;
  std::map<DTWireId, double> sumWPerWire_;
  std::map<DTWireId, double> sumW2PerWire_;
};

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"

#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "CondFormats/DTObjects/interface/DTT0.h"

#include "TH1F.h"
#include "TFile.h"

DTTPAnalyzer::DTTPAnalyzer(const edm::ParameterSet& pset)
    : subtractT0_(pset.getParameter<bool>("subtractT0")),
      digiToken_(consumes<DTDigiCollection>(pset.getParameter<edm::InputTag>("digiLabel"))),
      dtGeomToken_(esConsumes()) {
  std::string rootFileName = pset.getUntrackedParameter<std::string>("rootFileName");
  rootFile_ = new TFile(rootFileName.c_str(), "RECREATE");
  rootFile_->cd();

  if (subtractT0_)
    tTrigSync_ = DTTTrigSyncFactory::get()->create(pset.getParameter<std::string>("tTrigMode"),
                                                   pset.getParameter<edm::ParameterSet>("tTrigModeConfig"),
                                                   consumesCollector());
}

DTTPAnalyzer::~DTTPAnalyzer() { rootFile_->Close(); }

void DTTPAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // Get the digis from the event
  const edm::Handle<DTDigiCollection>& digis = event.getHandle(digiToken_);

  if (subtractT0_) {
    tTrigSync_->setES(setup);
  }
  // Get the DT Geometry
  dtGeom_ = setup.getHandle(dtGeomToken_);

  // Iterate through all digi collections ordered by LayerId
  DTDigiCollection::DigiRangeIterator dtLayerIt;
  for (dtLayerIt = digis->begin(); dtLayerIt != digis->end(); ++dtLayerIt) {
    // Get the iterators over the digis associated with this LayerId
    const DTDigiCollection::Range& digiRange = (*dtLayerIt).second;

    // Get the layerId
    const DTLayerId layerId = (*dtLayerIt).first;  //FIXME: check to be in the right sector

    // Loop over all digis in the given layer
    for (DTDigiCollection::const_iterator digi = digiRange.first; digi != digiRange.second; ++digi) {
      const DTWireId wireId(layerId, (*digi).wire());

      double t0 = (*digi).countsTDC();

      //FIXME: Reject digis not coming from TP

      if (subtractT0_) {
        const DTLayer* layer = nullptr;  //fake
        const GlobalPoint glPt;          //fake
        double offset = tTrigSync_->offset(layer, wireId, glPt);
        t0 -= offset;
      }

      if (nDigisPerWire_.find(wireId) == nDigisPerWire_.end()) {
        nDigisPerWire_[wireId] = 0;
        sumWPerWire_[wireId] = 0.;
        sumW2PerWire_[wireId] = 0.;
      }

      ++nDigisPerWire_[wireId];
      sumWPerWire_[wireId] += t0;
      sumW2PerWire_[wireId] += t0 * t0;
    }
  }
}

void DTTPAnalyzer::endJob() {
  rootFile_->cd();
  std::map<DTLayerId, TH1F*> meanHistoMap;
  std::map<DTLayerId, TH1F*> sigmaHistoMap;
  for (std::map<DTWireId, int>::const_iterator wireIdIt = nDigisPerWire_.begin(); wireIdIt != nDigisPerWire_.end();
       ++wireIdIt) {
    DTWireId wireId((*wireIdIt).first);

    int nDigis = nDigisPerWire_[wireId];
    double sumW = sumWPerWire_[wireId];
    double sumW2 = sumW2PerWire_[wireId];

    double mean = sumW / nDigis;
    double rms = sumW2 / nDigis - mean * mean;
    rms = sqrt(rms);

    DTLayerId layerId = wireId.layerId();
    if (meanHistoMap.find(layerId) == meanHistoMap.end()) {
      std::string histoName = getHistoName(layerId);
      const int firstChannel = dtGeom_->layer(layerId)->specificTopology().firstChannel();
      const int nWires = dtGeom_->layer(layerId)->specificTopology().channels();
      TH1F* meanHistoTP = new TH1F((histoName + "_tpMean").c_str(),
                                   "mean from test pulses by channel",
                                   nWires,
                                   firstChannel,
                                   (firstChannel + nWires));
      TH1F* sigmaHistoTP = new TH1F((histoName + "_tpSigma").c_str(),
                                    "sigma from test pulses by channel",
                                    nWires,
                                    firstChannel,
                                    (firstChannel + nWires));
      meanHistoMap[layerId] = meanHistoTP;
      sigmaHistoMap[layerId] = sigmaHistoTP;
    }
    // Fill the histograms
    int nBin = meanHistoMap[layerId]->GetXaxis()->FindFixBin(wireId.wire());
    meanHistoMap[layerId]->SetBinContent(nBin, mean);
    sigmaHistoMap[layerId]->SetBinContent(nBin, rms);
  }

  for (std::map<DTLayerId, TH1F*>::const_iterator key = meanHistoMap.begin(); key != meanHistoMap.end(); ++key) {
    meanHistoMap[(*key).first]->Write();
    sigmaHistoMap[(*key).first]->Write();
  }
}

std::string DTTPAnalyzer::getHistoName(const DTLayerId& lId) {
  std::string histoName;
  std::stringstream theStream;
  theStream << "Ch_" << lId.wheel() << "_" << lId.station() << "_" << lId.sector() << "_SL" << lId.superlayer() << "_L"
            << lId.layer();
  theStream >> histoName;
  return histoName;
}

DEFINE_FWK_MODULE(DTTPAnalyzer);
