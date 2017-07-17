#include <memory>
#include <algorithm>
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/RecoMuonObjects/interface/DYTThrObject.h"
#include "CondFormats/DataRecord/interface/DYTThrObjectRcd.h" 
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/DYTInfo.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#define MAX_THR 1e7

class DYTTuner : public edm::EDAnalyzer {
public:
  explicit DYTTuner(const edm::ParameterSet&);
  ~DYTTuner();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void   beginJob() ;
  virtual void   analyze(const edm::Event&, const edm::EventSetup&);
  virtual void   endJob() ;
  virtual double doIntegral(std::vector<double>&, DetId&);
  virtual void   writePlots();
  virtual void   beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void   endRun(edm::Run const&, edm::EventSetup const&);
  virtual void   beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void   endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  typedef edm::ValueMap<reco::DYTInfo> DYTestimators;
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  double MinEnVal, MaxEnVal, IntegralCut, MaxEstVal;
  unsigned int MinNumValues, MaxValPlots, NBinsPlots;
  bool saveROOTfile;
  DYTThrObject* thresholds;
  std::map<DetId, std::vector<double> > mapId;
  std::map<DetId, TH1F*> EstPlots;
  edm::EDGetTokenT<DYTestimators> dytInfoToken;
  edm::EDGetTokenT<reco::MuonCollection> muonsToken;
};


DYTTuner::DYTTuner(const edm::ParameterSet& iConfig)
{
  MinEnVal     = iConfig.getParameter<double>("MinEnergyVal");
  MaxEnVal     = iConfig.getParameter<double>("MaxEnergyVal");
  IntegralCut  = iConfig.getParameter<double>("IntegralCut");
  MaxEstVal    = iConfig.getParameter<double>("MaxEstVal");
  MinNumValues = iConfig.getParameter<unsigned int>("MinNumValues");
  saveROOTfile = iConfig.getParameter<bool>("writePlots");  
  MaxValPlots  = iConfig.getParameter<unsigned int>("MaxValPlots");
  NBinsPlots   = iConfig.getParameter<unsigned int>("NBinsPlots");

  edm::ConsumesCollector iC  = consumesCollector();
  dytInfoToken=iC.consumes<DYTestimators>(edm::InputTag("tevMuons", "dytInfo"));
  muonsToken=iC.consumes<reco::MuonCollection>(edm::InputTag("muons"));

  if (MaxEstVal == -1) MaxEstVal = MAX_THR;

  if (!poolDbService->isNewTagRequest("DYTThrObjectRcd")) 
    throw cms::Exception("NotAvailable") << "The output file already contains a valid \"DYTThrObjectRcd\" record.\nPlease provide a different file name or tag.";
}


DYTTuner::~DYTTuner()
{
}


void DYTTuner::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  Handle<DYTestimators> dytInfoH;
  iEvent.getByToken(dytInfoToken, dytInfoH);
  const DYTestimators &dytInfoC = *dytInfoH;
  Handle<MuonCollection> muons;
  iEvent.getByToken(muonsToken, muons);
  for(size_t i = 0; i != muons->size(); ++i) {
    
    // Energy range cut
    if (muons->at(i).pt() < MinEnVal || muons->at(i).pt() > MaxEnVal) continue;

    const TrackRef& tkRef = muons->at(i).globalTrack();
    if(dytInfoC.contains(tkRef.id())){
      DYTInfo dytInfo = dytInfoC[muons->at(i).globalTrack()];
      vector<double> estimators = dytInfo.DYTEstimators();
      vector<DetId> ids         = dytInfo.IdChambers();
      
      for (int j = 0; j < 4; j++) {
	if (ids[j].null()) continue;
	DetId chamberId = ids[j];
	double estValue = estimators[j];
	if (estValue >= 0 && estValue <= MaxEstVal)
	  mapId[chamberId].push_back(estValue);
      }
    } else {continue;}
  }
}


void DYTTuner::beginJob()
{
}


void DYTTuner::endJob() 
{
  if (saveROOTfile) writePlots();
  thresholds = new DYTThrObject();

  // Full barrel/endcap computation
  std::map<DetId, std::vector<double> >::iterator it;
  std::map<int, std::vector<double> > estBarrel, estEndcap;
  for ( it = mapId.begin() ; it != mapId.end(); it++ ) {
    DetId id = (*it).first;
    std::vector<double> estValCh = (*it).second;
    if ((*it).first.subdetId() == MuonSubdetId::DT) 
      for (unsigned int b = 0; b < estValCh.size(); b++) {
	int station = DTChamberId(id).station();
	estBarrel[station].push_back(estValCh[b]);
      }
    if ((*it).first.subdetId() == MuonSubdetId::CSC) 
      for (unsigned int e = 0; e < estValCh.size(); e++) {
	int station = CSCDetId(id).station();
	estEndcap[station].push_back(estValCh[e]);
      }
  }
  DetId empty = DetId();
  double barrelCut[4], endcapCut[4];
  for (unsigned st = 1; st <= 4; st++) {
    barrelCut[st-1] = doIntegral(estBarrel[st], empty);
    endcapCut[st-1] = doIntegral(estEndcap[st], empty);
  }  

  // Chamber by chamber computation
  for ( it = mapId.begin() ; it != mapId.end(); it++ ) {
    DetId id = (*it).first;
    std::vector<double> estValCh = (*it).second;
    DYTThrObject::DytThrStruct obj;
    obj.id = id;
    if (estValCh.size() < MinNumValues) {
      if (id.subdetId() == MuonSubdetId::DT) {
	int station = DTChamberId(id).station();
	obj.thr = barrelCut[station-1];
      }
      if (id.subdetId() == MuonSubdetId::CSC) {
	int station = CSCDetId(id).station();
	obj.thr = endcapCut[station-1];
      }
      thresholds->thrsVec.push_back(obj);
      continue;
    }
    obj.thr = doIntegral(estValCh, id);
    thresholds->thrsVec.push_back(obj);
  }

  // Writing to DB
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() )
  {
    poolDbService->writeOne( thresholds, poolDbService->beginOfTime(), "DYTThrObjectRcd"  ); 
  }
  else throw cms::Exception("NotAvailable") << "PoolDBOutputService is not available.";
}


double DYTTuner::doIntegral(std::vector<double>& estValues, DetId& id) {
  double cutOnE = -1;
  int nPosVal = 0;
  sort( estValues.begin(), estValues.end() );
  for (unsigned int j = 0; j < estValues.size(); j++) 
    if (estValues[j] > 0 && estValues[j] < MaxEstVal) nPosVal++;
  double limit = nPosVal * IntegralCut;
  int nVal = 0; 
  for (unsigned int j = 0; j < estValues.size(); j++) {
    if (estValues[j] < 0) continue;
    nVal++;
    if (nVal >= limit) {
      cutOnE = estValues[j-1];
      break;
    }
  }
  std::cout << "Det Id: " << id.rawId() << " - Threshold:: " << cutOnE << std::endl;
  return cutOnE;
}


void DYTTuner::writePlots() {
  edm::Service<TFileService> fs;
  std::map<DetId, std::vector<double> >::iterator it;
  for ( it = mapId.begin() ; it != mapId.end(); it++ ) {
    DetId id = (*it).first;
    std::vector<double> estValCh = (*it).second;
    char plotName[200];
    sprintf(plotName, "%i", id.rawId());
    TH1F* tmpPlot = new TH1F(plotName, plotName, NBinsPlots, 0., MaxValPlots);
    for (unsigned int i = 0; i < estValCh.size(); i++) 
      tmpPlot->Fill(estValCh[i]);
    EstPlots[id] = fs->make<TH1F>(*tmpPlot);
    delete tmpPlot;
  }
}


void DYTTuner::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

void DYTTuner::endRun(edm::Run const&, edm::EventSetup const&)
{
}


void DYTTuner::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}


void DYTTuner::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}


void DYTTuner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(DYTTuner);
