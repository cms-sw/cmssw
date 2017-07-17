#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "CondFormats/DataRecord/interface/QGLikelihoodRcd.h"

class QGLikelihoodDBReader : public edm::EDAnalyzer{
public:
  explicit QGLikelihoodDBReader(const edm::ParameterSet&);
  ~QGLikelihoodDBReader(){};
  
private:
  virtual void beginJob() override{};
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override{};
 
  std::string mPayloadName;
  bool mCreateTextFile,mPrintScreen;
};


QGLikelihoodDBReader::QGLikelihoodDBReader(const edm::ParameterSet& iConfig){
  mPayloadName    = iConfig.getUntrackedParameter<std::string>("payloadName");
  mPrintScreen    = iConfig.getUntrackedParameter<bool>("printScreen");
  mCreateTextFile = iConfig.getUntrackedParameter<bool>("createTextFile");
}


void QGLikelihoodDBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  edm::LogInfo("UserOutput") << "Inspecting QGLikelihood payload with label:" << mPayloadName << std::endl;
  edm::ESHandle<QGLikelihoodObject> QGLParamsColl;
  QGLikelihoodRcd const& rcdhandle = iSetup.get<QGLikelihoodRcd>();
  rcdhandle.get(mPayloadName, QGLParamsColl);

  edm::LogInfo("UserOutput") << "Ranges in which the QGTagger could be applied:"
			     << "  pt: " << QGLParamsColl->qgValidRange.PtMin << " --> " << QGLParamsColl->qgValidRange.PtMax
			     << ", eta: " << QGLParamsColl->qgValidRange.EtaMin << " --> " << QGLParamsColl->qgValidRange.EtaMax
			     << ", rho: " << QGLParamsColl->qgValidRange.RhoMin << " --> " << QGLParamsColl->qgValidRange.RhoMax << std::endl;

  std::vector<QGLikelihoodObject::Entry> const& data = QGLParamsColl->data;
  edm::LogInfo("UserOutput") <<  "There are " << data.size() << " entries (categories with associated PDF):" << std::endl;
  for(auto idata = data.begin(); idata != data.end(); ++idata){    
    int varIndex = idata->category.VarIndex;
    int qgBin = idata->category.QGIndex;
    double etaMin = idata->category.EtaMin;
    double etaMax = idata->category.EtaMax;
    double rhoMin = idata->category.RhoMin;
    double rhoMax = idata->category.RhoMax;
    double ptMin = idata->category.PtMin;
    double ptMax = idata->category.PtMax;

    char buff[1000];
    sprintf(buff, "var=%1d, qg=%1d, ptMin=%8.2f, ptMax=%8.2f, etaMin=%3.1f, etaMax=%3.1f, rhoMin=%6.2f, rhoMax=%6.2f", varIndex, qgBin, ptMin, ptMax, etaMin, etaMax, rhoMin, rhoMax);
    edm::LogVerbatim("UserOutput") << buff << std::endl;
  }
}

DEFINE_FWK_MODULE(QGLikelihoodDBReader);
