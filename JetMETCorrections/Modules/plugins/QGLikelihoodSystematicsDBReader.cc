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
#include "CondFormats/DataRecord/interface/QGLikelihoodSystematicsRcd.h"

class QGLikelihoodSystematicsDBReader : public edm::EDAnalyzer{
public:
  explicit QGLikelihoodSystematicsDBReader(const edm::ParameterSet&);
  ~QGLikelihoodSystematicsDBReader(){};
  
private:
  virtual void beginJob() override{};
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override{};
 
  std::string mPayloadName;
  bool mCreateTextFile,mPrintScreen;
};


QGLikelihoodSystematicsDBReader::QGLikelihoodSystematicsDBReader(const edm::ParameterSet& iConfig){
  mPayloadName    = iConfig.getUntrackedParameter<std::string>("payloadName");
  mPrintScreen    = iConfig.getUntrackedParameter<bool>("printScreen");
  mCreateTextFile = iConfig.getUntrackedParameter<bool>("createTextFile");
}


void QGLikelihoodSystematicsDBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  edm::LogInfo("UserOutput") << "Inspecting QGLikelihood payload with label:" << mPayloadName << std::endl;
  edm::ESHandle<QGLikelihoodSystematicsObject> QGLSysPar;
  QGLikelihoodSystematicsRcd const& rcdhandle = iSetup.get<QGLikelihoodSystematicsRcd>();
  rcdhandle.get(mPayloadName, QGLSysPar);

  std::vector<QGLikelihoodSystematicsObject::Entry> const& data = QGLSysPar->data;
  edm::LogInfo("UserOutput") <<  "There are " << data.size() << " entries (categories with parameters for smearing):" << std::endl;
  for(auto idata = data.begin(); idata != data.end(); ++idata){    
    int qgBin = idata->systCategory.QGIndex;
    double etaMin = idata->systCategory.EtaMin;
    double etaMax = idata->systCategory.EtaMax;
    double rhoMin = idata->systCategory.RhoMin;
    double rhoMax = idata->systCategory.RhoMax;
    double ptMin = idata->systCategory.PtMin;
    double ptMax = idata->systCategory.PtMax;
    double a = idata->a;
    double b = idata->b;
    double lmin = idata->lmin;
    double lmax = idata->lmax;

    char buff[1000];
    sprintf(buff, "qg=%1d, ptMin=%8.2f, ptMax=%8.2f, etaMin=%3.1f, etaMax=%3.1f, rhoMin=%6.2f, rhoMax=%6.2f, a=%7.3f, b=%7.3f, lmin=%6.2f, lmax=%6.2f", qgBin, ptMin, ptMax, etaMin, etaMax, rhoMin, rhoMax, a, b, lmin, lmax);
    edm::LogVerbatim("UserOutput") << buff << std::endl;
  }
}

DEFINE_FWK_MODULE(QGLikelihoodSystematicsDBReader);
