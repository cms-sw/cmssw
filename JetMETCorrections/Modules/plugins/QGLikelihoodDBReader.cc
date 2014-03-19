// -*- C++ -*-
//
// Package:    QGLikelihoodDBReader
// Class:      
// 
/**\class QGLikelihoodDBReader

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Salvatore Rappoccio
//
//


// system include files
#include <memory>

// user include files
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
//
// class declaration
//

class QGLikelihoodDBReader : public edm::EDAnalyzer {
public:
  explicit QGLikelihoodDBReader(const edm::ParameterSet&);
  ~QGLikelihoodDBReader();
  
  
private:
  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;
 
  std::string mPayloadName;
  bool mCreateTextFile,mPrintScreen;
};


QGLikelihoodDBReader::QGLikelihoodDBReader(const edm::ParameterSet& iConfig)
{
  mPayloadName    = iConfig.getUntrackedParameter<std::string>("payloadName");
  mPrintScreen    = iConfig.getUntrackedParameter<bool>("printScreen");
  mCreateTextFile = iConfig.getUntrackedParameter<bool>("createTextFile");
}


QGLikelihoodDBReader::~QGLikelihoodDBReader()
{
 
}

void QGLikelihoodDBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::LogInfo   ("UserOutput") <<  "Getting QGL objects from DB" << std::endl;
  edm::ESHandle<QGLikelihoodObject> QGLParamsColl;
  edm::LogInfo   ("UserOutput") << "Inspecting QGLikelihood payload with label: "<< mPayloadName <<std::endl;
  QGLikelihoodRcd const & rcdhandle = iSetup.get<QGLikelihoodRcd>();
  rcdhandle.get(mPayloadName,QGLParamsColl);
  std::vector<QGLikelihoodObject::Entry> const & data = QGLParamsColl->data;
  edm::LogInfo   ("UserOutput") <<  "There are " << data.size() << " objects in this payload" << std::endl;
  for ( auto ibegin = data.begin(),
	  iend = data.end(), idata = ibegin; idata != iend; ++idata ) {    
    int varIndex = idata->category.VarIndex;
    int qgBin = idata->category.QGIndex;
    int etaBin = idata->category.EtaBin;
    double rhoVal = idata->category.RhoVal;
    double ptMin = idata->category.PtMin;
    double ptMax = idata->category.PtMax;
    // Print out for debugging
    char buff[1000];
    sprintf( buff, "var=%1d, eta=%1d, qg=%1d, ptMin=%8.2f, ptMax=%8.2f, rhoVal=%6.2f", varIndex, etaBin, qgBin, ptMin, ptMax, rhoVal );
    edm::LogVerbatim   ("UserOutput") << buff << std::endl;
    
  }
}

void 
QGLikelihoodDBReader::beginJob()
{
}

void 
QGLikelihoodDBReader::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(QGLikelihoodDBReader);
