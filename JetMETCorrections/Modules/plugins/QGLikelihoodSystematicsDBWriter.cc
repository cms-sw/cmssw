// Author: Benedikt Hegner, Tom Cornelis
// Email:  benedikt.hegner@cern.ch, tom.cornelis@cern.ch

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class  QGLikelihoodSystematicsDBWriter : public edm::EDAnalyzer{
 public:
  QGLikelihoodSystematicsDBWriter(const edm::ParameterSet&);
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override {}
  virtual void endJob() override {}
  ~QGLikelihoodSystematicsDBWriter() {}

 private:
  std::string fileName;
  std::string payloadTag;
};

// Constructor
QGLikelihoodSystematicsDBWriter::QGLikelihoodSystematicsDBWriter(const edm::ParameterSet& pSet){
  fileName         = pSet.getParameter<std::string>("src");
  payloadTag       = pSet.getParameter<std::string>("payload");
}

// Begin Job
void QGLikelihoodSystematicsDBWriter::beginJob(){

  QGLikelihoodSystematicsObject *payload = new QGLikelihoodSystematicsObject();
  payload->data.clear();

  std::ifstream database;
  database.open(edm::FileInPath(fileName.c_str()).fullPath().c_str(),std::ios::in);
  if(!database.is_open()){ edm::LogError("FileNotFound") << "Could not open file "<< fileName << std::endl; return;}
  std::string line;
  while(std::getline(database,line)){
    float ptMin, ptMax, etaMin, etaMax, rhoMin, rhoMax, a_q, b_q, a_g, b_g, lmin, lmax;
    char tag[1023],leadchar;
    sscanf(line.c_str(),"%c",&leadchar);
    if((leadchar=='#') || (leadchar=='!')) continue; //Skip those lines
    sscanf(line.c_str(),"%s %f %f %f %f %f %f %f %f %f %f %f %f", &tag[0], &ptMin, &ptMax, &rhoMin, &rhoMax, &etaMin, &etaMax, &a_q, &b_q, &a_g, &b_g, &lmin, &lmax);
      
    QGLikelihoodCategory category;
    category.RhoMin = rhoMin;
    category.RhoMax = rhoMax;
    category.PtMin = ptMin;
    category.PtMax = ptMax;
    category.EtaMin = etaMin;
    category.EtaMax = etaMax;
    category.QGIndex = 0;
    category.VarIndex = -1;

    //quark entry
    QGLikelihoodSystematicsObject::Entry quarkEntry;
    quarkEntry.systCategory = category;
    quarkEntry.a = a_q;
    quarkEntry.b = b_q;
    quarkEntry.lmin = lmin;
    quarkEntry.lmax = lmax;

    //gluon entry
    QGLikelihoodSystematicsObject::Entry gluonEntry = quarkEntry;
    gluonEntry.systCategory.QGIndex = 1;
    gluonEntry.a = a_g;
    gluonEntry.b = b_g;

    payload->data.push_back(quarkEntry);
    payload->data.push_back(gluonEntry);
  }	
  database.close();

  // Now write it into the DB
  edm::LogInfo("UserOutput") << "Opening PoolDBOutputService" << std::endl;
  edm::Service<cond::service::PoolDBOutputService> s;
  if(s.isAvailable()){ 
    edm::LogInfo("UserOutput") <<  "Setting up payload with " << payload->data.size() <<  " entries and tag " << payloadTag << std::endl;
    if(s->isNewTagRequest(payloadTag)) s->createNewIOV<QGLikelihoodSystematicsObject>(payload, s->beginOfTime(), s->endOfTime(), payloadTag);
    else s->appendSinceTime<QGLikelihoodSystematicsObject>(payload, 111, payloadTag);
  }
  edm::LogInfo("UserOutput") <<  "Wrote in CondDB QGLikelihoodSystematic payload label: " << payloadTag << std::endl;
}


DEFINE_FWK_MODULE(QGLikelihoodSystematicsDBWriter);
