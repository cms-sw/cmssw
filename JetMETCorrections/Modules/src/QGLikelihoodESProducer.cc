/*
 Description: ESProducer to get the quark-gluon likelihood object "QGLikelihoodObject"
              from record "QGLikelihoodRcd". 

 Implementation:
     Completely trivial, simply returns the QGLikelihoodObject to the user. There is only
     one QGLikelihoodObject object in each record. 
*/
// Original Author:  Salvatore Rappoccio
//         Created:  Thu, 13 Mar 2014 15:02:39 GMT

#include "JetMETCorrections/Modules/interface/QGLikelihoodESProducer.h"

QGLikelihoodESProducer::QGLikelihoodESProducer(const edm::ParameterSet& iConfig){
   //the following line is needed to tell the framework what
   // data is being produced
   std::string label	= iConfig.getParameter<std::string>("@module_label");
   mAlgo 		= iConfig.getParameter<std::string>("algo");
   setWhatProduced(this, label);
}

// The same PDF's are valid for any time
void QGLikelihoodESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval& oInterval){
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

// Produce the data
boost::shared_ptr<QGLikelihoodObject> QGLikelihoodESProducer::produce(const QGLikelihoodRcd& iRecord){
   edm::ESHandle<QGLikelihoodObject> qglObj;
   iRecord.get(mAlgo, qglObj);

   boost::shared_ptr<QGLikelihoodObject> pMyType(new QGLikelihoodObject(*qglObj));
   return pMyType;
}
