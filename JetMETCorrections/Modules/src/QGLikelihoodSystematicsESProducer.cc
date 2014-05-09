/*
 Description: ESProducer to get the quark-gluon likelihood object "QGLikelihoodSystematicsObject"
              from record "QGLikelihoodRcd". 

 Implementation:
     Completely trivial, simply returns the QGLikelihoodSystematicsObject to the user. There is only
     one QGLikelihoodSystematicsObject object in each record. 
*/
// Original Author:  Salvatore Rappoccio
//         Created:  Thu, 13 Mar 2014 15:02:39 GMT


#include "JetMETCorrections/Modules/interface/QGLikelihoodSystematicsESProducer.h"

QGLikelihoodSystematicsESProducer::QGLikelihoodSystematicsESProducer(const edm::ParameterSet& iConfig){
   //the following line is needed to tell the framework what
   // data is being produced
   std::string label(iConfig.getParameter<std::string>("@module_label"));
   mAlgo = iConfig.getParameter<std::string>("algo");
   setWhatProduced(this, label);
}


// The same PDF's is valid for any time
void QGLikelihoodSystematicsESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval& oInterval){
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

// Produce the data
boost::shared_ptr<QGLikelihoodSystematicsObject> QGLikelihoodSystematicsESProducer::produce(const QGLikelihoodSystematicsRcd& iRecord){
   edm::ESHandle<QGLikelihoodSystematicsObject> qglObj;
   iRecord.get(mAlgo, qglObj);

   boost::shared_ptr<QGLikelihoodSystematicsObject> pMyType(new QGLikelihoodSystematicsObject(*qglObj));
   return pMyType;
}
