// -*- C++ -*-
//
// Package:    JetMETCorrections/QGLikelihoodESProducer
// Class:      QGLikelihoodESProducer
// 
/**\class QGLikelihoodESProducer QGLikelihoodESProducer.h JetMETCorrections/QGLikelihoodESProducer/plugins/QGLikelihoodESProducer.cc

 Description: ESProducer to get the quark-gluon likelihood object "QGLikelihoodObject"
              from record "QGLikelihoodRcd". 

 Implementation:
     Completely trivial, simply returns the QGLikelihoodObject to the user. There is only
     one QGLikelihoodObject object in each record. 
*/
//
// Original Author:  Salvatore Rappoccio
//         Created:  Thu, 13 Mar 2014 15:02:39 GMT
//
//


#include "JetMETCorrections/Modules/interface/QGLikelihoodESProducer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
QGLikelihoodESProducer::QGLikelihoodESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   std::string label(iConfig.getParameter<std::string>("@module_label"));
   mAlgo             = iConfig.getParameter<std::string>("algo");
   setWhatProduced(this, label);
   //findingRecordWithKey<QGLikelihoodRcd>();
}


QGLikelihoodESProducer::~QGLikelihoodESProducer()
{
}


//
// member functions
//

void 
QGLikelihoodESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
				       const edm::IOVSyncValue&,
				       edm::ValidityInterval& oInterval ) {
  // the same PDF's is valid for any time
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime (), 
                                     edm::IOVSyncValue::endOfTime ()) ;
}

// ------------ method called to produce the data  ------------
QGLikelihoodESProducer::ReturnType
QGLikelihoodESProducer::produce(const QGLikelihoodRcd& iRecord)
{
   using namespace edm::es;
   // boost::shared_ptr<QGLikelihoodObject> pQGLikelihoodObject;
   // return products(pQGLikelihoodObject);   

   edm::ESHandle<QGLikelihoodObject> qglObj;
   iRecord.get(mAlgo, qglObj);

   boost::shared_ptr<QGLikelihoodObject> pMyType( new QGLikelihoodObject(*qglObj) );
   return pMyType;

}
