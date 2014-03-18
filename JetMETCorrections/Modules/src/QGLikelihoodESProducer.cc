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
   mAlgo             = iConfig.getParameter<std::string>("algo");
   setWhatProduced(this, mAlgo);
   // findingRecordWithKey<QGLikelihoodRcd>();

   std::cout << "QGLikelihoodESProducer is called on " << mAlgo << ", I am " << this << std::endl;
}


QGLikelihoodESProducer::~QGLikelihoodESProducer()
{
  std::cout << "Destructed " << this << std::endl;
}


//
// member functions
//

// void 
// QGLikelihoodESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
// 				       const edm::IOVSyncValue&,
// 				       edm::ValidityInterval& oInterval ) {
//   // the same PDF's is valid for any time
//   oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime (), 
//                                      edm::IOVSyncValue::endOfTime ()) ;
// }

// ------------ method called to produce the data  ------------
QGLikelihoodESProducer::ReturnType
QGLikelihoodESProducer::produce(const QGLikelihoodRcd& iRecord)
{
  std::cout << "QGLProducer producing on " << this << std::endl;
   using namespace edm::es;
   edm::ESHandle<QGLikelihoodObject> qglObj;
   std::cout << "Getting algo " << mAlgo << std::endl;
   iRecord.get(mAlgo, qglObj);
   std::cout << "Done." << std::endl;

   boost::shared_ptr<QGLikelihoodObject> pMyType( new QGLikelihoodObject(*qglObj) );
   std::cout << "Returning" << std::endl;
   return pMyType;

}
