#ifndef JetMETCorrections_Modules_QGLikelihoodESProducer_h
#define JetMETCorrections_Modules_QGLikelihoodESProducer_h

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


// system include files
#include <memory>
#include <iostream>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ESProducts.h"
#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "CondFormats/DataRecord/interface/QGLikelihoodRcd.h"


//
// class declaration
//

class QGLikelihoodESProducer : public edm::ESProducer { //, public  edm::EventSetupRecordIntervalFinder {
   public:
      QGLikelihoodESProducer(const edm::ParameterSet&);
      ~QGLikelihoodESProducer();

      typedef boost::shared_ptr<QGLikelihoodObject> ReturnType;

      ReturnType produce(const QGLikelihoodRcd&);

      /// set validity interval
      void setIntervalFor( const edm::eventsetup::EventSetupRecordKey &,
      			   const edm::IOVSyncValue &,
      			   edm::ValidityInterval & );
   private:
      // ----------member data ---------------------------
      std::string mAlgo;
};

#endif 

