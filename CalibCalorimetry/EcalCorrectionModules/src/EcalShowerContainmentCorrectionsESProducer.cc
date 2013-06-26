// -*- C++ -*-
//
// Package:    EcalShowerContainmentCorrectionsESProducer
// Class:      EcalShowerContainmentCorrectionsESProducer
// 
/**\class EcalShowerContainmentCorrectionsESProducer EcalShowerContainmentCorrectionsESProducer.h User/EcalShowerContainmentCorrectionsESProducer/interface/EcalShowerContainmentCorrectionsESProducer.h

 Description: Trivial ESProducer to provide EventSetup with (hard coded)
              shower containment corrections

     
 \author  Stefano Argiro
         Created:  Mon Mar  5 08:39:12 CET 2007
 \id $Id: EcalShowerContainmentCorrectionsESProducer.cc,v 1.2 2007/07/13 17:44:28 meridian Exp $
*/

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalCorrections/interface/EcalShowerContainmentCorrections.h"
#include "CondFormats/DataRecord/interface/EcalShowerContainmentCorrectionsRcd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"



class EcalShowerContainmentCorrectionsESProducer : public edm::ESProducer {

   public:
      EcalShowerContainmentCorrectionsESProducer(const edm::ParameterSet&);
     ~EcalShowerContainmentCorrectionsESProducer();

      typedef std::auto_ptr<EcalShowerContainmentCorrections> ReturnType;

      ReturnType produce(const EcalShowerContainmentCorrectionsRcd&);
   private:
  

};


EcalShowerContainmentCorrectionsESProducer::EcalShowerContainmentCorrectionsESProducer(const edm::ParameterSet& iConfig)
{   
   setWhatProduced(this);
}


EcalShowerContainmentCorrectionsESProducer::~EcalShowerContainmentCorrectionsESProducer(){ }


//
// member functions
//

EcalShowerContainmentCorrectionsESProducer::ReturnType
EcalShowerContainmentCorrectionsESProducer::produce(const EcalShowerContainmentCorrectionsRcd& iRecord)
{

   using namespace edm::es;
   using namespace std;

   auto_ptr<EcalShowerContainmentCorrections> pEcalShowerContainmentCorrections(new EcalShowerContainmentCorrections) ;
   int sm=1; // in testbeam data sw believes we always are on sm01

   // where is the n of xtals per sm coded ?
   for (int xtal=1; xtal<=1700 ; ++xtal){

     //     // from  /afs/cern.ch/user/h/h4ecal/h4ana/data/gapCorrections/parametres_pol6_X204_2_1.out

 // corrections computed on module3 - (sm16, 1run)

     double values[] = {   0.998959,       // 3x3 x right	  
			   0.00124547,	  
			  -0.000348259,
			   6.04065e-006,  
			   0.999032,       // 3x3 x left
			   7.90628e-005,
			  -0.000175699,
			   -2.60715e-007,
                           //
			   0.999983,       // 3x3 y right
			 -0.000132085,
			  2.04773e-005,
			 -1.21629e-005,
			   1.00002,        // 3x3 y left 
			  0.00016518,
			  5.36343e-005,
			  1.32094e-005, 
			   //
			   0.998944,	  // 5x5
			   0.00100987,
			   -0.000223207,
			   2.15615e-006,
			   0.999127,   
			   0.000253437,
			   -9.80656e-005, 
			   1.48651e-006,
			   1.00006,
			   -0.000179675,
			   8.15627e-005,
			   -1.21549e-005,
			   1.00022,
			   0.000363728,
			   0.000128066,
			   1.54473e-005 };




     const size_t size = sizeof values / sizeof values[0];
     EcalShowerContainmentCorrections::Coefficients coeff;
     std::copy(values,values+size,coeff.data);

     EBDetId id(sm,xtal,EBDetId::SMCRYSTALMODE);
     
     // we are filling always the same group ...
     pEcalShowerContainmentCorrections->fillCorrectionCoefficients(id,3,coeff);
   }

   return pEcalShowerContainmentCorrections ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalShowerContainmentCorrectionsESProducer);
