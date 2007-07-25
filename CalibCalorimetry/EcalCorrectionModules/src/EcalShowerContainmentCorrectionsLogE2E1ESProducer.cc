// -*- C++ -*-
//$Id$

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"


#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalCorrections/interface/EcalShowerContainmentCorrectionsLogE2E1.h"
#include "CondFormats/DataRecord/interface/EcalShowerContainmentCorrectionsLogE2E1Rcd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "CalibCalorimetry/EcalCorrectionModules/interface/EcalShowerContainmentCorrectionsLogE2E1ESProducer.h"

EcalShowerContainmentCorrectionsLogE2E1ESProducer::EcalShowerContainmentCorrectionsLogE2E1ESProducer(const edm::ParameterSet& iConfig)
{   
   setWhatProduced(this);
}


EcalShowerContainmentCorrectionsLogE2E1ESProducer::~EcalShowerContainmentCorrectionsLogE2E1ESProducer(){ }


//
// member functions
//

EcalShowerContainmentCorrectionsLogE2E1ESProducer::ReturnType
EcalShowerContainmentCorrectionsLogE2E1ESProducer::produce(const EcalShowerContainmentCorrectionsLogE2E1Rcd& iRecord)
{

   using namespace edm::es;
   using namespace std;

   auto_ptr<EcalShowerContainmentCorrectionsLogE2E1> pEcalShowerContainmentCorrectionsLogE2E1(new EcalShowerContainmentCorrectionsLogE2E1) ;
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
     EcalShowerContainmentCorrectionsLogE2E1::Coefficients coeff;
     std::copy(values,values+size,coeff.data);

     EBDetId id(sm,xtal,EBDetId::SMCRYSTALMODE);
     
     // we are filling always the same group ...
     pEcalShowerContainmentCorrectionsLogE2E1->fillCorrectionCoefficients(id,3,coeff);
   }

   return pEcalShowerContainmentCorrectionsLogE2E1 ;
}
