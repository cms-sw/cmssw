// -*- C++ -*-
//
// Package:    L1CaloInputScalesProducer
// Class:      L1CaloInputScalesProducer
// 
/**\class L1CaloInputScalesProducer L1CaloInputScalesProducer.h L1TriggerConfig/RCTConfigProducers/src/L1CaloInputScalesProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Fri May 16 16:09:43 CEST 2008
// $Id: L1CaloInputScalesProducer.cc,v 1.2 2008/10/20 17:11:55 bachtis Exp $
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1CaloInputScalesProducer.h"

//
// class decleration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1CaloInputScalesProducer::L1CaloInputScalesProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this, &L1CaloInputScalesProducer::produceEcalScale);
   setWhatProduced(this, &L1CaloInputScalesProducer::produceHcalScale);

   //now do what ever other initialization is needed

   // { Et for each rank, eta bin 0 }, { Et for each rank, eta bin 1 }, ...
   m_ecalEtThresholdsPosEta =
     iConfig.getParameter< std::vector<double> >("L1EcalEtThresholdsPositiveEta");
   m_ecalEtThresholdsNegEta =
     iConfig.getParameter< std::vector<double> >("L1EcalEtThresholdsNegativeEta");
   m_hcalEtThresholdsPosEta =
     iConfig.getParameter< std::vector<double> >("L1HcalEtThresholdsPositiveEta");
   m_hcalEtThresholdsNegEta =
     iConfig.getParameter< std::vector<double> >("L1HcalEtThresholdsNegativeEta");
}


L1CaloInputScalesProducer::~L1CaloInputScalesProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
boost::shared_ptr<L1CaloEcalScale>
L1CaloInputScalesProducer::produceEcalScale(const L1CaloEcalScaleRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1CaloEcalScale> pL1CaloEcalScale =
     boost::shared_ptr<L1CaloEcalScale>( new L1CaloEcalScale ) ;

   std::vector< double >::const_iterator posItr =
     m_ecalEtThresholdsPosEta.begin() ;
   std::vector< double >::const_iterator negItr =
     m_ecalEtThresholdsNegEta.begin() ;

   for( unsigned short ieta = 1 ;
	ieta <= L1CaloEcalScale::nBinEta ;
	++ieta )
     {
       for( unsigned short irank = 0 ;
	    irank < L1CaloEcalScale::nBinRank;
	    ++irank )
	 {
	   pL1CaloEcalScale->setBin( irank, ieta, 1, *posItr ) ;
	   pL1CaloEcalScale->setBin( irank, ieta, -1, *negItr ) ;

	   ++posItr ;
	   ++negItr ;
	 }
     }

   return pL1CaloEcalScale ;
}

// ------------ method called to produce the data  ------------
boost::shared_ptr<L1CaloHcalScale>
L1CaloInputScalesProducer::produceHcalScale(const L1CaloHcalScaleRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1CaloHcalScale> pL1CaloHcalScale =
     boost::shared_ptr<L1CaloHcalScale>( new L1CaloHcalScale ) ;

   std::vector< double >::const_iterator posItr =
     m_hcalEtThresholdsPosEta.begin() ;

   std::vector< double >::const_iterator negItr =
     m_hcalEtThresholdsNegEta.begin() ;


   for( unsigned short ieta = 1 ;
	ieta <= L1CaloHcalScale::nBinEta ;
	++ieta )
     {
       for( unsigned short irank = 0 ;
	    irank < L1CaloHcalScale::nBinRank;
	    ++irank )
	 {
	   pL1CaloHcalScale->setBin( irank, ieta, 1, *posItr ) ;
	   pL1CaloHcalScale->setBin( irank, ieta, -1, *negItr ) ;

	   ++posItr ;
	   ++negItr ;
	 }
     }

   return pL1CaloHcalScale ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1CaloInputScalesProducer);
