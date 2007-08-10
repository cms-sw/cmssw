/**
 * \file testChannel.cc
 *
 * $Date: 2007/04/23 07:41:33 $
 * $Revision: 1.6 $
 * \author P. Govoni (pietro.govoni@cernNOSPAM.ch)
 *
*/


#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include "TFile.h"

#include "CalibCalorimetry/EcalPedestalOffsets/interface/testChannel.h"
#include <fstream>
#include <iostream>

//! ctor
testChannel::testChannel (const ParameterSet& paramSet) :
  m_digiCollection (paramSet.getParameter<std::string> ("digiCollection")) ,
  m_digiProducer (paramSet.getParameter<std::string> ("digiProducer")) ,
  m_headerProducer (paramSet.getParameter<std::string> ("headerProducer")) ,
  m_xmlFile (paramSet.getParameter<std::string> ("xmlFile")) ,
  m_DACmin (paramSet.getParameter<int> ("DACmin")) ,
  m_DACmax (paramSet.getParameter<int> ("DACmax")) ,
  m_RMSmax (paramSet.getParameter<double> ("RMSmax")) ,
  m_bestPed (paramSet.getParameter<int> ("bestPed")) ,
  m_xtal (paramSet.getParameter<int> ("xtal")) ,
  m_pedVSDAC ("pedVSDAC","pedVSDAC",100,150,250,m_DACmax-m_DACmin,m_DACmin,m_DACmax) ,
  m_singlePedVSDAC_1 ("singlePedVSDAC_1","pedVSDAC (g1) for xtal "+m_xtal,100,150,250,m_DACmax-m_DACmin,m_DACmin,m_DACmax) ,
  m_singlePedVSDAC_2 ("singlePedVSDAC_2","pedVSDAC (g2) for xtal "+m_xtal,100,150,250,m_DACmax-m_DACmin,m_DACmin,m_DACmax) ,
  m_singlePedVSDAC_3 ("singlePedVSDAC_3","pedVSDAC (g3) for xtal "+m_xtal,100,150,250,m_DACmax-m_DACmin,m_DACmin,m_DACmax)
{
  edm::LogInfo ("testChannel") << " reading "
                               << " m_DACmin: " << m_DACmin
                               << " m_DACmax: " << m_DACmax
                               << " m_RMSmax: " << m_RMSmax
                               << " m_bestPed: " << m_bestPed ;
}


//! dtor
testChannel::~testChannel ()
{
}


//! begin the job
void testChannel::beginJob (EventSetup const& eventSetup)
{
   LogDebug ("testChannel") << "entering beginJob ..." ;
}


//! perform te analysis
void testChannel::analyze (Event const& event, 
                           EventSetup const& eventSetup) 
{
   LogDebug ("testChannel") << "entering analyze ..." ;

   // get the headers
   // (one header for each supermodule)
   edm::Handle<EcalRawDataCollection> DCCHeaders ;
   try {
     event.getByLabel (m_headerProducer, DCCHeaders) ;
   } catch ( std::exception& ex ) {
     edm::LogError ("testChannel") << "Error! can't get the product " 
                                   << m_headerProducer.c_str () ;
   }

   std::map <int,int> DACvalues ;
   
   // loop over the headers
   for ( EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin () ;
         headerItr != DCCHeaders->end () ; 
	     ++headerItr ) 
     {
       EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings () ;
       DACvalues[getHeaderSMId (headerItr->id ())] = settings.ped_offset ;
//       std::cout << "DCCid: " << headerItr->id () << "" ;
//       std::cout << "Ped offset DAC: " << settings.ped_offset << "" ;
     } //! loop over the headers

   // get the digis
   // (one digi for each crystal)
   Handle<EBDigiCollection> pDigis;
   try {
     event.getByLabel (m_digiProducer, pDigis) ;
   } catch ( std::exception& ex ) 
   {
     edm::LogError ("testChannel") << "Error! can't get the product " 
                                   << m_digiCollection.c_str () ;
   }
   
   // loop over the digis
   for (EBDigiCollection::const_iterator itdigi = pDigis->begin () ; 
        itdigi != pDigis->end () ; 
        ++itdigi) 
    {    
       EBDataFrame df( *itdigi );
       int gainId = df.sample (0).gainId () ;
       int crystalId = EBDetId(itdigi->id ()).ic () ;
       int smId = EBDetId(itdigi->id ()).ism () ;

       edm::LogInfo ("testChannel") << "channel " << event.id ()  
                                    << "\tcry: " << crystalId 
                                    << "\tG: " << gainId 
                                    << "\tDAC: " << DACvalues[smId] ;

       // loop over the samples
       for (int iSample = 0; iSample < EBDataFrame::MAXSAMPLES ; ++iSample) 
         {
            edm::LogInfo ("testChannel") << "\t`-->" << df.sample (iSample).adc () ;
            m_pedVSDAC.Fill (df.sample (iSample).adc (),DACvalues[smId]) ;
            if (crystalId == m_xtal)
              { 
                if (gainId == 1) m_singlePedVSDAC_1.Fill (df.sample (iSample).adc (),DACvalues[smId]) ;
                if (gainId == 2) m_singlePedVSDAC_2.Fill (df.sample (iSample).adc (),DACvalues[smId]) ;
                if (gainId == 3) m_singlePedVSDAC_3.Fill (df.sample (iSample).adc (),DACvalues[smId]) ;
              }
         } // loop over the samples
    } // loop over the digis

}

//! perform the minimiation and write results
void testChannel::endJob () 
{
  char ccout[80] ;
  sprintf (ccout,"out_%d.root",m_xtal) ;
  TFile out (ccout,"RECREATE") ;
  out.cd () ;
  m_pedVSDAC.Write () ;
  m_singlePedVSDAC_1.Write () ;
  m_singlePedVSDAC_2.Write () ;
  m_singlePedVSDAC_3.Write () ;
  TProfile * profilo1 = m_singlePedVSDAC_1.ProfileX () ;
  TProfile * profilo2 = m_singlePedVSDAC_2.ProfileX () ;
  TProfile * profilo3 = m_singlePedVSDAC_3.ProfileX () ;
  profilo1->Write ("singleProfile_1") ;
  profilo2->Write ("singleProfile_2") ;
  profilo3->Write ("singleProfile_3") ;
  out.Close () ;
}

/*
void testChannel::writeDb (EcalCondDBInterface* econn, 
                           MonRunIOV* moniov) 
{}
*/

int testChannel::getHeaderSMId (const int headerId) 
{
  //PG FIXME temporary solution
  //PG FIXME check it is consistent with the TB!
  return 1 ;
}




void testChannel::subscribe ()
{}

void testChannel::subscribeNew ()
{}

void testChannel::unsubscribe ()
{}


