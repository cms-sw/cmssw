// -*- C++ -*-
//
// Package:    L1GtRunSettingsViewer
// Class:      L1GtRunSettingsViewer
// 
/**\class L1GtRunSettingsViewer L1GtRunSettingsViewer.cc CondTools/L1GtRunSettingsViewer/src/L1GtRunSettingsViewer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Thu May 19 04:32:54 CEST 2011
//
//


// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

#include "CondTools/L1Trigger/interface/Exception.h"
#include "CondTools/L1Trigger/interface/DataWriter.h"

//
// class decleration
//

class L1GtRunSettingsViewer : public edm::EDAnalyzer {
   public:
      explicit L1GtRunSettingsViewer(const edm::ParameterSet&);
      ~L1GtRunSettingsViewer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
  std::string m_prescalesKey ;
  std::string m_maskAlgoKey ;
  std::string m_maskTechKey ;
  std::string m_maskVetoAlgoKey ;
  std::string m_maskVetoTechKey ;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GtRunSettingsViewer::L1GtRunSettingsViewer(const edm::ParameterSet& iConfig)
  : m_prescalesKey( iConfig.getParameter< std::string >( "prescalesKey" ) ),
    m_maskAlgoKey( iConfig.getParameter< std::string >( "maskAlgoKey" ) ),
    m_maskTechKey( iConfig.getParameter< std::string >( "maskTechKey" ) ),
    m_maskVetoAlgoKey( iConfig.getParameter< std::string >( "maskVetoAlgoKey" ) ),
    m_maskVetoTechKey( iConfig.getParameter< std::string >( "maskVetoTechKey" ) )
{
   //now do what ever initialization is needed
}


L1GtRunSettingsViewer::~L1GtRunSettingsViewer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1GtRunSettingsViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // Utility class
   l1t::DataWriter dataWriter ;

   // Get most recent L1TriggerKeyList
   L1TriggerKeyList keyList ;
   dataWriter.fillLastTriggerKeyList( keyList ) ;

   // For the given GTRS key, find the corresponding payload tokens.
   // Use the payload tokens to retrieve the corresponding objects.
   // Call the print functions for these objects.

   if( !m_prescalesKey.empty() )
     {
       std::string pfAlgoToken = keyList.token( "L1GtPrescaleFactorsAlgoTrigRcd",
						"L1GtPrescaleFactors",
						m_prescalesKey ) ;
       if( pfAlgoToken.empty() )
	 {
	   edm::LogError( "L1-O2O" ) << "No payload for L1GtPrescaleFactorsAlgoTrigRcd with key "
				     << m_prescalesKey ;
	 }
       else
	 {
	   L1GtPrescaleFactors pfAlgo ;
	   dataWriter.readObject( pfAlgoToken, pfAlgo ) ;
	   pfAlgo.print( std::cout ) ;
	 }

       std::string pfTechToken = keyList.token( "L1GtPrescaleFactorsTechTrigRcd",
						"L1GtPrescaleFactors",
						m_prescalesKey ) ;
       if( pfTechToken.empty() )
	 {
	   edm::LogError( "L1-O2O" ) << "No payload for L1GtPrescaleFactorsTechTrigRcd with key "
				     << m_prescalesKey ;
	 }
       else
	 {
	   L1GtPrescaleFactors pfTech ;
	   dataWriter.readObject( pfTechToken, pfTech ) ;
	   pfTech.print( std::cout ) ;
	 }
     }

   if( !m_maskAlgoKey.empty() )
     {
       std::string token = keyList.token( "L1GtTriggerMaskAlgoTrigRcd",
					  "L1GtTriggerMask",
					  m_maskAlgoKey ) ;
       if( token.empty() )
	 {
	   edm::LogError( "L1-O2O" ) << "No payload for L1GtTriggerMaskAlgoTrigRcd with key "
				     << m_maskAlgoKey ;
	 }
       else
	 {
	   L1GtTriggerMask mask ;
	   dataWriter.readObject( token, mask ) ;
	   mask.print( std::cout ) ;
	 }
     }

   if( !m_maskTechKey.empty() )
     {
       std::string token = keyList.token( "L1GtTriggerMaskTechTrigRcd",
					  "L1GtTriggerMask",
					  m_maskTechKey ) ;
       if( token.empty() )
	 {
	   edm::LogError( "L1-O2O" ) << "No payload for L1GtTriggerMaskTechTrigRcd with key "
				     << m_maskTechKey ;
	 }
       else
	 {
	   L1GtTriggerMask mask ;
	   dataWriter.readObject( token, mask ) ;
	   mask.print( std::cout ) ;
	 }
     }

   if( !m_maskVetoAlgoKey.empty() )
     {
       std::string token = keyList.token( "L1GtTriggerMaskVetoAlgoTrigRcd",
					  "L1GtTriggerMask",
					  m_maskVetoAlgoKey ) ;
       if( token.empty() )
	 {
	   edm::LogError( "L1-O2O" ) << "No payload for L1GtTriggerMaskVetoAlgoTrigRcd with key "
				     << m_maskVetoAlgoKey ;
	 }
       else
	 {
	   L1GtTriggerMask mask ;
	   dataWriter.readObject( token, mask ) ;
	   mask.print( std::cout ) ;
	 }
     }

   if( !m_maskVetoTechKey.empty() )
     {
       std::string token = keyList.token( "L1GtTriggerMaskVetoTechTrigRcd",
					  "L1GtTriggerMask",
					  m_maskVetoTechKey ) ;
       if( token.empty() )
	 {
	   edm::LogError( "L1-O2O" ) << "No payload for L1GtTriggerMaskVetoTechTrigRcd with key "
				     << m_maskVetoTechKey ;
	 }
       else
	 {
	   L1GtTriggerMask mask ;
	   dataWriter.readObject( token, mask ) ;
	   mask.print( std::cout ) ;
	 }
     }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1GtRunSettingsViewer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1GtRunSettingsViewer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1GtRunSettingsViewer);
