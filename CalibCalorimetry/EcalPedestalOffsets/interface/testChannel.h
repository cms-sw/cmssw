#ifndef testChannel_H
#define testChannel_H

/**
 * \file testChannel.h
 * \class testChannel
 * \brief calculate the best DAC value to obtain a pedestal = 200
 * $Date: 2011/05/20 17:17:23 $
 * $Revision: 1.7 $
 * \author P. Govoni (testChannel.govoni@cernNOSPAM.ch)
 *
*/

#include <map>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedValues.h"
#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedResult.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TROOT.h"
#include "TStyle.h"

#include "TH2.h"
#include "TProfile.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class testChannel: public edm::EDAnalyzer
{

  public:
    
    //! Constructor
    testChannel (const edm::ParameterSet& ps) ;
    
    //! Destructor
    virtual ~testChannel () ;
    
    //! Subscribe/Unsubscribe to Monitoring Elements
    void subscribe (void) ;
    void subscribeNew (void) ;
    void unsubscribe (void) ;
    
    ///! Analyze
    void analyze (edm::Event const& event, edm::EventSetup const& eventSetup) ;
    
    //! BeginJob
    void beginJob () ;
    
    //! EndJob
    void endJob (void) ;
        
  private:
 
    int getHeaderSMId (const int headerId) ;
    
    std::string m_digiCollection ; //! secondary name given to collection of digis
    std::string m_digiProducer ;   //! name of module/plugin/producer making digis
    std::string m_headerProducer ; //! name of module/plugin/producer making headers

    std::string m_xmlFile ;        //! name of the xml file to be saved

    int m_DACmin ;
    int m_DACmax ;
    double m_RMSmax ;
    int m_bestPed ;
    
    int m_xtal ;

    TH2F m_pedVSDAC ;
    TH2F m_singlePedVSDAC_1 ;
    TH2F m_singlePedVSDAC_2 ;
    TH2F m_singlePedVSDAC_3 ;
    
} ; 

#endif
