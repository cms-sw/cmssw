#ifndef testChannel_H
#define testChannel_H

/**
 * \file testChannel.h
 * \class testChannel
 * \brief calculate the best DAC value to obtain a pedestal = 200
 * $Date: 2006/04/18 13:54:05 $
 * $Revision: 1.1 $
 * \author P. Govoni (testChannel.govoni@cernNOSPAM.ch)
 *
*/

#include <map>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/EDProduct.h" 
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedValues.h"
#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedResult.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

//#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DQMServices/UI/interface/MonitorUIRoot.h"

//#include "CalibCalorimetry/EcalDBInterface/interface/EcalCondDBInterface.h"
//#include "CalibCalorimetry/EcalDBInterface/interface/RunTag.h"
//#include "CalibCalorimetry/EcalDBInterface/interface/RunIOV.h"
//#include "CalibCalorimetry/EcalDBInterface/interface/MonRunIOV.h"

//#include "CalibCalorimetry/EcalDBInterface/interface/MonPedestalsDat.h"

//#include "CalibCalorimetry/EcalDBInterface/interface/MonPNPedDat.h"

#include "TROOT.h"
#include "TStyle.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms ;
using namespace edm ;

class testChannel: public edm::EDAnalyzer
{

  public:
    
    //! Constructor
    testChannel (const ParameterSet& ps) ;
    
    //! Destructor
    virtual ~testChannel () ;
    
    //! Subscribe/Unsubscribe to Monitoring Elements
    void subscribe (void) ;
    void subscribeNew (void) ;
    void unsubscribe (void) ;
    
    ///! Analyze
    void analyze (Event const& event, EventSetup const& eventSetup) ;
    
    //! BeginJob
    void beginJob (EventSetup const& eventSetup) ;
    
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
