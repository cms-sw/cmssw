/** 
 * \class EcalRawToDigi
 *
 * This class takes care of unpacking ECAL's raw data info
 *
 * \author Pedro Silva (adapted from HcalRawToDigi and ECALTBRawToDigi)
 *
 * \version 1.0
 * \date June 08,2006  
 *
 */

#ifndef _ECALRAWTODIGI_H_ 
#define _ECALRAWTODIGI_H_ 1

#include <iostream>                                  //C++

#include "EventFilter/EcalRawToDigi/interface/DCCMapper.h"

#include <EventFilter/EcalTBRawToDigi/interface/EcalDCCUnpackingModule.h>
#include <EventFilter/EcalTBRawToDigi/src/EcalTBDaqFormatter.h>
#include <EventFilter/EcalTBRawToDigi/src/ECALParserException.h>
#include <EventFilter/EcalTBRawToDigi/src/ECALParserBlockException.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"


class DCCMapper;

using namespace std;
using namespace edm;

class EcalRawToDigi : public edm::EDProducer
{
public:
  /**
   * Class constructor
   */
  explicit EcalRawToDigi(const edm::ParameterSet& ps);

  /**
   * Functions that are called by framework at each event
   */
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

  /**
   * Class destructor
   */
  virtual ~EcalRawToDigi();

private:
  //list of FEDs to unpack
  std::vector<int> fedUnpackList_;

  //a dcc id mapper class 
  DCCMapper *myMap;
};

#endif
