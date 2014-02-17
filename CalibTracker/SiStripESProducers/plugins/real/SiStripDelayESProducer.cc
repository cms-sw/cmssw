// -*- C++ -*-
//
// Package:    SiStripDelayESProducer
// Class:      SiStripDelayESProducer
// 
/**\class SiStripDelayESProducer SiStripDelayESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripDelayESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  M. De Mattia
//         Created:  26/10/2010
// $Id: SiStripDelayESProducer.cc,v 1.1 2010/10/26 14:55:59 demattia Exp $
//
//



#include "CalibTracker/SiStripESProducers/plugins/real/SiStripDelayESProducer.h"



SiStripDelayESProducer::SiStripDelayESProducer(const edm::ParameterSet& iConfig):
  pset_(iConfig),
  toGet(iConfig.getParameter<Parameters>("ListOfRecordToMerge"))
{  
  setWhatProduced(this);
  
  edm::LogInfo("SiStripDelayESProducer") << "ctor" << std::endl;

  delay.reset(new SiStripDelay());
}


boost::shared_ptr<SiStripDelay> SiStripDelayESProducer::produce(const SiStripDelayRcd& iRecord)
{
  edm::LogInfo("SiStripDelayESProducer") << "produce called" << std::endl;

  delay->clear();

  edm::ESHandle<SiStripBaseDelay> baseDelay;

  std::string label;  
  std::string recordName;
  int sumSign = 0;

  for( Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet ) {
    recordName = itToGet->getParameter<std::string>("Record");
    label = itToGet->getParameter<std::string>("Label");
    sumSign = itToGet->getParameter<int>("SumSign");
    
    edm::LogInfo("SiStripDelayESProducer") << "[SiStripDelayESProducer::produce] Getting data from record " << recordName << " with label " << label << std::endl;

    if( recordName=="SiStripBaseDelayRcd" ) {
      iRecord.getRecord<SiStripBaseDelayRcd>().get(label, baseDelay);
      delay->fillNewDelay( *(baseDelay.product()), sumSign, std::make_pair(recordName, label) );
    } else {
      edm::LogError("SiStripDelayESProducer") << "[SiStripDelayESProducer::produce] Skipping the requested data for unexisting record " << recordName << " with tag " << label << std::endl;
      continue;
    }
  }

  delay->makeDelay();
  
  return delay;
}

