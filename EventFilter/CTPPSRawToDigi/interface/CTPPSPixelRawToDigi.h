#ifndef CTPPSPixelRawToDigi_H
#define CTPPSPixelRawToDigi_H

/** \class CTPPSPixelRawToDigi_H
 *  Plug-in module that performs Raw data to digi conversion 
 *  for CTPPS pixel detector
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"



class CTPPSPixelRawToDigi : public edm::stream::EDProducer<> {
public:

  explicit CTPPSPixelRawToDigi( const edm::ParameterSet& );

  virtual ~CTPPSPixelRawToDigi();

  /// get data, convert to digis attach againe to Event
  virtual void produce( edm::Event&, const edm::EventSetup& ) override;

private:

  edm::ParameterSet config_;

  edm::EDGetTokenT<FEDRawDataCollection> tFEDRawDataCollection; 

  std::set<unsigned int> fedIds;


  edm::InputTag label;
 
  std::string mappingLabel;
};
#endif
