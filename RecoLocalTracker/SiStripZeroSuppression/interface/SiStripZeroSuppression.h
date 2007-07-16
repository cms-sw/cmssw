#ifndef SiStripZeroSuppression_h
#define SiStripZeroSuppression_h

/** \class SiStripZeroSuppression
 *
 * SiStripZeroSuppression is the EDProducer subclass which makes a Digis collection
 * (SiStripDigi/interface/StripDigi.h) strarting from a RawDigis collection
 * (SiStripDigi/interface/StripRawDigi.h)
 *
 * \author Domenico Giordano
 *
 * \version   1st Version Jan 20 2006

 *
 ************************************************************/

//edm
#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//FIXME: the first 2 include are needed??
//#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"
//Data Formats
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
//ES Data
//#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h" 
//#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
//ZeroSuppression 
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripZeroSuppressionAlgorithm.h"
//SiStripPedestalsService
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

#include <iostream> 
#include <memory>
#include <string>

namespace cms
{
  class SiStripZeroSuppression : public edm::EDProducer
  {

    typedef std::vector<edm::ParameterSet> Parameters;

  public:

    explicit SiStripZeroSuppression(const edm::ParameterSet& conf);

    virtual ~SiStripZeroSuppression();

    virtual void produce(edm::Event& , const edm::EventSetup& );

  private:
    edm::ParameterSet conf_;
    SiStripZeroSuppressionAlgorithm SiStripZeroSuppressionAlgorithm_;
    Parameters                      RawDigiProducersList;
  };
}
#endif
