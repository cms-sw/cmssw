#ifndef ReadPixelRecHit_h
#define ReadPixelRecHit_h

/** \class ReadPixelRecHit
 *
 * ReadPixelRecHit is the EDProducer subclass which finds seeds
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Aug. 01, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class ReadPixelRecHit : public edm::EDAnalyzer
{
 public:
  
  explicit ReadPixelRecHit(const edm::ParameterSet& conf);
  
  virtual ~ReadPixelRecHit();
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  
 private:
  edm::ParameterSet conf_;
  
};


#endif
