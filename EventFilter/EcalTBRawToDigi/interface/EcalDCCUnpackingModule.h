#ifndef EcalDCCUnpackingModule_H
#define EcalDCCUnpackingModule_H

/** \class EcalUnpackingModule
 * 
 *
 *  $Date: $
 *  $Revision: $
 * \author N. Marinelli 
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>

#include <iostream>

class EcalTBDaqFormatter;


  class EcalDCCUnpackingModule: public edm::EDProducer {
  public:
    /// Constructor
    EcalDCCUnpackingModule(const edm::ParameterSet& pset);

    /// Destructor
    virtual ~EcalDCCUnpackingModule();
    
    /// Produce digis out of raw data
    void produce(edm::Event & e, const edm::EventSetup& c);

  private:
    EcalTBDaqFormatter* formatter;
  };

DEFINE_FWK_MODULE(EcalDCCUnpackingModule);

#endif
