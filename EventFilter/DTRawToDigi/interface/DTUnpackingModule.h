#ifndef DTUnpackingModule_H
#define DTUnpackingModule_H

/** \class DTUnpackingModule
 *  No description available.
 *
 *  $Date: 2005/07/13 09:06:50 $
 *  $Revision: 1.1 $
 * \author N. Amapane - S. Argiro'
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>

#include <iostream>

class DTDaqCMSFormatter;


  class DTUnpackingModule: public edm::EDProducer {
  public:
    /// Constructor
    DTUnpackingModule(const edm::ParameterSet& pset);

    /// Destructor
    virtual ~DTUnpackingModule();
    
    /// Produce digis out of raw data
    void produce(edm::Event & e, const edm::EventSetup& c);

  private:
    DTDaqCMSFormatter* formatter;
  };

DEFINE_FWK_MODULE(DTUnpackingModule)

#endif
