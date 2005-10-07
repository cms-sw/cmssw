#ifndef EcalDCCUnpackingModule_H
#define EcalDCCUnpackingModule_H

/** \class EcalUnpackingModule
 * 
 *
 *  $Date: 2005/08/03 15:28:39 $
 *  $Revision: 1.1 $
 * \author N. Marinelli 
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>

#include "TROOT.h"
#include "TFile.h"

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

    TFile* rootFile;
  };

DEFINE_FWK_MODULE(EcalDCCUnpackingModule);

#endif
