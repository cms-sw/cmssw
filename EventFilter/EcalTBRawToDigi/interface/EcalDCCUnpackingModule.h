#ifndef EcalDCCTBUnpackingModule_H
#define EcalDCCTBUnpackingModule_H

/** \class EcalUnpackingModule
 * 
 *
 *  $Date: 2012/09/12 18:18:43 $
 *  $Revision: 1.13 $
 * \author N. Marinelli 
 * \author G. Della Ricca
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <string>


class EcalTBDaqFormatter;
class EcalSupervisorTBDataFormatter;
class CamacTBDataFormatter;
class TableDataFormatter;
class MatacqTBDataFormatter;

  class EcalDCCTBUnpackingModule: public edm::EDProducer {
  public:
    /// Constructor
    EcalDCCTBUnpackingModule(const edm::ParameterSet& pset);

    /// Destructor
    virtual ~EcalDCCTBUnpackingModule();
    
    /// Produce digis out of raw data
    void produce(edm::Event & e, const edm::EventSetup& c);

    // BeginJob
    void beginJob();

    // EndJob
    void endJob(void);

  private:

    EcalTBDaqFormatter* formatter_;
    EcalSupervisorTBDataFormatter* ecalSupervisorFormatter_;
    CamacTBDataFormatter* camacTBformatter_;
    TableDataFormatter* tableFormatter_;
    MatacqTBDataFormatter* matacqFormatter_;
    edm::InputTag fedRawDataCollectionTag_;
  };

#endif
