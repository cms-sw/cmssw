#ifndef EcalDCCTB07UnpackingModule_H
#define EcalDCCTB07UnpackingModule_H

/** \class EcalUnpackingModule
 * 
 *
 *  $Date: 2012/09/12 18:18:43 $
 *  $Revision: 1.6 $
 * \author Y. Maravin
 * \author G. Franzoni
 * \author G. Della Ricca
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <string>


class EcalTB07DaqFormatter;
class EcalSupervisorTBDataFormatter;
class CamacTBDataFormatter;
class TableDataFormatter;
class MatacqTBDataFormatter;

  class EcalDCCTB07UnpackingModule: public edm::EDProducer {
  public:
    /// Constructor
    EcalDCCTB07UnpackingModule(const edm::ParameterSet& pset);

    /// Destructor
    virtual ~EcalDCCTB07UnpackingModule();
    
    /// Produce digis out of raw data
    void produce(edm::Event & e, const edm::EventSetup& c);

    // BeginJob
    void beginJob();

    // EndJob
    void endJob(void);

  private:

    EcalTB07DaqFormatter* formatter_;
    EcalSupervisorTBDataFormatter* ecalSupervisorFormatter_;
    CamacTBDataFormatter* camacTBformatter_;
    TableDataFormatter* tableFormatter_;
    MatacqTBDataFormatter* matacqFormatter_;

    bool ProduceEEDigis_;
    bool ProduceEBDigis_;
    edm::InputTag fedRawDataCollectionTag_;
  };

#endif
