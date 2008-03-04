#ifndef EcalDCCUnpackingModule_H
#define EcalDCCUnpackingModule_H

/** \class EcalUnpackingModule
 * 
 *
 *  $Date: 2007/04/12 08:36:46 $
 *  $Revision: 1.9 $
 * \author N. Marinelli 
 * \author G. Della Ricca
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>

#include <iostream>
#include <string>


class EcalTBDaqFormatter;
class EcalSupervisorDataFormatter;
class CamacTBDataFormatter;
class TableDataFormatter;
class MatacqDataFormatter;

  class EcalDCCUnpackingModule: public edm::EDProducer {
  public:
    /// Constructor
    EcalDCCUnpackingModule(const edm::ParameterSet& pset);

    /// Destructor
    virtual ~EcalDCCUnpackingModule();
    
    /// Produce digis out of raw data
    void produce(edm::Event & e, const edm::EventSetup& c);

    // BeginJob
    void beginJob(const edm::EventSetup& c);

    // EndJob
    void endJob(void);

  private:

    EcalTBDaqFormatter* formatter_;
    EcalSupervisorDataFormatter* ecalSupervisorFormatter_;
    CamacTBDataFormatter* camacTBformatter_;
    TableDataFormatter* tableFormatter_;
    MatacqDataFormatter* matacqFormatter_;

  };

#endif
