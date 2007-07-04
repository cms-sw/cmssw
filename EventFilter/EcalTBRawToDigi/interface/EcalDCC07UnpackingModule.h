#ifndef EcalDCC07UnpackingModule_H
#define EcalDCC07UnpackingModule_H

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


class EcalTB07DaqFormatter;
class EcalSupervisorDataFormatter;
class CamacTBDataFormatter;
class TableDataFormatter;
class MatacqDataFormatter;

  class EcalDCC07UnpackingModule: public edm::EDProducer {
  public:
    /// Constructor
    EcalDCC07UnpackingModule(const edm::ParameterSet& pset);

    /// Destructor
    virtual ~EcalDCC07UnpackingModule();
    
    /// Produce digis out of raw data
    void produce(edm::Event & e, const edm::EventSetup& c);

    // BeginJob
    void beginJob(const edm::EventSetup& c);

    // EndJob
    void endJob(void);

  private:

    EcalTB07DaqFormatter* formatter_;
    EcalSupervisorDataFormatter* ecalSupervisorFormatter_;
    CamacTBDataFormatter* camacTBformatter_;
    TableDataFormatter* tableFormatter_;
    MatacqDataFormatter* matacqFormatter_;

  };

#endif
