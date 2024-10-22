#ifndef EcalDCCTBUnpackingModule_H
#define EcalDCCTBUnpackingModule_H

/** \class EcalUnpackingModule
 * 
 *
 * \author N. Marinelli 
 * \author G. Della Ricca
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/stream/EDProducer.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <string>

class EcalTBDaqFormatter;
class EcalSupervisorTBDataFormatter;
class CamacTBDataFormatter;
class TableDataFormatter;
class MatacqTBDataFormatter;

class EcalDCCTBUnpackingModule : public edm::stream::EDProducer<> {
public:
  /// Constructor
  EcalDCCTBUnpackingModule(const edm::ParameterSet& pset);

  /// Destructor
  ~EcalDCCTBUnpackingModule() override;

  /// Produce digis out of raw data
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  EcalTBDaqFormatter* formatter_;
  EcalSupervisorTBDataFormatter* ecalSupervisorFormatter_;
  CamacTBDataFormatter* camacTBformatter_;
  TableDataFormatter* tableFormatter_;
  MatacqTBDataFormatter* matacqFormatter_;
  edm::InputTag fedRawDataCollectionTag_;
};

#endif
