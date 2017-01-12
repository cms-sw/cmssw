#ifndef ECALRAWDATARECOVERY_H
#define ECALRAWDATARECOVERY_H

//Emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <inttypes.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

//forward declaration
class FEDRawDataCollection;

/** EcalRawDataRecover class
 *
 * CMSSW module to fix identified ECAL data corruption. Current implementation
 * fixes the problem of run 274157 (see fixRun274157). The code can be extended
 * to deal with data issue that would be found in new runs.
 * Original author: Ph. Gras CEA/Saclay
 */
class EcalRawDataRecovery : public edm::EDProducer {
  //ctors
public:
  explicit EcalRawDataRecovery(const edm::ParameterSet&);
  ~EcalRawDataRecovery();

//methods
public:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  //Fixes data problem of Run 274157 described in
  //http://cmsonline.cern.ch/cms-elog/923609
  void fixRun274157();

  void beginJob();
  void endJob();
  void endRun();
  
//fields
private:

  /** Input collection token
   */
    edm::EDGetTokenT<FEDRawDataCollection>  ecalRawDataIn_;

  /** Output collection name
   */
  std::string ecalRawDataOut_;

  /** Pointer to the ECAL raw data in the
   * input colletion.
   */
  const FEDRawDataCollection* rawData_;
  
  /** Pointer to the ECAL raw data in the
   * output colletion.
   */
  std::auto_ptr<FEDRawDataCollection> newRawData_;
  
  /** Number of processed events;
   */
  int64_t iEvent_;

  /** Number of run 274157 evenets that required a fix
   */
  int64_t nFixedEvents_;

  /** Number of run 274157 evenets that did not require a fix
   */
  int64_t nOkEvents_;
};

#endif //ECALRAWDATARECOVERY_H not defined
