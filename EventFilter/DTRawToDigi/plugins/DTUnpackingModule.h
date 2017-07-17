#ifndef DTRawToDigi_DTUnpackingModule_h
#define DTRawToDigi_DTUnpackingModule_h

/** \class DTUnpackingModule
 *  The unpacking module for DTs.
 *
 * \author N. Amapane - S. Argiro' - M. Zanetti
 */

#include <FWCore/Framework/interface/stream/EDProducer.h>
#include "FWCore/Utilities/interface/InputTag.h"
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <iostream>

class DTUnpacker;

class DTUnpackingModule: public edm::stream::EDProducer<> {
 public:
  /// Constructor
  DTUnpackingModule(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTUnpackingModule();
    
  /// Call the Unpackers and create the digis 
  void produce(edm::Event & e, const edm::EventSetup& c) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  DTUnpacker * unpacker;

  /// if not you need the label
  edm::EDGetTokenT<FEDRawDataCollection> inputLabel;
  /// do you want to use the standard DT FED ID's, i.e. [770-775]? (why the hell 6??)
  bool useStandardFEDid_;
  /// if not you need to set the range by hand
  int minFEDid_;
  int maxFEDid_;
  bool dqmOnly;
  bool performDataIntegrityMonitor;
  std::string dataType;
};

#endif
