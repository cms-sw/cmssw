#ifndef DTRawToDigi_DTUnpackingModule_h
#define DTRawToDigi_DTUnpackingModule_h

/** \class DTUnpackingModule
 *  The unpacking module for DTs.
 *
 *  $Date: 2005/07/14 08:00:50 $
 *  $Revision: 1.2 $
 * \author N. Amapane - S. Argiro'
 */

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

#endif
