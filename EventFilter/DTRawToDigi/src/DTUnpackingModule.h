#ifndef DTRawToDigi_DTUnpackingModule_h
#define DTRawToDigi_DTUnpackingModule_h

/** \class DTUnpackingModule
 *  The unpacking module for DTs.
 *
 *  $Date: 2005/11/21 17:38:48 $
 *  $Revision: 1.3 $
 * \author N. Amapane - S. Argiro' - M. Zanetti
 */

#include <FWCore/Framework/interface/EDProducer.h>

#include <iostream>

class DTUnpacker;

class DTUnpackingModule: public edm::EDProducer {
 public:
  /// Constructor
  DTUnpackingModule(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTUnpackingModule();
    
  /// Call the Unpackers and create the digis 
  void produce(edm::Event & e, const edm::EventSetup& c);


 private:

  DTUnpacker * unpacker;

};

#endif
