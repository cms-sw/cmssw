#ifndef EventFilter_DTDigiToRaw_h
#define EventFilter_DTDigiToRaw_h

/** \class DTDigiToRaw
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include <FWCore/Framework/interface/EDProducer.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

class FEDRawDataCollection;

class DTDigiToRaw {
public:
  /// Constructor
  DTDigiToRaw();

  /// Destructor
  virtual ~DTDigiToRaw();

  /// Take a vector of digis and fill the FEDRawDataCollection
  void createFedBuffers(const DTDigiCollection& digis, 
			FEDRawDataCollection& fed_buffers);

private:

};
#endif

