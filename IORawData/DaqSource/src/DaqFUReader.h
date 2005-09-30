#ifndef DaqFUReader_H
#define DaqFUReader_H

/** \class DaqFUReader
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include <IORawData/DaqSource/interface/DaqBaseReader.h>

class DaqFUReader : public DaqBaseReader {

public:
  /// Constructor
  DaqFUReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DaqFUReader();

  // Read in a full event
  virtual bool fillRawData(edm::EventID& eID,
			   edm::Timestamp& tstamp, 
			   raw::FEDRawDataCollection& data);

private:

};
#endif

