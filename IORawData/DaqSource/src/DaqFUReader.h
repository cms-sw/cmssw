#ifndef DaqFUReader_H
#define DaqFUReader_H

/** \class DaqFUReader
 *  No description available.
 *
 *  $Date: 2005/09/30 08:17:48 $
 *  $Revision: 1.1 $
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
			   FEDRawDataCollection& data);

private:

};
#endif

