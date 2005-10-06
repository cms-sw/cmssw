#ifndef DaqSource_DaqFUReader_h
#define DaqSource_DaqFUReader_h

/** \class DaqFUReader
 *  Gets raw data from the DAQ and puts it to the event in the FU
 *
 *  $Date: 2005/10/04 18:38:48 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include <IORawData/DaqSource/interface/DaqBaseReader.h>

class DaqFUReader : public DaqBaseReader {

public:
  /// Constructor
  DaqFUReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DaqFUReader();

  /// Read in a full event and fill the raw data containers
  virtual bool fillRawData(edm::EventID& eID,
			   edm::Timestamp& tstamp, 
			   FEDRawDataCollection& data);

private:

};
#endif

