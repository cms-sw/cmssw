#ifndef DaqFakeReader_H
#define DaqFakeReader_H

/** \class DaqFakeReader
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <algorithm>

class DaqFakeReader : public DaqBaseReader {
 public:
  /// Constructor
  DaqFakeReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DaqFakeReader();

  // Generate raw data for a full event
  virtual bool fillRawData(edm::EventID& eID,
			   edm::Timestamp& tstamp, 
			   raw::FEDRawDataCollection& data);

 private:
  void fillFEDs(const std::pair<int,int>& fedRange,
		edm::EventID& eID,
		edm::Timestamp& tstamp, 
		raw::FEDRawDataCollection& data,
		float meansize,
		float width);

};
#endif

