#ifndef DaqFakeReader_H
#define DaqFakeReader_H

/** \class DaqFakeReader
 *  No description available.
 *
 *  $Date: 2005/09/30 08:17:48 $
 *  $Revision: 1.1 $
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
			   FEDRawDataCollection& data);

 private:
  void fillFEDs(const std::pair<int,int>& fedRange,
		edm::EventID& eID,
		edm::Timestamp& tstamp, 
		FEDRawDataCollection& data,
		float meansize,
		float width);

};
#endif

