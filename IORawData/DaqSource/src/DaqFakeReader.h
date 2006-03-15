#ifndef DaqSource_DaqFakeReader_h
#define DaqSource_DaqFakeReader_h

/** \class DaqFakeReader
 *  Generates empty FEDRawData of random size for all FEDs
 *  Proper headers and trailers are included; but the payloads are all 0s
 *
 *  $Date: 2005/10/06 18:23:47 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <DataFormats/Common/interface/EventID.h>
#include <algorithm>

class DaqFakeReader : public DaqBaseReader {
 public:
  /// Constructor
  DaqFakeReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DaqFakeReader();

  /// Generate and fill FED raw data for a full event
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

  edm::RunNumber_t runNum;
  edm::EventNumber_t eventNum;
  
};
#endif

