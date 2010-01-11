#ifndef DaqSource_DaqFakeReader_h
#define DaqSource_DaqFakeReader_h

/** \class DaqFakeReader
 *  Generates empty FEDRawData of random size for all FEDs
 *  Proper headers and trailers are included; but the payloads are all 0s
 *
 *  $Date: 2007/04/19 13:28:46 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <DataFormats/Provenance/interface/EventID.h>
#include <algorithm>


class DaqFakeReader : public DaqBaseReader
{
 public:
  //
  // construction/destruction
  //
  DaqFakeReader(const edm::ParameterSet& pset);
  virtual ~DaqFakeReader();
  

  //
  // public member functions
  //

  // Generate and fill FED raw data for a full event
  virtual int fillRawData(edm::EventID& eID,
			  edm::Timestamp& tstamp, 
			  FEDRawDataCollection*& data);
  
private:
  //
  // private member functions
  //
  void fillFEDs(const std::pair<int,int>& fedRange,
		edm::EventID& eID,
		edm::Timestamp& tstamp, 
		FEDRawDataCollection& data,
		float meansize,
		float width);
  
private:
  //
  // member data
  //
  edm::RunNumber_t   runNum;
  edm::EventNumber_t eventNum;
  bool               empty_events;

};

#endif
