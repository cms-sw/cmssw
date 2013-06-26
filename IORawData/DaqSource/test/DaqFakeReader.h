#ifndef DaqSource_DaqFakeReader_h
#define DaqSource_DaqFakeReader_h

/** \class DaqFakeReader
 *  Generates empty FEDRawData of random size for all FEDs
 *  Proper headers and trailers are included; but the payloads are all 0s
 *
 *  $Date: 2010/03/12 14:24:24 $
 *  $Revision: 1.3 $
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
  void fillFEDs(const int, const int,
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
