#ifndef FUSHMREADER_H
#define FUSHMREADER_H 1


#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "IORawData/DaqSource/interface/DaqBaseReader.h"
#include "DataFormats/Provenance/interface/EventID.h"


class FUShmReader : public DaqBaseReader
{
public:
  //
  //construction/destruction
  //
  FUShmReader();
  virtual ~FUShmReader();
  
  
  //
  // memeber functions
  //

  // DaqBaseReader interface
  virtual
  void setRunNumber(edm::RunNumber_t runNumber) { runNumber_=runNumber; }
  
  int fillRawData(edm::EventID& eID,
		   edm::Timestamp& tstamp, 
		   FEDRawDataCollection*& data);
  
  
private:
  //
  // member data
  //
  FEDRawDataCollection* event_;
  evf::FUShmBuffer*     shmBuffer_;
  
  unsigned int          runNumber_;
  unsigned int          evtNumber_;
  unsigned int          lastCellIndex_;
  bool                  lumi_;
  
};


#endif
