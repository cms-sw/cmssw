#ifndef DaqSource_DTROS8FileReader_h
#define DaqSource_DTROS8FileReader_h

/** \class DTROS8FileReader
 *  Read DT ROS8 raw data files
 *
 *  $Date: 2005/11/21 18:35:41 $
 *  $Revision: 1.1 $
 *  \author M. Zanetti - INFN Padova
 */

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <FWCore/EDProduct/interface/EventID.h>

#include <fstream>

class DTROS8FileReader : public DaqBaseReader {
 public:
  /// Constructor
  DTROS8FileReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTROS8FileReader();

  /// Generate and fill FED raw data for a full event
  virtual bool fillRawData(edm::EventID& eID,
			   edm::Timestamp& tstamp, 
			   FEDRawDataCollection& data);

  virtual bool checkEndOfFile();

 private:

  std::ifstream inputFile;

  edm::RunNumber_t runNum;
  edm::EventNumber_t eventNum;

  static const int ros8WordLenght = 4;

};
#endif


template<class T> char* dataPointer( const T* ptr ) {
  union bPtr {
    const T* dataP;
    char*    fileP;
  };
  union bPtr buf;
  buf.dataP = ptr;
  return buf.fileP;
}


template<class T> T* typePointer( const char* ptr ) {
  union bPtr {
    T*          dataP;
    const char* fileP;
  };
  union bPtr buf;
  buf.fileP = ptr;
  return buf.dataP;
}
