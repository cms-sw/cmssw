#ifndef DTRawToDigi_DTROS25Data_h
#define DTRawToDigi_DTROS25Data_h

/** \class DTROS25Data
 *  The collection containing DT ROS25 status data.
 *
 *  $Date: 2006/03/21 19:55:35 $
 *  $Revision: 1.2 $
 *  \author M. Zanetti - INFN Padova
 */

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>

#include <vector>


using namespace std;

typedef pair<int, DTTDCMeasurementWord> DTTDCData;

class DTROS25Data {

public:
  
  /// Constructors
  DTROS25Data(int ROSId = 0): theROSId(ROSId) {}

  
  /// Destructor
  virtual ~DTROS25Data() {}

  /// Setters
  inline void setROSId(const int & ID) { theROSId = ID; }

  inline void addROSTrailer( const DTROSTrailerWord & word)  { theROSTrailers.push_back(word); }
  inline void addROSError( const DTROSErrorWord & word)  { theROSErrors.push_back(word); }
  inline void addROSDebug( const DTROSDebugWord & word)  { theROSDebugs.push_back(word); }
  inline void addROBTrailer( const DTROBTrailerWord & word)  { theROBTrailers.push_back(word); }
  inline void addTDCMeasurement( const DTTDCMeasurementWord & word)  { theTDCMeasurements.push_back(word); }
  inline void addTDCData( const DTTDCData & tdcData)  { theTDCData.push_back(tdcData); }

  /// Getters
  inline int getROSID() const { return theROSId; }

  inline const vector<DTROSTrailerWord>& getROSTrailers() const {return theROSTrailers;}
  inline const vector<DTROSErrorWord>& getROSErrors() const {return theROSErrors;}
  inline const vector<DTROSDebugWord>& getROSDebugs() const {return theROSDebugs;}
  inline const vector<DTROBTrailerWord>& getROBTrailers() const {return theROBTrailers;}
  inline const vector<DTTDCMeasurementWord>& getTDCMeasurements() const {return theTDCMeasurements;}
  inline const vector<DTTDCData>& getTDCData() const {return theTDCData;}

private:

  int theROSId;

  vector<DTROSTrailerWord> theROSTrailers;
  vector<DTROSErrorWord> theROSErrors;
  vector<DTROSDebugWord> theROSDebugs;
  vector<DTROBTrailerWord> theROBTrailers;
  vector<DTTDCMeasurementWord> theTDCMeasurements;
  vector<DTTDCData> theTDCData;

};

#endif
