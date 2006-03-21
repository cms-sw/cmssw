#ifndef DTRawToDigi_DTROS25Data_h
#define DTRawToDigi_DTROS25Data_h

/** \class DTROS25Data
 *  The collection containing DT ROS25 status data.
 *
 *  $Date: 2006/02/21 19:14:53 $
 *  $Revision: 1.1 $
 *  \author M. Zanetti - INFN Padova
 */

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>

#include <vector>

using namespace std;

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

  /// Getters
  inline int getROSID() const { return theROSId; }

  inline vector<DTROSTrailerWord> getROSTrailers() const {return theROSTrailers;}
  inline vector<DTROSErrorWord> getDTROSErrors() const {return theROSErrors;}
  inline vector<DTROSDebugWord> getDTROSDebugs() const {return theROSDebugs;}
  inline vector<DTROBTrailerWord> getDTROBTrailers() const {return theROBTrailers;}
  inline vector<DTTDCMeasurementWord> getDTTDCMeasurements() const {return theTDCMeasurements;}

private:

  int theROSId;

  vector<DTROSTrailerWord> theROSTrailers;
  vector<DTROSErrorWord> theROSErrors;
  vector<DTROSDebugWord> theROSDebugs;
  vector<DTROBTrailerWord> theROBTrailers;
  vector<DTTDCMeasurementWord> theTDCMeasurements;
 
};

#endif
