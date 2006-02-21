#ifndef DTRawToDigi_DTROS25Data_h
#define DTRawToDigi_DTROS25Data_h

/** \class DTROS25Data
 *  The collection containing DT ROS25 status data.
 *
 *  $Date: 2006/02/07 23:27:28 $
 *  $Revision: 1.3 $
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
  
  /// Getters
  inline int getROSID() const { return theROSId; }

  inline vector<DTROSTrailerWord> getROSTrailers() const {return theROSTrailers;}
  inline vector<DTROSErrorWord> getDTROSErrors() const {return theROSErrors;}
  inline vector<DTROSDebugWord> getDTROSDebugs() const {return theROSDebugs;}


private:

  int theROSId;

  vector<DTROSTrailerWord> theROSTrailers;
  vector<DTROSErrorWord> theROSErrors;
  vector<DTROSDebugWord> theROSDebugs;

};

#endif
