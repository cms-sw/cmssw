#ifndef DTRawToDigi_DTControlData_h
#define DTRawToDigi_DTControlData_h

/** \class DTROS25Data
 *  The collection containing DT ROS25 status data.
 *
 *  $Date: 2009/11/15 11:42:29 $
 *  $Revision: 1.9 $
 *  \author M. Zanetti - INFN Padova
 *  \revision I. Josa - Ciemat Madrid
 */

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/src/fed_trailer.h>

#include <vector>

typedef std::pair<int, DTROBHeaderWord> DTROBHeader;   
typedef std::pair<int, DTTDCMeasurementWord> DTTDCData;
typedef std::pair<int, DTTDCErrorWord> DTTDCError;
typedef std::pair<DTLocalTriggerDataWord, int> DTSectorCollectorData;

class DTROS25Data {

public:

 /// Constructors
 DTROS25Data(int ROSId = 0): theROSId(ROSId) {}


 /// Destructor
 virtual ~DTROS25Data() {}

 /// Setters  ///////////////////////
 inline void setROSId(const int & ID) { theROSId = ID; }
  
 inline void addROSHeader( const DTROSHeaderWord & word)  { theROSHeader = DTROSHeaderWord(word) ; }
 inline void addROSTrailer( const DTROSTrailerWord & word)  { theROSTrailer = DTROSTrailerWord(word) ; }
 inline void addROSError( const DTROSErrorWord & word)  { theROSErrors.push_back(word); }
 inline void addROSDebug( const DTROSDebugWord & word)  { theROSDebugs.push_back(word); }
 inline void addROBHeader( const DTROBHeader & robHeader)  { theROBHeaders.push_back(robHeader); }   // IJ
 inline void addROBTrailer( const DTROBTrailerWord & word)  { theROBTrailers.push_back(word); }
 inline void addTDCMeasurement( const DTTDCMeasurementWord & word)  { theTDCMeasurements.push_back(word); }
 inline void addTDCData( const DTTDCData & tdcData)  { theTDCData.push_back(tdcData); }
 inline void addTDCError( const DTTDCError & tdcError)  { theTDCError.push_back(tdcError); }
 inline void addSCData ( const DTSectorCollectorData & scData) { theSCData.push_back(scData); }
 inline void addSCHeader( const DTLocalTriggerHeaderWord &scHeader) { theSCHeader = scHeader; }
 inline void addSCPrivHeader( const DTLocalTriggerSectorCollectorHeaderWord& scPrivHeader) { theSCPrivateHeader = scPrivHeader; }
  inline void addSCPrivSubHeader( const DTLocalTriggerSectorCollectorSubHeaderWord& scPrivSubHeader) { theSCPrivateSubHeader = scPrivSubHeader; }
 inline void addSCTrailer( const DTLocalTriggerTrailerWord& scTrailer) { theSCTrailer = scTrailer; }

 /// Getters ////////////////////////
 inline int getROSID() const { return theROSId; }

 inline const DTROSTrailerWord & getROSTrailer() const {return theROSTrailer;}
 inline const DTROSHeaderWord & getROSHeader() const {return theROSHeader;}
 inline const std::vector<DTROSErrorWord>& getROSErrors() const {return theROSErrors;}
 inline const std::vector<DTROSDebugWord>& getROSDebugs() const {return theROSDebugs;}
 inline const std::vector<DTROBHeader>& getROBHeaders() const {return theROBHeaders;}
 inline const std::vector<DTROBTrailerWord>& getROBTrailers() const {return theROBTrailers;}
 inline const std::vector<DTTDCMeasurementWord>& getTDCMeasurements() const {return theTDCMeasurements;}
 inline const std::vector<DTTDCData>& getTDCData() const {return theTDCData;}
 inline const std::vector<DTTDCError>& getTDCError() const {return theTDCError;}
 inline const std::vector<DTSectorCollectorData>& getSCData() const {return theSCData;}
 inline const DTLocalTriggerHeaderWord& getSCHeader() const {return theSCHeader;}
 inline const DTLocalTriggerSectorCollectorHeaderWord& getSCPrivHeader() const {return theSCPrivateHeader;}
 inline const DTLocalTriggerTrailerWord& getSCTrailer() const {return theSCTrailer;}
 inline const DTLocalTriggerSectorCollectorSubHeaderWord& getSCPrivSubHeader() const { return theSCPrivateSubHeader;}

 inline void clean() {
   theROSHeader = 0; 
   theROSTrailer = 0;
   theROSErrors.clear(); 
   theROSDebugs.clear(); 
   theROBHeaders.clear(); 
   theROBTrailers.clear(); 
   theTDCMeasurements.clear(); 
   theTDCData.clear(); 
   theTDCError.clear(); 
   theSCData.clear(); 
 }
 

private:

 int theROSId;

 DTROSHeaderWord theROSHeader;
 DTROSTrailerWord theROSTrailer;
 std::vector<DTROSErrorWord> theROSErrors;
 std::vector<DTROSDebugWord> theROSDebugs;
 std::vector<DTROBHeader> theROBHeaders;    
 std::vector<DTROBTrailerWord> theROBTrailers;
 std::vector<DTTDCMeasurementWord> theTDCMeasurements;
 std::vector<DTTDCData> theTDCData;
 std::vector<DTTDCError> theTDCError;
 std::vector<DTSectorCollectorData> theSCData;
 DTLocalTriggerHeaderWord theSCHeader;
 DTLocalTriggerSectorCollectorHeaderWord theSCPrivateHeader;
 DTLocalTriggerTrailerWord theSCTrailer;
 DTLocalTriggerSectorCollectorSubHeaderWord theSCPrivateSubHeader;



};


class DTDDUData {

public:

 /// Constructor
 DTDDUData(const FEDHeader & dduHeader, const FEDTrailer & dduTrailer):
   theDDUHeader(dduHeader),
   theDDUTrailer(dduTrailer),
   crcErrorBitSet(false)
 {}


 /// Destructor
 virtual ~DTDDUData() {}

 /// Setters
 inline void addDDUHeader( const FEDHeader & word)  { theDDUHeader = word; }
 inline void addDDUTrailer( const FEDTrailer & word)  { theDDUTrailer = word; }
 inline void addROSStatusWord( const DTDDUFirstStatusWord & word) {
   theROSStatusWords.push_back(word);
 }
 inline void addDDUStatusWord( const DTDDUSecondStatusWord & word) {
   theDDUStatusWord = word;
 }
 inline void checkCRCBit(const unsigned char* trailer) {
   const fedt_struct* theTrailer(reinterpret_cast<const fedt_t*>(trailer));
   if(((theTrailer->conscheck & 0x00000004) >> 2) == 1) {
     crcErrorBitSet = true;
   }
   crcErrorBitSet = false;
 }

 /// Getters
 inline const FEDHeader & getDDUHeader() const {return theDDUHeader;}
 inline const FEDTrailer & getDDUTrailer() const {return theDDUTrailer;}
 inline const std::vector<DTDDUFirstStatusWord> & getFirstStatusWord() const {
   return theROSStatusWords;}
 inline const DTDDUSecondStatusWord & getSecondStatusWord() const {
   return theDDUStatusWord;}
 inline bool crcErrorBit() const {
   return crcErrorBitSet;
 }
  
private:

 FEDHeader theDDUHeader;
 FEDTrailer theDDUTrailer;
 std::vector<DTDDUFirstStatusWord> theROSStatusWords;
 DTDDUSecondStatusWord theDDUStatusWord;
 bool crcErrorBitSet;

};


#endif
