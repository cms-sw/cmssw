/*----------------------------------------------------------*/
/* DCC DATA PARSER                                          */
/*                                                          */
/* Author : N.Almeida (LIP)         Date   : 30/05/2004     */
/*----------------------------------------------------------*/

#ifndef DCCDATAPARSER_HH
#define DCCDATAPARSER_HH

#include <fstream>                   //STL
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <stdio.h>                     //C

#include "ECALParserException.h"      //DATA DECODER
#include "DCCEventBlock.h"
#include "DCCDataMapper.h"

using namespace std;

class DCCDataMapper;
class DCCEventBlock;


class DCCDataParser{

public : 
  
  /**
     Class constructor: takes a vector of 10 parameters and flags for parseInternalData and debug
     Parameters are: 
     0 - crystal samples (default is 10)
     1 - number of trigger time samples (default is 1)
     2 - number of TT (default is 68)
     3 - number of SR Flags (default is 68)
     4 - DCC id
     5 - SR id
     [6-9] - TCC[6-9] id
  */
  DCCDataParser( vector<ulong> parserParameters , bool parseInternalData = true, bool debug = true);
  
  /**
    Parse data from file 
  */
  void parseFile( string fileName, bool singleEvent = false);
	
  /**
     Parse data from a buffer
  */
  void parseBuffer( ulong * buffer, ulong bufferSize, bool singleEvent = false);

  /**
     Get method for DCCDataMapper
  */
  DCCDataMapper *mapper();
		
  /**
     Check if EVENT LENGTH is coeherent and if BOE/EOE are correctly written
     returns 3 bits code with the error found + event length
  */
  pair<ulong,ulong> checkEventLength(ulong * pointerToEvent, ulong bytesToEnd, bool singleEvent = false);
  
  /**
     Get methods for parser parameters;
  */
  vector<ulong> parserParameters(); 
  ulong numbXtalSamples();
  ulong numbTriggerSamples();
  ulong numbTTs();
  ulong numbSRF();
  ulong dccId();
  ulong srpId();
  ulong tcc1Id();
  ulong tcc2Id();
  ulong tcc3Id();
  ulong tcc4Id();
  

  /**
     Set method for parser parameters
  */
  void  setParameters( vector<ulong> newParameters );


  /**
     Get methods for block sizes
  */
  ulong srpBlockSize();
  ulong tccBlockSize();

  /**
     Get methods for debug flag
  */
  bool  debug();

  /**
     Get method for DCCEventBlocks vector
   */
  vector<DCCEventBlock *> & dccEvents();

  /**
     Get method for error counters map
  */
  map<string,ulong> & errorCounters();

  /**
   * Get method for events
   */
  vector< pair< ulong, pair<ulong *, ulong> > > events();


  /**
     Reset Error Counters
  */
  void resetErrorCounters();
		

  /**
     Methods to get data strings formatted as decimal/hexadecimal, indexes and indexed data
  */
  string getDecString(ulong data);		
  string getHexString(ulong data);
  string index(ulong position);
  string getIndexedData( ulong indexed, ulong * pointer);

  /**
   * Retrieves a pointer to the data buffer
   */
  ulong *getBuffer() { return buffer_;}
  
  /**
     Class destructor
  */
  ~DCCDataParser();

  enum DCCDataParserGlobalFields{
    EMPTYEVENTSIZE = 32                   //bytes
  };
 
protected :
  void computeBlockSizes();

  ulong *buffer_;                //data buffer
  ulong bufferSize_;             //buffer size

  ulong srpBlockSize_;           //SR block size
  ulong tccBlockSize_;           //TCC block size

  ulong processedEvent_;
  string eventErrors_;
  DCCDataMapper *mapper_;
  
  vector<DCCEventBlock *> dccEvents_;
  
  // pair< errorMask, pair< pointer to event, event size (number of DW)> >
  vector< pair< ulong, pair<ulong *, ulong> > > events_;
  
  bool parseInternalData_;          //parse internal data flag
  bool debug_;                      //debug flag
  map<string,ulong> errors_;        //errors map
  vector<ulong> parameters;         //parameters vector

  enum DCCDataParserFields{
    EVENTLENGTHMASK = 0xFFFFFF,
    
    BOEBEGIN = 28,                  //begin of event (on 32 bit string starts at bit 28)
    BOEMASK = 0xF,                  //mask is 4 bits (F)
    BOE =0x5,                       //B'0101'

    EOEBEGIN = 28,                  //end of event
    EOEMASK = 0xF,                  //4 bits
    EOE =0xA                        //B'1010'
  };
		
};

inline DCCDataMapper *DCCDataParser::mapper() { return mapper_;}

inline vector<ulong> DCCDataParser::parserParameters() { return parameters; }
inline ulong DCCDataParser::numbXtalSamples()     { return parameters[0]; }
inline ulong DCCDataParser::numbTriggerSamples()  { return parameters[1]; }
inline ulong DCCDataParser::numbTTs()             { return parameters[2]; }
inline ulong DCCDataParser::numbSRF()             { return parameters[3]; }
inline ulong DCCDataParser::dccId()               { return parameters[4]; }
inline ulong DCCDataParser::srpId()               { return parameters[5]; }
inline ulong DCCDataParser::tcc1Id()              { return parameters[6]; } 
inline ulong DCCDataParser::tcc2Id()              { return parameters[7]; } 
inline ulong DCCDataParser::tcc3Id()              { return parameters[8]; } 
inline ulong DCCDataParser::tcc4Id()              { return parameters[9]; }

inline void  DCCDataParser::setParameters( vector<ulong> newParameters ){ parameters = newParameters; computeBlockSizes();}

inline ulong DCCDataParser::srpBlockSize()        { return srpBlockSize_; } 
inline ulong DCCDataParser::tccBlockSize()        { return tccBlockSize_; } 

inline bool DCCDataParser::debug()                          { return debug_;     }
inline vector<DCCEventBlock *> &DCCDataParser::dccEvents()  { return dccEvents_;    }
inline map<string,ulong> &DCCDataParser::errorCounters()    { return errors_;       }
inline vector< pair< ulong, pair<ulong *, ulong> > > DCCDataParser::events() { return events_;   }


#endif

