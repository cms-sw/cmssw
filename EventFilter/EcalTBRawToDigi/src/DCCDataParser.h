/*----------------------------------------------------------*/
/* DCC DATA PARSER                                          */
/*                                                          */
/* Author : N.Almeida (LIP)         Date   : 30/05/2004     */
/*----------------------------------------------------------*/

#ifndef DCCTBDATAPARSER_HH
#define DCCTBDATAPARSER_HH

#include <fstream>                   //STL
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <stdio.h>                     //C

#include "ECALParserException.h"      //DATA DECODER
#include "DCCEventBlock.h"
#include "DCCDataMapper.h"


class DCCTBDataMapper;
class DCCTBEventBlock;


class DCCTBDataParser{

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
  DCCTBDataParser( const std::vector<uint32_t>& parserParameters , bool parseInternalData = true, bool debug = true);
  
  /**
    Parse data from file 
  */
  void parseFile( std::string fileName, bool singleEvent = false);
	
  /**
     Parse data from a buffer
  */
  void parseBuffer( uint32_t * buffer, uint32_t bufferSize, bool singleEvent = false);

  /**
     Get method for DCCTBDataMapper
  */
  DCCTBDataMapper *mapper();
		
  /**
     Check if EVENT LENGTH is coeherent and if BOE/EOE are correctly written
     returns 3 bits code with the error found + event length
  */
  std::pair<uint32_t,uint32_t> checkEventLength(uint32_t * pointerToEvent, uint32_t bytesToEnd, bool singleEvent = false);
  
  /**
     Get methods for parser parameters;
  */
  std::vector<uint32_t> parserParameters(); 
  uint32_t numbXtalSamples();
  uint32_t numbTriggerSamples();
  uint32_t numbTTs();
  uint32_t numbSRF();
  uint32_t dccId();
  uint32_t srpId();
  uint32_t tcc1Id();
  uint32_t tcc2Id();
  uint32_t tcc3Id();
  uint32_t tcc4Id();
  

  /**
     Set method for parser parameters
  */
  void  setParameters( const std::vector<uint32_t>& newParameters );


  /**
     Get methods for block sizes
  */
  uint32_t srpBlockSize();
  uint32_t tccBlockSize();

  /**
     Get methods for debug flag
  */
  bool  debug();

  /**
     Get method for DCCEventBlocks vector
   */
  std::vector<DCCTBEventBlock *> & dccEvents();

  /**
     Get method for error counters map
  */
  std::map<std::string,uint32_t> & errorCounters();

  /**
   * Get method for events
   */
  std::vector< std::pair< uint32_t, std::pair<uint32_t *, uint32_t> > > events();


  /**
     Reset Error Counters
  */
  void resetErrorCounters();
		

  /**
     Methods to get data strings formatted as decimal/hexadecimal, indexes and indexed data
  */
  std::string getDecString(uint32_t data);		
  std::string getHexString(uint32_t data);
  std::string index(uint32_t position);
  std::string getIndexedData( uint32_t indexed, uint32_t * pointer);

  /**
   * Retrieves a pointer to the data buffer
   */
  uint32_t *getBuffer() { return buffer_;}
  
  /**
     Class destructor
  */
  ~DCCTBDataParser();

  enum DCCDataParserGlobalFields{
    EMPTYEVENTSIZE = 32                   //bytes
  };
 
protected :
  void computeBlockSizes();

  uint32_t *buffer_;                //data buffer
  uint32_t bufferSize_;             //buffer size

  uint32_t srpBlockSize_;           //SR block size
  uint32_t tccBlockSize_;           //TCC block size

  uint32_t processedEvent_;
  std::string eventErrors_;
  DCCTBDataMapper *mapper_;
  
  std::vector<DCCTBEventBlock *> dccEvents_;
  
  // std::pair< errorMask, std::pair< pointer to event, event size (number of DW)> >
  std::vector< std::pair< uint32_t, std::pair<uint32_t *, uint32_t> > > events_;
  
  bool parseInternalData_;          //parse internal data flag
  bool debug_;                      //debug flag
  std::map<std::string,uint32_t> errors_;        //errors map
  std::vector<uint32_t> parameters;         //parameters vector

  enum DCCTBDataParserFields{
    EVENTLENGTHMASK = 0xFFFFFF,
    
    BOEBEGIN = 28,                  //begin of event (on 32 bit string starts at bit 28)
    BOEMASK = 0xF,                  //mask is 4 bits (F)
    BOE =0x5,                       //B'0101'

    EOEBEGIN = 28,                  //end of event
    EOEMASK = 0xF,                  //4 bits
    EOE =0xA                        //B'1010'
  };
		
};

inline DCCTBDataMapper *DCCTBDataParser::mapper() { return mapper_;}

inline std::vector<uint32_t> DCCTBDataParser::parserParameters() { return parameters; }
inline uint32_t DCCTBDataParser::numbXtalSamples()     { return parameters[0]; }
inline uint32_t DCCTBDataParser::numbTriggerSamples()  { return parameters[1]; }
inline uint32_t DCCTBDataParser::numbTTs()             { return parameters[2]; }
inline uint32_t DCCTBDataParser::numbSRF()             { return parameters[3]; }
inline uint32_t DCCTBDataParser::dccId()               { return parameters[4]; }
inline uint32_t DCCTBDataParser::srpId()               { return parameters[5]; }
inline uint32_t DCCTBDataParser::tcc1Id()              { return parameters[6]; } 
inline uint32_t DCCTBDataParser::tcc2Id()              { return parameters[7]; } 
inline uint32_t DCCTBDataParser::tcc3Id()              { return parameters[8]; } 
inline uint32_t DCCTBDataParser::tcc4Id()              { return parameters[9]; }

inline void  DCCTBDataParser::setParameters( const std::vector<uint32_t>& newParameters ){ parameters = newParameters; computeBlockSizes();}

inline uint32_t DCCTBDataParser::srpBlockSize()        { return srpBlockSize_; } 
inline uint32_t DCCTBDataParser::tccBlockSize()        { return tccBlockSize_; } 

inline bool DCCTBDataParser::debug()                          { return debug_;     }
inline std::vector<DCCTBEventBlock *> &DCCTBDataParser::dccEvents()  { return dccEvents_;    }
inline std::map<std::string,uint32_t> &DCCTBDataParser::errorCounters()    { return errors_;       }
inline std::vector< std::pair< uint32_t, std::pair<uint32_t *, uint32_t> > > DCCTBDataParser::events() { return events_;   }


#endif

