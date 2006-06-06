// Date   : 30/05/2004
// Author : N.Almeida (LIP)

#ifndef DCCDATAPARSER_HH
#define DCCDATAPARSER_HH

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <map>

using namespace edm;
using namespace std;

class DCCDataMapper;
class DCCEventBlock;

class DCCDataParser{

	public : 

	  DCCDataParser( bool parseInternalData = true, bool debug = true);	
		// parameters[0] is the xtal samples (default is 10)
		// parameters[1] is the number of trigger time samples (default is 1)
		// parameters[2] is the number of TT (default is 68)
		// parameters[3] is the number of SR Flags (default is 68)
		// parameters[4] is the dcc id
		// parameters[5] is the sr id
		// parameters[6] is the tcc1 id
		// parameters[7] is the tcc2 id
		// parameters[8] is the tcc3 id
		// parameters[9] is the tcc4 id

		DCCDataParser( vector<ulong> parserParameters , bool parseInternalData = true, bool debug = true);
		~DCCDataParser();
	
		void parseBuffer( ulong * buffer, ulong bufferSize, bool singleEvent = false);
		void parseFile( string fileName, bool singleEvent = false);

		DCCDataMapper * mapper(){ return mapper_;}
		
		// returns error mask and error counter /////////////////////////////////////
		pair<ulong,ulong> checkEventLength(ulong * pointerToEvent, ulong bytesToEnd, bool singleEvent = false);
		
		
		vector<ulong> parserParameters(){return parameters;}
		
		void          setParameters( vector<ulong> newParameters ){ parameters = newParameters; computeBlockSizes();}
		
		ulong numbXtalSamples()                                { return parameters[0]; }
		ulong numbTriggerSamples()                             { return parameters[1]; }
		ulong numbTTs()                                        { return parameters[2]; }
		ulong numbSRF()                                        { return parameters[3]; }
		ulong dccId()                                          { return parameters[4]; }
		ulong srpId()                                          { return parameters[5]; }
		ulong tcc1Id()                                         { return parameters[6]; } 
		ulong tcc2Id()                                         { return parameters[7]; } 
		ulong tcc3Id()                                         { return parameters[8]; } 
		ulong tcc4Id()                                         { return parameters[9]; }
		ulong srpBlockSize()                                   { return srpBlockSize_; } 
		ulong tccBlockSize()                                   { return tccBlockSize_; } 
		
		bool  debug()                                          { return debug_;        }
		vector<DCCEventBlock *> & dccEvents()                  { return dccEvents_;    }
		map<string,ulong> & errorCounters()                    { return errors_;       }
		vector< pair< ulong, pair<ulong *, ulong> > > events() { return events_;       }
		void resetErrorCounters();
		
		string getDecString(ulong data);		
		string getHexString(ulong data);
		string index(ulong position);
		string getIndexedData( ulong indexed, ulong * pointer);
		
		enum DCCDataParserGlobalFields{
			
			EMPTYEVENTSIZE = 32 //bytes
		};

	protected :
		void computeBlockSizes();
		ulong * buffer_;
		ulong bufferSize_;
		ulong srpBlockSize_;
		ulong tccBlockSize_;
		ulong processedEvent_;
		string eventErrors_;
		DCCDataMapper *mapper_;
	
		vector<DCCEventBlock *> dccEvents_;
		
		// pair< errorMask, pair< pointer to event, event size (number of DW)> >
		vector< pair< ulong, pair<ulong *, ulong> > > events_;
		
		bool parseInternalData_;
		bool debug_;
		map<string,ulong> errors_;
		vector<ulong> parameters;

		enum DCCDataParserFields{
			EVENTLENGTHMASK = 0xFFFFFF,
			BOEBEGIN = 28,
			EOEBEGIN = 28,
			EOEMASK = 0xF,
			BOEMASK = 0xF,
			BOE =0x5, EOE =0xA 
		};

};

#endif

