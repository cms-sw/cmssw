/*  
 *
 *  \author G. Franzoni
 *
 */

#include "CamacTBDataFormatter.h"

using namespace edm;
using namespace std;

#include <iostream>


struct hodo_fibre_index 
{
  int nfiber;
  int ndet;
};

// nHodoscopes = 2; nFibres = 64 
const static struct hodo_fibre_index hodoFiberMap[2][64] = {
  { // Hodo 0
    // unit 1A
    {23,44}, {29,47}, {31,48}, {21,43},
    { 5,35}, {15,40}, { 7,36}, {13,39},
    { 1,33}, {11,38}, { 3,34}, { 9,37},
    { 6, 3}, {16, 8}, { 8, 4}, {14, 7},
    // unit 1C
    {17,41}, {19,42}, {27,46}, {25,45},
    {32,16}, {22,11}, {24,12}, {30,15},
    {12, 6}, { 2, 1}, { 4, 2}, {10, 5},
    {28,14}, {18, 9}, {20,10}, {26,13},
    // unit 2A
    {54,27}, {56,28}, {64,32}, {62,31},
    {49,57}, {59,62}, {51,58}, {57,61},
    {53,59}, {63,64}, {55,60}, {61,63},
    {45,55}, {39,52}, {37,51}, {47,56},
    // unit 2C
    {34,17}, {42,21}, {44,22}, {36,18},
    {50,25}, {60,30}, {58,29}, {52,26},
    {38,19}, {40,20}, {48,24}, {46,23},
    {41,53}, {35,50}, {33,49}, {43,54}
  },
  { // Hodo 1
    // unit 1A
    {31,48}, {29,47}, {23,44}, {21,43},
    { 5,35}, { 7,36}, {15,40}, {13,39},
    { 1,33}, { 3,34}, {11,38}, { 9,37},
    { 6, 3}, { 8, 4}, {16, 8}, {14, 7},
    // unit 1C
    {17,41}, {27,46}, {19,42}, {25,45},
    {24,12}, {22,11}, {32,16}, {30,15},
    { 4, 2}, { 2, 1}, {12, 6}, {10, 5},
    {20,10}, {18, 9}, {28,14}, {26,13},
    // unit 2A
    {54,27}, {64,32}, {56,28}, {62,31},
    {49,57}, {51,58}, {59,62}, {57,61},
    {53,59}, {55,60}, {63,64}, {61,63},
    {45,55}, {47,56}, {37,51}, {39,52},
    // unit 2C
    {34,17}, {42,21}, {36,18}, {44,22},
    {50,25}, {52,26}, {58,29}, {60,30},
    {38,19}, {48,24}, {40,20}, {46,23},
    {41,53}, {43,54}, {33,49}, {35,50}
  }
};



CamacTBDataFormatter::CamacTBDataFormatter () {
  nWordsPerEvent = 114;
}


// for tests based on standalone file
// void CamacTBDataFormatter::interpretRawData(ulong * buffer, ulong bufferSize,
// 					    EcalTBEventHeader& tbEventHeader, 
// 					    EcalTBHodoscopeRawInfo & productHodo,
// 					    EcalTBTDCRawInfo & productTdc)

void CamacTBDataFormatter::interpretRawData( const FEDRawData & fedData, 
					     EcalTBEventHeader& tbEventHeader,
					     EcalTBHodoscopeRawInfo& hodoRaw,
					     EcalTBTDCRawInfo& tdcRawInfo )
{
  

  // to do: introduce here checks on fed size!
  //  const ulong * buffer 
  const unsigned char*  buffer= fedData.data();
  int fedLenght                        = fedData.size();
  
  // check ultimate fed size and strip off fed-header and -trailer
  if (fedLenght != 114)
    {
      LogError("CamacTBDataFormatter") << "CamacTBData has size "  <<  fedLenght
				       <<" as opposed to expected 114. Returning."<< endl;
      return;
    }

  
  ulong a=1;
  ulong b=1;
  for (int wordNumber=0; wordNumber<nWordsPerEvent; wordNumber++)
    { statusWords[wordNumber -1] = true;}

  for (int wordNumber=0; wordNumber<nWordsPerEvent; wordNumber++)
    { checkStatus( buffer[wordNumber],  wordNumber);}

  for (int wordNumber=0; wordNumber<nWordsPerEvent; wordNumber++)
    {
      if (! statusWords[wordNumber])
	{
	  LogError("CamacTBDataFormatter") << "bad status in some of the event words; returning;" << endl;	  
	}
    }
  






  // read first word
  a = buffer[0];
  LogDebug("CamacTBDataFormatter") << "\n\nword:\t" << a << endl;
  
  b = (a& 0xff000000);
  b = b >> 24;
  LogDebug("CamacTBDataFormatter") << "format  ver:\t" << b << endl;

  b = (a& 0xff0000);
  b = b >> 16;
  LogDebug("CamacTBDataFormatter") << "major:\t" << b << endl;

  b = (a& 0xff00);
  b = b >> 8;
  LogDebug("CamacTBDataFormatter") << "minor:\t" << b << endl;

  a = buffer[1];
  LogDebug("CamacTBDataFormatter") << "\n\n word:\t" << a << endl;
  LogDebug("CamacTBDataFormatter") << "time stamp secs: "<<a << endl;

  a = buffer[2];
  LogDebug("CamacTBDataFormatter") << "\n\n word:\t" << a << endl;
  LogDebug("CamacTBDataFormatter") << "time stamp musecs: " <<a << endl;


  a = buffer[3];
  LogDebug("CamacTBDataFormatter") << "\n\n word:\t" << a << endl;
  b = (a& 0xffffff);
  LogDebug("CamacTBDataFormatter") << "LV1A: "<< b << endl;

  a = buffer[4];
  LogDebug("CamacTBDataFormatter") << "\n\n word:\t" << a << endl;
  b = (a& 0xffff0000);
  b = b >> 16;
  LogDebug("CamacTBDataFormatter") << "run number: "<< b << endl;
  b = (a& 0xffff);
  LogDebug("CamacTBDataFormatter") << "spill number: "<< b << endl;

  a = buffer[5];
  LogDebug("CamacTBDataFormatter") << "\n\n word:\t" << a << endl;
  b = (a& 0xffff0000);
  b = b >> 16;
  LogDebug("CamacTBDataFormatter") << "vme errors: "<< b << endl;
  b = (a& 0xffff);
  LogDebug("CamacTBDataFormatter") << "camac errors: "<< b << endl;


  ulong bufferHodo[16];
  for (int hodo=0; hodo<16; hodo++)
    {
      a = buffer[6+hodo];
      bufferHodo[hodo]  = buffer[6+hodo];
      b =a;
      LogDebug("CamacTBDataFormatter") << "hodo: " << hodo << "\t: " << b << endl;
    }


  for (int iplane=0; iplane<nHodoPlanes; iplane++) 
    {         
      int detType = 1;       // new mapping for electronics channels  
               
      for (int fiber=0; fiber<nHodoFibers; fiber++) { hodoHits[iplane][fiber] = 0; }            
               
      int ch=0;
      
      // loop on [4-24bits words] = 1 plane 
      for(int j=0; j<hodoRawLen; j++) 
	{
	  int word=  bufferHodo[  j+iplane*hodoRawLen  ]  &0xffff;
	  for(int i=1; i<0x10000; i<<=1) 
	    {
	      if ( word & i ) 
		{
		  // map electronics channel to No of fibre
		  hodoHits[iplane][ hodoFiberMap[detType][ch].nfiber - 1]++;
		}
	      ch ++;
	    }
	} 
    }

  
  // building the hodo infos (returning decoded hodoscope hits information)
  hodoRaw.setPlanes((unsigned int)nHodoPlanes);
  for (int ipl = 0; ipl < nHodoPlanes; ipl++) 
    {             
      EcalTBHodoscopePlaneRawHits theHodoPlane;
      theHodoPlane.setChannels((unsigned int)nHodoFibers);
      for (int fib = 0; fib < nHodoFibers; fib++){ theHodoPlane.setHit((unsigned int)fib, (bool)hodoHits[ipl][fib]); }
      hodoRaw.setPlane((unsigned int)ipl, theHodoPlane);
    }
  
  
  for (int scaler=0; scaler<72; scaler++)
    {
      a = buffer[22+scaler];
      b =a;
      LogDebug("CamacTBDataFormatter") << "scaler: " << scaler << "\t: " << b << endl;
    }
      
  LogDebug("CamacTBDataFormatter") <<"\n";
  for (int finger=0; finger<2; finger++)
    {
      a = buffer[94+finger];
      b =a;
      LogDebug("CamacTBDataFormatter") << "finger: " << finger << "\t: " << b << endl;
    }
  

  a = buffer[97];
  LogDebug("CamacTBDataFormatter") << "\n\n word:\t" << a << endl;
  b = (a& 0xff);
  b = a;
  LogDebug("CamacTBDataFormatter") << "number TDC words: "<< b << endl;
  
  int numberTDCwords = b;
  numberTDCwords = 16;
  for (int tdc=0; tdc< numberTDCwords ; tdc++)
    {
      a = buffer[98+tdc];
      b =a;
      LogDebug("CamacTBDataFormatter") << "tdc: " << tdc << "\t: " << b << endl;
    }

  a = buffer[114];
  LogDebug("CamacTBDataFormatter") << "\n\n word:\t" << a << endl;
  b = a;
  LogDebug("CamacTBDataFormatter") << "last word of event: "<< b << endl;
  //    }

}









void CamacTBDataFormatter::checkStatus(ulong word, int wordNumber){
  
  if ( wordNumber > nWordsPerEvent)
    { 
      LogError("CamacTBDataFormatter::checkStatus") << "checking word number: "
						    <<  wordNumber << " which is out of allowed range (" 
						    << nWordsPerEvent << ")" << endl;
    }


  if  (word & 0x80000000) // daq item not used
    { 
      LogError("CamacTBDataFormatter::checkStatus") << "daq item not used at word: "<<  wordNumber << endl;
      statusWords[wordNumber -1] = false;      
    }
  
  if (word & 0x40000000) // vme error on data
    { 
      LogError("CamacTBDataFormatter::checkStatus") << "vme error on word: "<<  wordNumber << endl;
      statusWords[wordNumber -1] = false;      
    }
    
  if (word & 0x20000000) // vme error on status
    { 
      LogError("CamacTBDataFormatter::checkStatus") << "vme status error at word: "<<  wordNumber << endl;
      statusWords[wordNumber -1] = false;      
    }
    
  if (word & 0x10000000) // camac error (no X)
    { 
      LogError("CamacTBDataFormatter::checkStatus") << "camac error (no X) at word: "<<  wordNumber << endl;
      statusWords[wordNumber -1] = false;      
    }
    
  if (word & 0x08000000) // camac error (no Q)
    { 
      LogError("CamacTBDataFormatter::checkStatus") << "camac error (no Q) at word: "<<  wordNumber << endl;
      statusWords[wordNumber -1] = false;      
    }
 
  if (word & 0x04000000) // no camac check error
    { 
      LogError("CamacTBDataFormatter::checkStatus") << "no camac check error at word: "<<  wordNumber << endl;
      statusWords[wordNumber -1] = false;      
    }

}
