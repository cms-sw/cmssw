// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLHTRPatterns
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:20 CDT 2007
// $Id: XMLHTRPatterns.cc,v 1.3 2007/12/06 02:26:35 kukartse Exp $
//

// system include files
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>

XERCES_CPP_NAMESPACE_USE 

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLHTRPatterns.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

//N = Number
#define NHBHECR 9        //HB HE crates
#define NHFCR 3          //HF crates
#define NHOCR 4          //HO crates
#define NFBR 8           //Fibers for htrs
#define NFCH 3           //3 Fiber channels ranging 0-2
#define NTOPBOT 2        
#define NHTRS 3          //Htrs 0, 1, 2 for HB HE
#define NHSETS 4         //4 sets of HB/HE htrs
#define NHTRSHO 4        //Htrs 0-3 for HO
#define NHSETSHO 3       //3 sets of H0 htrs
#define NRMFIBR 6        //6 rm fibers ranging 2-7
#define NRMSLOT 4        //4 rm slots ranging 1-4
#define NHOETA 15
#define NHOPHI 72  



//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
XMLHTRPatterns::HTRPatternConfig::_HTRPatternConfig()
{
  CFGBrickSet = "example";
  crate = 0;
  slot = 13;
  topbottom = 0;
  fiber = 1;
  generalizedindex = slot*100 + topbottom*10 + fiber;
  creationtag = "example";

  char timebuf[50];
  //time_t _time = time( NULL );
  time_t _time = 1193697120;
  //strftime( timebuf, 50, "%c", gmtime( &_time ) );
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S.0", gmtime( &_time ) );
  creationstamp = timebuf;

  pattern_spec_name = "example";
}

XMLHTRPatterns::XMLHTRPatterns() : XMLDOMBlock( "HCAL_HTR_DATA_PATTERNS.crate.template" )
{
}

XMLHTRPatterns::XMLHTRPatterns( string templateBase ) : XMLDOMBlock( templateBase )
{
}

// XMLHTRPatterns::XMLHTRPatterns(const XMLHTRPatterns& rhs)
// {
//    // do actual copying here;
// }

XMLHTRPatterns::~XMLHTRPatterns()
{
}

//
// assignment operators
//
// const XMLHTRPatterns& XMLHTRPatterns::operator=(const XMLHTRPatterns& rhs)
// {
//   //An exception safe implementation is
//   XMLHTRPatterns temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
int XMLHTRPatterns::addPattern( HTRPatternConfig * config, string templateFileName )
{
  DOMElement * root = document -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();

  DOMNodeList * _list = dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "Parameter" ) );
  for ( int i = 0; i < _list -> getLength(); i++ )
    {
      DOMElement * _tag = (DOMElement *)( _list -> item( i ));
      string parameterName = XMLString::transcode( _tag -> getAttribute( XMLProcessor::_toXMLCh( "name" ) ) );
      //cout << XMLString::transcode( _tag -> getAttribute( XMLProcessor::_toXMLCh( "name" ) ) ) << endl;
      if ( parameterName == "CRATE" )             setTagValue( "Parameter", config -> crate, i, dataSet );
      if ( parameterName == "SLOT" )              setTagValue( "Parameter", config -> slot, i, dataSet );
      if ( parameterName == "TOPBOTTOM" )         setTagValue( "Parameter", config -> topbottom, i, dataSet );
      if ( parameterName == "FIBER" )             setTagValue( "Parameter", config -> fiber, i, dataSet );
      if ( parameterName == "GENERALIZEDINDEX" )  setTagValue( "Parameter", config -> generalizedindex, i, dataSet );
      if ( parameterName == "CREATIONTAG" )       setTagValue( "Parameter", config -> creationtag, i, dataSet );
      if ( parameterName == "CREATIONSTAMP" )     setTagValue( "Parameter", config -> creationstamp, i, dataSet );
      if ( parameterName == "PATTERN_SPEC_NAME" ) setTagValue( "Parameter", config -> pattern_spec_name, i, dataSet );
    }

  setTagValue( "Data", data_elements, 0, dataSet );

  // changes to the pattern
  //setTagValue( "CRATE", config -> crate, 0, dataSet );
  
  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = document -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}

int XMLHTRPatterns::fillPatternInfo( const HTRPatternConfig & _conf )
{
  HTRPatternID id_;
  id_ . crate = _conf . crate;
  id_ . slot  = _conf . slot;
  id_ . topbottom    = _conf . topbottom;
  id_ . fiber = _conf . fiber;

  configByCrate[ id_ ] = _conf . crate;

  return 0;
}

int XMLHTRPatterns::createByCrate( void )
{

  for ( int i = 0; i < 1024; i++ ) data_elements . append( " 10803" );

//Global iterator variables
int i, j;

//Stream variable

stringstream mystream;

//Declare the auxiliary functions (maybe put them in a library later?)
void printHBHEHF();
void printHO();

void writeXMLFooter (std::ostream& fOutput);
void writeXMLHeader (std::ostream& fOutput, int swi);
void writeXML       (std::ostream& fOutput);
void writeXMLHO       (std::ostream& fOutput);


//Variables that need to be printed
int irm,     irm_fi,  iwedge,  ipixel,  iqie,    iadc,  isector;
int islb,    irctcra, irctcar, irctcon, irctnam, idphi;
int iside,   ieta,    iphi,    idepth,  icrate,  ihtr;
int ihtr_fi, ifi_ch,  ispigot, idcc,    idcc_sl, ifed;

string rbx, slbin, slbin2, slnam;
string det,  fpga, rctnam, letter;
char tempbuff[30];

//These are used in the printing functions.
int titlecounter = 0;

FILE* HOmap;
FILE* HBEFmap; 
  
  //Note: If you have no "Maps" directory, modify this or you might get a segmentation fault
  HOmap   = fopen ("../Maps/HCALmapHO_9.14.2007.txt","w");
  HBEFmap = fopen ("../Maps/HCALmapHBEF_9.14.2007.txt","w");
  
  /* HBHE crate numbering */
  int hbhecrate[NHBHECR]={0,1,4,5,10,11,14,15,17};
  /* HF crate numbering */
  int hfcrate[NHFCR]={2,9,12};
  /* HO crate numbering */
  int hocrate[NHOCR]={3,7,6,13};
  /* HBHE FED numbering of DCCs */
  int fedhbhenum[NHBHECR][2]={{702,703},{704,705},{700,701},
			      {706,707},{716,717},{708,709},
			      {714,715},{710,711},{712,713}};
  /* HF FED numbering of DCCs */
  int fedhfnum[NHFCR][2]={{718,719},{720,721},{722,723}};
  /* HO FED numbering of DCCs */
  int fedhonum[NHOCR][2]={{724,725},{726,727},{728,729},{730,731}};
  /* HBHE/HF htr slot offsets for set of three htrs */
  int ihslot[NHSETS]={2,5,13,16};
  /* HO htr slot offsets for three sets of four htrs */
  int ihslotho[NHSETSHO][NHTRSHO]={{2,3,4,5},{6,7,13,14},{15,16,17,18}};
  /* iphi (lower) starting index for each HBHE crate */
  int ihbhephis[NHBHECR]={11,19,3,27,67,35,59,43,51};
  /* iphi (lower) starting index for each HF crate */
  int ihfphis[NHFCR]={3,27,51};
  /* iphi (lower) starting index for each HO crate */
  int ihophis[NHOCR]={71,17,35,53};
  /* ihbheetadepth - unique HBHE {eta,depth} assignments per fiber and fiber channel */
  int ihbheetadepth[NHTRS][NTOPBOT][NFBR][NFCH][2]={
                                    {{{{11,1},{ 7,1},{ 3,1}},  /* htr 0 (HB) -bot(+top) */
                                      {{ 5,1},{ 1,1},{ 9,1}},
                                      {{11,1},{ 7,1},{ 3,1}},
                                      {{ 5,1},{ 1,1},{ 9,1}},
                                      {{10,1},{ 6,1},{ 2,1}},
                                      {{ 8,1},{ 4,1},{12,1}},
                                      {{10,1},{ 6,1},{ 2,1}},
                                      {{ 8,1},{ 4,1},{12,1}}},
                                     {{{11,1},{ 7,1},{ 3,1}},  /* htr 0 (HB) +bot(-top) */
                                      {{ 5,1},{ 1,1},{ 9,1}},
                                      {{11,1},{ 7,1},{ 3,1}},
                                      {{ 5,1},{ 1,1},{ 9,1}},
                                      {{10,1},{ 6,1},{ 2,1}},
                                      {{ 8,1},{ 4,1},{12,1}},
                                      {{10,1},{ 6,1},{ 2,1}},
                                      {{ 8,1},{ 4,1},{12,1}}}},
                                    {{{{16,2},{15,2},{14,1}},  /* htr 1 (HBHE) -bot(+top) */
                                      {{15,1},{13,1},{16,1}},
                                      {{16,2},{15,2},{14,1}},
                                      {{15,1},{13,1},{16,1}},
                                      {{17,1},{16,3},{26,1}},
                                      {{18,1},{18,2},{26,2}},
                                      {{17,1},{16,3},{25,1}},
                                      {{18,1},{18,2},{25,2}}},
                                     {{{16,2},{15,2},{14,1}},  /* htr 1 (HBHE) +bot(-top) */
                                      {{15,1},{13,1},{16,1}},
                                      {{16,2},{15,2},{14,1}},
                                      {{15,1},{13,1},{16,1}},
                                      {{17,1},{16,3},{25,1}},
                                      {{18,1},{18,2},{25,2}},
                                      {{17,1},{16,3},{26,1}},
                                      {{18,1},{18,2},{26,2}}}},
                                    {{{{28,1},{28,2},{29,1}},  /* htr 2 (HE) -bot(+top) */
                                      {{28,3},{24,2},{24,1}},
                                      {{27,1},{27,2},{29,2}},
                                      {{27,3},{23,2},{23,1}},
                                      {{19,2},{20,1},{22,2}},
                                      {{19,1},{20,2},{22,1}},
                                      {{19,2},{20,1},{21,2}},
                                      {{19,1},{20,2},{21,1}}},
                                     {{{27,1},{27,2},{29,2}},  /* htr 2 (HE) +bot(-top) */
                                      {{27,3},{23,2},{23,1}},
                                      {{28,1},{28,2},{29,1}},
                                      {{28,3},{24,2},{24,1}},
                                      {{19,2},{20,1},{21,2}},
                                      {{19,1},{20,2},{21,1}},
                                      {{19,2},{20,1},{22,2}},
                                      {{19,1},{20,2},{22,1}}}}
                                    };
  /* ihfetadepth - unique HF {eta,depth} assignments per fiber and fiber channel */
  int ihfetadepth[NTOPBOT][NFBR][NFCH][2]={
                                     {{{33,1},{31,1},{29,1}},  /* top */
                                      {{32,1},{30,1},{34,1}},
                                      {{33,2},{31,2},{29,2}},
                                      {{32,2},{30,2},{34,2}},
                                      {{34,2},{32,2},{30,2}},
                                      {{31,2},{29,2},{33,2}},
                                      {{34,1},{32,1},{30,1}},
                                      {{31,1},{29,1},{33,1}}},
                                     {{{41,1},{37,1},{35,1}},  /* bot */
                                      {{38,1},{36,1},{39,1}},
                                      {{41,2},{37,2},{35,2}},
                                      {{38,2},{36,2},{39,2}},
                                      {{40,2},{38,2},{36,2}},
                                      {{37,2},{35,2},{39,2}},
                                      {{40,1},{38,1},{36,1}},
                                      {{37,1},{35,1},{39,1}}}
                                    };

  //Aram's insert: I shall now define an array which contains the RM and the RM fiber for HB HE
  //and variables associated with this table
  int irm_rmfiHBHE[NHTRS][NTOPBOT][NFBR][2]={
                                      {{{6,1},{7,1},{6,2},{7,2},{4,1},{5,1},{4,2},{5,2}},  // HTR 0 top
				       {{6,3},{7,3},{6,4},{7,4},{4,3},{5,3},{4,4},{5,4}}}, // HTR 0 bot
				      {{{2,1},{3,1},{2,2},{3,2},{2,1},{3,1},{2,2},{3,2}},  // HTR 1 top
				       {{2,3},{3,3},{2,4},{3,4},{2,3},{3,3},{2,4},{3,4}}}, // HTR 1 bot
				      {{{4,1},{5,1},{4,2},{5,2},{6,1},{7,1},{6,2},{7,2}},  // HTR 2 top
				       {{4,3},{5,3},{4,4},{5,4},{6,3},{7,3},{6,4},{7,4}}}  // HTR 2 bot
                                    };

  int irm_rmfiHF[NHTRS][NTOPBOT][NFBR][2]={
                                      {{{1,2},{2,2},{3,2},{4,2},{1,3},{2,3},{3,3},{4,3}},  // HTR 0 top
				       {{5,2},{6,2},{7,2},{8,2},{5,3},{6,3},{7,3},{8,3}}}, // HTR 0 bot
				      {{{1,1},{2,1},{3,1},{4,1},{1,2},{2,2},{3,2},{4,2}},  // HTR 1 top
				       {{5,1},{6,1},{7,1},{8,1},{5,2},{6,2},{7,2},{8,2}}}, // HTR 1 bot
				      {{{1,3},{2,3},{3,3},{4,3},{1,1},{2,1},{3,1},{4,1}},  // HTR 2 top
				       {{5,3},{6,3},{7,3},{8,3},{5,1},{6,1},{7,1},{8,1}}}  // HTR 2 bot
                                    };

  //Pixel tables as a function of rm, rm fiber and fiber channel

  int ipixelHB[NRMFIBR][NFCH][NRMSLOT] = {  //  fch = 0           fch = 1           fch = 2
                                          {{18, 17, 3,  2 }, {13, 3,  17, 7 }, {14, 1,  19, 6 }}, //rmfiber = 2
					  {{19, 2,  18, 1 }, {15, 7,  13, 5 }, {17, 19, 1,  3 }}, //rmfiber = 3
					  {{9,  4,  16, 11}, {5,  8,  12, 15}, {2,  13, 7,  18}}, //rmfiber = 4
					  {{12, 11, 9,  8 }, {7,  15, 5,  13}, {16, 6,  14, 4 }}, //rmfiber = 5
					  {{8,  5,  15, 12}, {4,  9,  11, 16}, {1,  14, 6,  19}}, //rmfiber = 6
					  {{6,  16, 4,  14}, {3,  18, 2,  17}, {11, 12, 8,  9 }}  //rmfiber = 7
                                         };
                                     
  int ipixelHE[NRMFIBR][NFCH][NRMSLOT] = {  //  fch = 0           fch = 1           fch = 2
                                          {{12, 12, 12, 12}, {16, 7,  16, 7 }, {7,  16, 7,  16}}, //rmfiber = 2
					  {{11, 11, 11, 11}, {19, 3,  19, 3 }, {3,  19, 3,  19}}, //rmfiber = 3
					  {{15, 15, 6,  6 }, {2,  18, 2,  18}, {6,  6,  15, 15}}, //rmfiber = 4
					  {{5,  14, 5,  14}, {14, 5,  14, 5 }, {18, 2,  18, 2 }}, //rmfiber = 5
					  {{17, 1,  17, 1 }, {9,  9,  9,  9 }, {1,  17, 1,  17}}, //rmfiber = 6
					  {{13, 4,  13, 4 }, {8,  8,  8,  8 }, {4,  13, 4,  13}}  //rmfiber = 7
                                         };
 
  //adc and qie table; qie is entry 0, adc is entry 1. Constant across HB, HE, HO
  int iadcquiHBHE[NRMFIBR][NFCH][2];
 
  for (i = 0; i < NRMFIBR; i++){
    for (j = 0; j < NFCH; j++){
      //Intentionally relying on integer truncation here
      iadcquiHBHE[i][j][0] = i / 2 + 1;
      
      if (i % 2 == 0) iadcquiHBHE[i][j][1] = j;
      else            iadcquiHBHE[i][j][1] = NFCH + (j + 1) % 3;
    }
  }
  
  //slb and rct tables

  //HB and HE
 
  const char* S_slbin_odd[] ={"A1","B0","B1","A0","A1","B0","B1","A0"};
  const char* S_slbin_even[]={"C1","D0","D1","C0","C1","D0","D1","C0"};
  const char* rct_rackHBHE[]={"S2E01-RH","S2E03-RH","S2E05-RH","S2E07-RH","S2E09-RH","S2E08-RL","S2E06-RL","S2E04-RL","S2E02-RL",
			     "S2E02-RH","S2E04-RH","S2E06-RH","S2E08-RH","S2E09-RL","S2E07-RL","S2E05-RL","S2E03-RL","S2E01-RL"};
 
  //HF
  const char* S_slbin_7[] ={"A0","A1","B0","B1"};
  const char* S_slbin_3[] ={"C0","C1","D0","D1"};
  const char* rct_rackHF[]={"S2E01-FH","S2E03-FH","S2E05-FH","S2E07-FH","S2E09-FH","S2E08-FL","S2E06-FL","S2E04-FL","S2E02-FL",
			   "S2E02-FH","S2E04-FH","S2E06-FH","S2E08-FH","S2E09-FL","S2E07-FL","S2E05-FL","S2E03-FL","S2E01-FL"};
 

  int slb_table[29] = {1,1,2,2,3,3,4,4,5,5,6,6,       // 1<=eta<=12
		       1,1,2,2,3,3,1,1,               // 13<=eta<=20
		       2,2,3,3,4,4,4,4,4};            // 21<=eta<=29

  //RM for the HO as a function of eta, phi and side as implemented in complete_ho_map.txt
  //There are only 24 phi columns because after that it begins to repeat. The relevant variable is phi mod 24.
  int HO_RM_table[24][15][2] = 
    {
      {{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{2,2},{2,2},{2,2},{2,2},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{3,3},{3,3},{3,3},{3,3},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{3,3},{3,3},{3,3},{3,3},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{4,4},{4,4},{4,4},{4,4},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{4,4},{4,4},{4,4},{4,4},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{3,3},{3,3},{3,3},{3,3},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{2,2},{2,2},{2,2},{2,2},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{2,2},{2,2},{2,2},{2,2},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{4,4},{4,4},{4,4},{4,4},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{4,4},{4,4},{4,4},{4,4},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{3,3},{3,3},{3,3},{3,3},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{3,3},{3,3},{3,3},{3,3},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{2,2},{2,2},{2,2},{2,2},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{2,2},{2,2},{2,2},{2,2},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{1,1},{1,1},{1,1},{1,1},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{1,1},{1,1},{1,1},{1,1},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{2,2},{2,2},{2,2},{2,2},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{2,2},{2,2},{2,2},{2,2},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{3,3},{3,3},{3,3},{3,3},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{3,3},{3,3},{3,3},{3,3},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{1,1},{1,1},{1,1},{1,1},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{1,1},{1,1},{1,1},{1,1},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2}}
    };
  
  //For |eta| 5 to 15, rm_fi is a function of |eta| only while htr_fi is a function of side and |eta|
  int HO_RM_fi_eta5to15[11] = {3, 2, 5, 4, 7, 6, 3, 2, 5, 4, 7};

  int HO_htr_fi_eta5to15[2][11] = {{5, 1, 2, 3, 4, 5, 6, 7, 8, 6, 7},   //iside = -1
				   {1, 8, 7, 6, 5, 4, 3, 2, 1, 2, 1}};  //iside = +1
  
  //For |eta| 1 to 4, it is a function of phi, eta and side. eta 1-3 always have the same value given a side, eta 4 is separate
  //and thus gets its own box
  int HO_RM_fi_eta1to4[72][2][2] = 
    {           //side = -1            side = 1
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 1  to 8
      {{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 9  to 16
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}}, //Phi 17 to 24
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 25 to 32
      {{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 33 to 40
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}}, //Phi 41 to 48
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 49 to 56
      {{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 57 to 64
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}}  //Phi 65 to 72
    };

  //Pixel and letter code for the HO. Ring 0 is separate and goes into entry 0, Rings +/- 1,2 are all the same and go to entry 1.
  //                   Fiber Channel     0        1      2        0       1       2           0        1        2
  int ipixelHO[NRMFIBR][NFCH][2] = { {{12,12},  {7,7}, {6,3}}, {{4,4},  {8,8},  {5,1}  }, {{19,11},{-1000,6},{17,2 }},   //RM fibers 2,3,4
				     {{-1000,9},{1,13},{3,5}}, {{11,19},{16,18},{15,17}}, {{13,15},{9,14},   {14,16}} }; //RM fibers 5,6,7
  //                   Fiber Channel        0         1         2            0         1         2            0         1         2
  string letterHO[NRMFIBR][NFCH][2] = {{{"E","E"},{"G","L"},{"F","S"}}, {{"Q","M"},{"N","T"},{"P","F"}}, {{"A","C"},{"Z","J"},{"J","Q"}},
				       {{"Z","K"},{"R","R"},{"H","D"}}, {{"D","A"},{"C","G"},{"B","N"}}, {{"L","H"},{"M","P"},{"K","B"}}}; 
  

  //Associated variables
  int hfphi; 
  char sidesign, S_side;
  
  //For slb and rct
  int phi, phideg, etaslb, oddcard, eta2, eta3, phimod8, ietamod;
  int crazy = 0;

  //For HO
  int phmod24, phimod6, sidear, iph, iet, isid, ring, sector;
  bool phi1458, phi271011, phir0v1, phir0v2, phir0v3, phir0v4;
  

  //Chris's original variables (I got rid of iphi_loc; it's not longer necessary)
  int ic,is,ih,itb,ifb,ifc,ifwtb;

  //Write the header for the HBEF XML
  
  //writeXMLHeader(OutXML, 0);

  // Kukartsev
  HTRPatternConfig _config;    

  /* all HBHE crates */
  for(ic=0; ic<NHBHECR; ic++){

    char _buf[50];
    sprintf( _buf, "testHTRPatterns_%d.xml", hbhecrate[ic] );
    string _fileNameByCrate = _buf;

    bool firstBlockWritten = false;

    getNewDocument( theFileName );

    /* four sets of three htrs per crate */
    for(is=0; is<NHSETS; is++){
      /* three htrs per set */
      for(ih=0; ih<NHTRS; ih++){
	/* top and bottom */
	for(itb=0; itb<NTOPBOT; itb++){
	  /* eight fibers per HTR FPGA */
	  for(ifb=0; ifb<NFBR; ifb++){
	    /* three channels per fiber */
	    for(ifc=0; ifc<NFCH; ifc++){
	      icrate=hbhecrate[ic];
	      iside=is<NHSETS/2?-1:1;
	      ifwtb=(is/2+itb+1)%2;
	      ieta=ihbheetadepth[ih][ifwtb][ifb][ifc][0];
	      idepth=ihbheetadepth[ih][ifwtb][ifb][ifc][1];
	      ihtr=ihslot[is]+ih;
	      (ieta>16||idepth>2) ? det = "HE": det = "HB";
	      (itb%2)==1 ? fpga = "bot" : fpga = "top";
	      ihtr_fi=ifb+1;
	      ifi_ch=ifc;
	      iphi=(ieta>20)?(ihbhephis[ic]+(is%2)*4+itb*2-1)%72+1:(ihbhephis[ic]+(is%2)*4+itb*2+(ifb/2+is/2+1)%2-1)%72+1;
	      ispigot=(is%2)*6+ih*2+itb;
	      idcc=is<NHSETS/2?1:2;
	      idcc_sl=idcc==1?9:19;
	      ifed=fedhbhenum[ic][idcc-1];
	      //Aram's insert: rm variables, rbx, wedge
	      //Careful here: per Pawel's map, the rm fiber is the first entry an the rm itself is the second.
	      
	      //If iside == -1, switch top and bottom. Why?
	      if (iside == -1){
		S_side = '-';
		sidesign = 'M';
		irm    = irm_rmfiHBHE[ih][(itb + 1) % 2][ifb][1];
		irm_fi = irm_rmfiHBHE[ih][(itb + 1) % 2][ifb][0];
		
		//For eta >=21, the phi's cover 10 degrees rather than 5 (see HCAL TDR)
		if (ieta >= 21 && (irm == 1 || irm == 3)) iwedge = (iphi + 1 + irm + 1) / 4;
		else                            	  iwedge = (iphi + irm + 1) / 4;
		
		//Roll over the wedge
		if (iwedge > 18) iwedge -= 18;
	      }
	      else{
		S_side = '+';
		sidesign = 'P';
		irm    = irm_rmfiHBHE[ih][itb][ifb][1];
		irm_fi = irm_rmfiHBHE[ih][itb][ifb][0];
		
		//For eta >=21, the phi's cover 10 degrees rather than 5 (see HCAL TDR)
		if (ieta >= 21 && (irm == 4 || irm == 2)) iwedge = (iphi + 1 - irm + 6) / 4;
		else		                          iwedge = (iphi - irm + 6) / 4;
		
		//Roll over the wedge
		if (iwedge > 18) iwedge -= 18;
	      }
	      
	      sprintf (tempbuff, "%s%c%i%c", det.c_str(), sidesign, iwedge,'\0');
	      mystream<<tempbuff;
	      rbx = mystream.str();
	      mystream.str("");

	      //Note that irm_fi ranges from 2 to 7 whereas arrays start at 0 so 
	      //I use irm_fi - 2. Likewise, irm goes from 1 to 4 so use irm - 1
	      
	     //Pixel is split by HB and HE
	      if (ieta > 16 || idepth > 2) ipixel = ipixelHE[irm_fi - 2][ifc][irm - 1]; //HE
	      else               	   ipixel = ipixelHB[irm_fi - 2][ifc][irm - 1]; //HB
	      	      
	      iqie = iadcquiHBHE[irm_fi - 2][ifc][0];
	      iadc = iadcquiHBHE[irm_fi - 2][ifc][1];
	     
	      //Calculate rctcrate
	    
	      // 	     int phi72 = (phi-1)/72;
	      // 	     phi = phi - 72*phi72;
	      // 	     if (phi < 1 ) phi = phi + 72;
	      
	      //The code commented out above appears to do absolutely nothing.	     
 
	      phideg = iphi - 3;
	      if (phideg < 0) phideg = phideg + 72;
	      phideg = (phideg / 4) * 20 + 10;
	      irctcra = (( 89 - phideg  + 720)%360)/20;
	      oddcard = irctcra % 2;
	      irctcra /= 2;
	      if (iside > 0) irctcra = irctcra + 9;
	      
	      etaslb = ((ieta - 1) / 2) * 2 + 1;
	      if (etaslb > 27) etaslb = 27;
	      
	      
	      sprintf(tempbuff,"SLB_H_%3.3d%c%2.2d%c",phideg,S_side,etaslb,'\0');
	      mystream<<tempbuff;
	      slnam = mystream.str();
	      mystream.str("");

	      islb = slb_table[ieta - 1];
	      
	      // calculate RCT destination (that is, rctcon, rctcar and rctnam
	      if (ieta <= 24) { // these are the normal cards 0-5
		irctcar = 2 * ((ieta - 1)/8) + oddcard;
		irctcon = 2 * (((ieta - 1)/2)%4);
	      }
	      else {            // these are on the special card 6 which folds back eta on the odd card half
		irctcar = 6;
		eta2 = ieta;
		if (eta2 > 28) eta2 = 28;
		if (oddcard == 0) eta3 = eta2;
		else              eta3 = 57 - eta2;
		irctcon =  2 * (((eta3 - 1) / 2) % 4);
	      }
	      irctcon = 11 * irctcon + 1;

	      sprintf(tempbuff,"%s-%1d-HD%2.2d",rct_rackHBHE[irctcra],irctcar,irctcon);
	      mystream<<tempbuff;
	      rctnam = mystream.str();
	      mystream.str("");

	      //Finally, the slbin
	      
	      phimod8 = iphi % 8;

	      for (i = 0; i < 18; i++){
		if (iphi < i * 4 + 3){
		  crazy = i % 2;
		  break;
		}
	      } 
	      
	      int ietamod;   // determine if eta is "odd" or "even". 
	      if (ieta == 29) ietamod = 0;
	      else            ietamod = ieta % 2;
	      if (ieta < 25) {         // use the regular table
		if (ietamod == 1) mystream<<S_slbin_odd[phimod8];
		else              mystream<<S_slbin_even[phimod8];
	      }
	      else if (crazy == 0) {   // use the regular table
		if (ietamod == 1) mystream<<S_slbin_odd[phimod8];
		else              mystream<<S_slbin_even[phimod8];
	      }
	      else {                   // swap odd/even!!!
		if (ietamod == 1) mystream<<S_slbin_even[phimod8];
		else              mystream<<S_slbin_odd[phimod8];
	      }  
	      
	      slbin = mystream.str();
	      mystream.str("");

	      if (ieta > 20){
		idphi = 2;
		slbin2 = slbin;
		slbin2[1] = '1';
	      }
	      else{
		idphi = 1;
		slbin2 = "NA";
	      }
	      
	      //printHBHEHF();
	      //writeXML(OutXML);

	      int _crate = icrate;
	      int _slot = ihtr;
	      int _topbottom = itb; // string _topbottom = fpga;
	      int _fiber = ihtr_fi;
	      //===> Kukartsev:
	      if ( !firstBlockWritten || _crate != _config . crate || _slot != _config . slot || _topbottom != _config . topbottom || _fiber != _config . fiber )
		{
		  firstBlockWritten = true;
		  _config . crate = _crate;
		  _config . slot = _slot;
		  _config . topbottom = _topbottom;
		  _config . fiber = _fiber;
		  _config . generalizedindex = 100*_slot + 10*_topbottom + _fiber;
		  _config . creationstamp = getTimestamp( time( NULL ) );
		  addPattern( &_config );
		  firstBlockWritten = true;
		}
	    }
	  }
	}
      }
    }
    write( _fileNameByCrate );
  }

  /* all HF crates */
  for(ic=0; ic<NHFCR; ic++){

    char _buf[50];
    sprintf( _buf, "testHTRPatterns_%d.xml", hfcrate[ic] );
    string _fileNameByCrate = _buf;

    bool firstBlockWritten = false;

    getNewDocument( theFileName );

    /* four sets of three htrs per crate */
    for(is=0; is<NHSETS; is++){
      /* three htrs per set */
      for(ih=0; ih<NHTRS; ih++){
	/* top and bottom */
	for(itb=0; itb<NTOPBOT; itb++){
	  /* eight fibers per HTR FPGA */
	  for(ifb=0; ifb<NFBR; ifb++){
	    /* three channels per fiber */
	    for(ifc=0; ifc<NFCH; ifc++){
	      icrate=hfcrate[ic];
	      iside=is<NHSETS/2?-1:1;
	      ieta=ihfetadepth[itb][ifb][ifc][0];
	      idepth=ihfetadepth[itb][ifb][ifc][1];
	      ihtr=ihslot[is]+ih;
	      det = "HF";
	      (itb%2)== 1 ? fpga = "bot" : fpga = "top";
	      ihtr_fi=ifb+1;
	      ifi_ch=ifc;
	      iphi=(ieta>39)?(ihfphis[ic]+(is%2)*12+ih*4-1)%72+1:(ihfphis[ic]+(is%2)*12+ih*4+(ifb/4)*2-1)%72+1;
	      ispigot=(is%2)*6+ih*2+itb;
	      idcc=is<NHSETS/2?1:2;
	      idcc_sl=idcc==1?9:19;
	      ifed=fedhfnum[ic][idcc-1];
	      
	      irm    = irm_rmfiHF[ih][itb][ifb][1];
	      irm_fi = irm_rmfiHF[ih][itb][ifb][0];

	      //Don't switch in the HF. Why?
	      if (iside == -1){
		S_side = '-';
		sidesign = 'M';
	      }
	      else{
		S_side = '+';
		sidesign = 'P';
	      }
	      
	      if (iphi >= 71) iwedge = 1;
	      else	      iwedge = (iphi + 1) / 4 + 1;
	      
	      
	      //eta == 40 is special. The cell is twice as wide
	      if (ieta == 40) hfphi = (iphi + 3) / 2;
	      else            hfphi = (iphi + 1) / 2;
	    
	      //In case it rolls over
	      if (hfphi > 36) hfphi -= 36;
	      	      
	      sprintf (tempbuff, "%s%c%i%c", det.c_str(), sidesign, (hfphi-1)/3 + 1,'\0');
	      mystream<<tempbuff;
	      rbx = mystream.str();
	      mystream.str("");

	      //No pixel in HF, follow Fedor's convention
	      ipixel = 0;
	     
	      //Integer truncation again consistent with Fedor's map. 
	      iqie = (irm_fi - 1) / 2 + 1;

	      if (irm_fi % 2 != 0) iadc = ifi_ch;
	      else       	   iadc = NFCH + (ifi_ch + 1) % 3;
	      	     
	      //slb and rct variables
	      //rctcrate
	      phideg = iphi - 3;
	      if (phideg < 0) phideg = phideg + 72;
	      phideg = (phideg / 4) * 20 + 10;
	      irctcra = (( 89 - phideg  + 720)%360)/40;
	      if (iside > 0) irctcra = irctcra + 9;
	      
	      //rct card and rct connector appear to be dummy here -- again, I follow Fedor's convention
	      irctcar = 99;
	      irctcon = 0;
	      
	      etaslb = 29;

	      sprintf(tempbuff,"SLB_H_%3.3d%c%2.2d",phideg,S_side,etaslb);
	      mystream<<tempbuff;
	      slnam = mystream.str();
	      mystream.str("");

	      sprintf(tempbuff,"%s-JSC-HF_IN",rct_rackHF[irctcra]);
	      mystream<<tempbuff;
	      rctnam = mystream.str();
	      mystream.str("");

	      islb = 6;
	      
	      int phibin = (iphi + 1) % 8 ;
	      int etabin = (ieta - 29) / 3;
	      if (etabin < 0) etabin = 0;
	      if (etabin > 3) etabin = 3;
	      if (phibin < 4) mystream<<S_slbin_7[etabin];
	      else            mystream<<S_slbin_3[etabin];

	      slbin = mystream.str();
	      mystream.str("");

	      //It has been decided that we not do this
// 	      slbin2 = slbin;
// 	      if      (slbin[0] == 'A') slbin2[0] = 'C';
// 	      else if (slbin[0] == 'B') slbin2[0] = 'D';
// 	      else if (slbin[0] == 'C') slbin2[0] = 'A';
// 	      else if (slbin[0] == 'D') slbin2[0] = 'B';
	     
	      slbin2 = "NA";

	      if (ieta < 40) idphi = 2;
	      else 	     idphi = 4;
	      
	      //printHBHEHF();


	      int _crate = icrate;
	      int _slot = ihtr;
	      int _topbottom = itb; // string _topbottom = fpga;
	      int _fiber = ihtr_fi;
	      //===> Kukartsev:
	      if ( !firstBlockWritten || _crate != _config . crate || _slot != _config . slot || _topbottom != _config . topbottom || _fiber != _config . fiber )
		{
		  firstBlockWritten = true;
		  _config . crate = _crate;
		  _config . slot = _slot;
		  _config . topbottom = _topbottom;
		  _config . fiber = _fiber;
		  _config . generalizedindex = 100*_slot + 10*_topbottom + _fiber;
		  _config . creationstamp = getTimestamp( time( NULL ) );
		  addPattern( &_config );
		  firstBlockWritten = true;
		}
	    }
	  }
	}
      }
    }
    write( _fileNameByCrate );
  }
  
  //writeXMLFooter(OutXML);       //End HBEF XML
 
  //writeXMLHeader(OutXMLHO, 1);  //Begin HO XML

  titlecounter = 0;

  configByCrate . clear();

  //Radical change: HO iterates over eta and phi rather than crate, HTR, etc. 
  
  for(isid = -1; isid < 2; isid+=2){
    for (iph = 0; iph < NHOPHI; iph++){
      for (iet = 0; iet < NHOETA; iet++){
	
	iphi = iph + 1;
	ieta = iet + 1;
	iside = isid;

	if (iphi >= 71 || iphi < 17)      ic = 0;
	else if (iphi >= 17 && iphi < 35) ic = 1;
	else if (iphi >= 35 && iphi < 53) ic = 2;
	else                              ic = 3;
	
	icrate=hocrate[ic];
	idepth=4;
	det = "HO";
	
	ihtr_fi=ifb+1;
	ifi_ch=ifc;
	
	//fpga = top/bottom depends on a pattern that repeats every 30 degrees (6 phi)
	//Hence, phimod6

	phimod6 = iphi % 6;
	
	if (ieta >= 5){
	  if (phimod6 <= 4 && phimod6 >= 2) fpga = "top";
	  else                              fpga = "bot";
	}
	else if (ieta == 4){
	  if (iside == 1){
	    if (phimod6 == 1 || phimod6 == 2) fpga = "bot";
	    else                              fpga = "top";
	  }
	  else{
	    if (phimod6 == 3 || phimod6 == 4) fpga = "top";
	    else                              fpga = "bot";
	  }
	}
	else{
	  if (phimod6 % 2 == 0) fpga = "top";
	  else                  fpga = "bot";
	}

	//dphi
	if      (ieta <= 20) idphi = 1;
	else                 idphi = -1000;
	      
	//create values usable in arrays from side and fpga
	if   (iside == 1) sidear = 1;
	else              sidear = 0;
	
	if (fpga == "bot") itb = 1;
	else               itb = 0;

	phmod24 = iph % 24;
	
	//Again, x - 1 because the array starts at 0 while the variables start at 1
	irm = HO_RM_table[phmod24][iet][sidear];
 
	//x - 5 for the eta array for the same reason
	if      (ieta >= 5) irm_fi = HO_RM_fi_eta5to15[ieta - 5];
	else if (ieta <= 3) irm_fi = HO_RM_fi_eta1to4[iph][0][sidear];
	else if (ieta == 4) irm_fi = HO_RM_fi_eta1to4[iph][1][sidear];
	else                irm_fi = -1000;

	//Determine which of HTR in the set belongs here. It depends only on eta and side.
	if (ieta <= 3 || (ieta >= 14 && iside == 1))     ih = 0;
	else if (ieta <= 13 && ieta >= 6 && iside == 1)  ih = 1;
	else if (ieta <= 13 && ieta >= 6 && iside == -1) ih = 3;
	else                                             ih = 2;
	
	//Each value of "is" covers 30 degrees (that is, 6 values of phi). To calculate which ones,
	//I define phimod18. Crates start at phi = 71, 17, 35, 53
	
	if (iphi % 18 == 17 || iphi % 18 <= 4)      is = 0;
	else if (iphi % 18 >= 5 && iphi % 18 <= 10) is = 1;
	else                                        is = 2;

	ihtr=ihslotho[is][ih];
	
	ispigot=ihtr<9?(ihtr-2)*2+itb:(ihtr-13)*2+itb;
	idcc=ihtr<9?1:2;
	idcc_sl = idcc == 1 ? 9:19;
	
	ifed=fedhonum[ic][idcc-1];

	//HTR fiber

	if (ieta >= 5) ihtr_fi = HO_htr_fi_eta5to15[sidear][ieta - 5];
	else if (ieta == 4){
	  if (phimod6 == 0 || phimod6 == 5) ihtr_fi = 3;
	  else if (iside == 1)              ihtr_fi = 2;
	  else                              ihtr_fi = 4;
	}
	else{
	  if (iside == 1){
	    if      (phimod6 == 4 || phimod6 == 3) ihtr_fi = 3;
	    else if (phimod6 == 1 || phimod6 == 2) ihtr_fi = 4;
	    else if (phimod6 == 0 || phimod6 == 5) ihtr_fi = 5;
	  }
	  else{
	    if      (phimod6 == 4 || phimod6 == 3) ihtr_fi = 6;
	    else if (phimod6 == 1 || phimod6 == 2) ihtr_fi = 7;
	    else if (phimod6 == 0 || phimod6 == 5) ihtr_fi = 8;
	  }
	}
	
	//Fiber Channel
	//Eta >= 5 bools
	phi1458   = (iphi % 12 == 1 || iphi % 12 == 4 || iphi % 12 == 5  || iphi % 12 == 8);
       	phi271011 = (iphi % 12 == 2 || iphi % 12 == 7 || iphi % 12 == 10 || iphi % 12 == 11);

	//Ring 0 bools
	phir0v1 = (iphi % 24 == 0 || iphi % 24 == 2 || iphi % 24 == 4  || iphi % 24 == 18 || iphi % 24 == 20 || iphi % 24 == 22);
	phir0v2 = (iphi % 24 == 1 || iphi % 24 == 3 || iphi % 24 == 17 || iphi % 24 == 19 || iphi % 24 == 21 || iphi % 24 == 23);
	//v3: phi 5 to 15 odd; v4: phi 6 to 16 even
	phir0v3 = (iphi % 24 == 5 || iphi % 24 == 7 || iphi % 24 == 9  || iphi % 24 == 11 || iphi % 24 == 13 || iphi % 24 == 15);
	phir0v4 = (iphi % 24 == 6 || iphi % 24 == 8 || iphi % 24 == 10 || iphi % 24 == 12 || iphi % 24 == 14 || iphi % 24 == 16);
	
	if (ieta >= 5){
	  if      (ieta % 2 == 0 && phi1458)       ifi_ch = 0; 
	  else if (ieta % 2 == 0 && iphi % 3 == 0) ifi_ch = 1;
	  else if (ieta % 2 == 0 && phi271011)     ifi_ch = 2;
	  else if (ieta % 2 == 1 && iphi % 3 == 0) ifi_ch = 0; 
	  else if (ieta % 2 == 1 && phi271011)     ifi_ch = 1;
	  else if (ieta % 2 == 1 && phi1458)       ifi_ch = 2;
	}
	else if (ieta == 4){
	  if (iside == -1){
	    if      (phir0v1)       ifi_ch = 0;
	    else if (phir0v4)       ifi_ch = 1;
	    else if (iphi % 2 == 1) ifi_ch = 2;
	  }
	  else{
	    if      (phir0v3)       ifi_ch = 0;
	    else if (phir0v2)       ifi_ch = 1;
	    else if (iphi % 2 == 0) ifi_ch = 2;
	  }
	}
	//eta = -3 and eta = +2
	else if ((ieta == 3 && iside == -1) || (ieta == 2 && iside == 1)){
	  if      (phir0v4)            ifi_ch = 0;
	  else if (phir0v3)            ifi_ch = 1;
	  else if (phir0v1 || phir0v2) ifi_ch = 2;
	}
	//eta = -2 and eta = +3
	else if ((ieta == 3 && iside == 1) || (ieta == 2 && iside == -1)){
	  if      (phir0v2)            ifi_ch = 0;
	  else if (phir0v1)            ifi_ch = 1;
	  else if (phir0v3 || phir0v4) ifi_ch = 2;
	}
	//ieta = 1
	else if (ieta == 1){
	  if      (phir0v1 || phir0v3) ifi_ch = 0;
	  else if (phir0v2 || phir0v4) ifi_ch = 1;
	}
	
	//Intentional integer truncation; iqie and iadc are the same across all subdetectors
	//(Although irm_fi for HF starts at 1 and for HO it starts at 2, so one can't just copy and paste)
	iqie = (irm_fi - 2) / 2 + 1;
	
	if (irm_fi % 2 == 0) iadc = ifi_ch;
	else       	     iadc = NFCH + (ifi_ch + 1) % 3;

	//Pixel and Letter Code (Ring 0 is separate)
	if (ieta <= 4){
	  ipixel = ipixelHO[irm_fi - 2][ifi_ch][0];
	  letter = letterHO[irm_fi - 2][ifi_ch][0];
	}
	else{
	  ipixel = ipixelHO[irm_fi - 2][ifi_ch][1];
	  letter = letterHO[irm_fi - 2][ifi_ch][1];
	}
	
	//RBX and sector

	if (iside == -1) sidesign = 'M';
	else      	 sidesign = 'P';
	      
	if      (ieta <= 4)                ring = 0;
	else if (ieta >= 5 && ieta <= 10)  ring = 1;
	else                               ring = 2;
	
	//Sector ranges from 1 to 12 depending on phi. Sector 1 goes 71,72,1,2,3,4 so I start at -2
	sector = 0;
	for (i = -2; i < iphi; i+=6){
	  sector++;
	}
	if (sector > 12) sector = 1; //It rolls over for phi = 71,72 

	isector = sector;

	//For rings 1 and 2, we only want even sectors for the rbx
	if (ring != 0 && sector % 2 != 0) sector++;

	if (ring == 0)  sprintf (tempbuff, "%s%i%2.2d", det.c_str(), ring, sector);
	else            sprintf (tempbuff, "%s%i%c%2.2d", det.c_str(), ring, sidesign, sector);
	mystream<<tempbuff;
	rbx = mystream.str();
	mystream.str("");

	iwedge = -1000;
	irctcra = -1000;
	irctcar = -1000;
	irctcon = -1000;
	rctnam = "RCTNAM";

	islb = -1000;
	slbin = "SLBIN";
	slnam = "SLNAM";
	slbin2 = "NA";
	      
	//printHO();
	//writeXMLHO(OutXMLHO);

	// Kukartsev
	int _crate = icrate;
	int _slot = ihtr;
	int _topbottom = itb; // string _topbottom = fpga;
	int _fiber = ihtr_fi;
	HTRPatternConfig _config;
	_config . crate = _crate;
	_config . slot = _slot;
	_config . topbottom = _topbottom;
	_config . fiber = _fiber;
	fillPatternInfo( _config );
      }
    }
  }
  //writeXMLFooter(OutXMLHO);  //End HO XML


  cout << configByCrate . size() << endl << endl;

  int currentCrate = -1;
  string _fileNameByHOCrate;  
  for( map< HTRPatternID, int >::const_iterator i = configByCrate . begin(); i != configByCrate . end(); i++ )
    {
      //cout << i -> second << endl;
      if ( currentCrate != i -> second )
	{
	  if ( currentCrate != -1 ) write( _fileNameByHOCrate );
	  char _buf[50];
	  sprintf( _buf, "testHTRPatterns_%d.xml", i -> second );
	  _fileNameByHOCrate = _buf;
	  getNewDocument( theFileName );
	}

      int _crate = i -> first . crate;
      int _slot = i -> first . slot;
      int _topbottom = i -> first . topbottom;
      int _fiber = i -> first . fiber;

      _config . crate = _crate;
      _config . slot = _slot;
      _config . topbottom = _topbottom;
      _config . fiber = _fiber;
      _config . generalizedindex = 100*_slot + 10*_topbottom + _fiber;
      _config . creationstamp = getTimestamp( time( NULL ) );
      addPattern( &_config );      
      
      currentCrate = i -> second;
    }

  //write the last file
  write( _fileNameByHOCrate );

}

//
// const member functions
//

//
// static member functions
//
