// -*- C++ -*-
//
// Package:    GctTimingAnalyzer
// Class:      GctTimingAnalyzer
// 
/**\class GctTimingAnalyzer GctTimingAnalyzer.cc L1Trigger/L1GctAnalzyer/src/GctTimingAnalyzer.cc

Description: Analyse the timing of all of the GCT pipelines

*/
//
// Original Author:  Alex Tapper
//         Created:  Mon Apr 21 14:21:06 CEST 2008
// $Id: GctTimingAnalyzer.h,v 1.2 2008/04/21 14:50:39 tapper Exp $
//
//

#include "FWCore/MessageLogger/interface/MessageLogger.h" // Logger
#include "FWCore/Utilities/interface/Exception.h" // Exceptions

// Include file
#include "L1Trigger/L1GctAnalyzer/interface/GctTimingAnalyzer.h"

GctTimingAnalyzer::GctTimingAnalyzer(const edm::ParameterSet& iConfig):
  m_outputFile(iConfig.getUntrackedParameter<std::string>("outFile", "gctTiming.txt")),
  m_dataSource(iConfig.getUntrackedParameter<edm::InputTag>("dataSource")
{
}

GctTimingAnalyzer::~GctTimingAnalyzer()
{
}

void GctTimingAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
//   Handle<L1GctFibreCollection> fibre;
//   iEvent.getByLabel(m_fibreSource,fibre);

//   bool bc0= false;

//   for (L1GctFibreCollection::const_iterator f=fibre->begin(); f!=fibre->end(); f++){

//     // Check for corrupt fibre data
//     if (!CheckFibreWord(*f)){
//       edm::LogInfo("GCT fibre data error") << "Missing phase bit (clock) in fibre data " << (*f);
//     }
    
//     // Check for BC0
//     if (CheckForBC0(*f) && (f==fibre->begin())) {
//       bc0=true;
//     }

//     // Check for mismatch between fibres
//     if ((bc0 && !CheckForBC0(*f)) ||
//         (!bc0 && CheckForBC0(*f))){
//       edm::LogInfo("GCT fibre data error") << "BC0 mismatch in fibre data " << (*f);
//     }

//     // Check logical ID pattern
//     if (m_doLogicalID) CheckLogicalID(*f);

//     // Check counter pattern
//     if (m_doCounter) CheckCounter(*f);

//   }   
}

// bool GctFibreAnalyzer::CheckForBC0(const L1GctFibreWord fibre)
// {
//   // Check for BC0 on this event
//   if (fibre.data() & 0x8000){
//     return true;
//   } else {
//     return false;
//   }
// }

// bool GctFibreAnalyzer::CheckFibreWord(const L1GctFibreWord fibre)
// {
//   // Check that the phase or clock bit (MSB bit on cycle 1) is set as it should be
//   if (fibre.data() & 0x80000000){
//     return true;
//   } else {
//     return false;
//   }
// }

// void GctFibreAnalyzer::CheckLogicalID(const L1GctFibreWord fibre)
// {

//   // Check that data data in cycle 0 and cycle 1 are equal
//   if ((fibre.data()&0x7FFF)!=((fibre.data()&0x7FFF0000)>>16)){
//     edm::LogInfo("GCT fibre data error") << "Fibre data different on cycles 0 and 1 " << fibre;
//   }

//   // Decode the fibre data
//   int sourceFibreNumber, sourceLogicalID;
    
//   sourceFibreNumber = (fibre.data() & 0x7)-1; 
//   sourceLogicalID   = (fibre.data() & 0x7F00) >> 8;
    
//   // Calculate logical ID and fibre number from block and index
//   int concFibreNumber, concRctCrate;

//   switch (fibre.block()){
    
//   case 0x884:
//     concRctCrate = fibre.index()/3;
//     concFibreNumber = 1+(fibre.index()%3);
//     break;

//   case 0x804:
//     concRctCrate = 4+(fibre.index()/3);
//     concFibreNumber = 1+(fibre.index()%3);
//     break;
      
//   case 0xc84:
//     concRctCrate = 9+(fibre.index()/3);
//     concFibreNumber = 1+(fibre.index()%3);
//     break;
      
//   case 0xc04:
//     concRctCrate = 13+(fibre.index()/3);
//     concFibreNumber = 1+(fibre.index()%3);
//     break;

//   default:
//     throw cms::Exception("Unknown GCT fibre data block ") << fibre.block();    
//   }
    
//   // Calculate logical ID from crate and fibre number
//   int concLogicalID;
    
//   if (concRctCrate>=9){
//     concLogicalID=8*(concRctCrate-9)+4;
//   } else {
//     concLogicalID=8*(concRctCrate);
//   }

//   // Check to see if logical IDs are consistent
//   if (concLogicalID!=sourceLogicalID){
//     edm::LogInfo("GCT fibre data error") << "Logical IDs are different " 
//                                          << "Source card logical ID=" << sourceLogicalID
//                                          << " Conc card logical ID=" << concLogicalID
//                                          << " " << fibre;      
//   }

//   // Check to see if fibre numbers are consistent
//   if (concFibreNumber!=sourceFibreNumber){
//     edm::LogInfo("GCT fibre data error") << "Fibre numbers are different " 
//                                          << "Source card fibre number=" << sourceFibreNumber
//                                          << " Conc card fibre number=" << concFibreNumber
//                                          << " " << fibre;      
//   }
    
// }

// void GctFibreAnalyzer::CheckCounter(const L1GctFibreWord fibre)
// {
//   // Remove MSB from both cycles
//   int cycle0Data, cycle1Data;
  
//   cycle0Data = fibre.data() & 0x7FFF;
//   cycle1Data = (fibre.data() >> 16) & 0x7FFF;

//   // Check to see if fibre numbers are consistent
//   if ((cycle0Data+1)!=cycle1Data){
//     edm::LogInfo("GCT fibre data error") << "Fibre data not incrementing in cycles 0 and 1 "
//                                          << " Cycle 0 data=" << cycle0Data
//                                          << " Cycle 1 data=" << cycle1Data
//                                          << " " << fibre;      
//   }  
  
//   // For now just write out the data
//   edm::LogInfo("GCT fibre counter data") << " Fibre data: cycle0=" << cycle0Data 
//                                          << " cycle1=" << cycle1Data
//                                          << " " << fibrenn;
// }





