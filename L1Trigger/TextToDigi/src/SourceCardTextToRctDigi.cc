// -*- C++ -*-
//
// Package:    SourceCardTextToRctDigi
// Class:      SourceCardTextToRctDigi
// 
/**\class SourceCardTextToRctDigi SourceCardTextToRctDigi.cc L1Trigger/TextToDigi/src/SourceCardTextToRctDigi.cc

Description: Input text file to be loaded into the source cards and output RCT digis for pattern tests. 

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Mar  9 19:11:51 CET 2007
// $Id: SourceCardTextToRctDigi.cc,v 1.1 2007/03/12 18:30:05 tapper Exp $
//
//

#include "L1Trigger/TextToDigi/src/SourceCardTextToRctDigi.h"
#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "FWCore/MessageLogger/interface/MessageLogger.h" // Logger

using namespace edm;
using namespace std;

// Set constant
const static unsigned NUM_RCT_CRATES = 18;

SourceCardTextToRctDigi::SourceCardTextToRctDigi(const edm::ParameterSet& iConfig):
  m_textFileName(iConfig.getParameter<std::string>("TextFileName"))
{
  // Produces collections
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();

  // Open the input file
  m_file.open(m_textFileName.c_str(),ios::in);

  if(!m_file.good())
    {
      throw cms::Exception("SourceCardTextToRctDigiTextFileOpenError")
        << "SourceCardTextToRctDigi::SourceCardTextToRctDigi : "
        << " couldn't open the file " << m_textFileName << " for reading" << endl;
    }

  // Make a SC routing object
  SourceCardRouting m_scRouting; 

}

SourceCardTextToRctDigi::~SourceCardTextToRctDigi()
{
  // Close the input file
  m_file.close();
}

// ------------ method called to produce the data  ------------
void SourceCardTextToRctDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  auto_ptr<L1CaloEmCollection> em (new L1CaloEmCollection);
  auto_ptr<L1CaloRegionCollection> rgn (new L1CaloRegionCollection);

  // Arrays etc.  
  unsigned long VHDCI[2][2];
  int routingMode;
  int crate;
  unsigned short eventNumber;
  unsigned short logicalCardID;
  unsigned short eIsoRank[4];
  unsigned short eIsoCardId[4];
  unsigned short eIsoRegionId[4];
  unsigned short eNonIsoRank[4];
  unsigned short eNonIsoCardId[4];
  unsigned short eNonIsoRegionId[4];
  unsigned short MIPbits[7];
  unsigned short Qbits[7];
  string dataString; 

  // Have to read one line per RCT crate, though order doesn't matter
  for (int i=0; i<NUM_RCT_CRATES; i++){  

    if(!getline(m_file,dataString))
    {
      throw cms::Exception("SourceCardTextToRctDigiTextFileReadError")
        << "SourceCardTextToRctDigi::SourceCardTextToRctDigi : "
        << " couldn't read from the file " << m_textFileName << endl;
    }   

    // Convert the string to useful info
    m_scRouting.STRINGtoVHDCI(logicalCardID,eventNumber,dataString,VHDCI);
 
    // Are we looking at electrons or regions
    m_scRouting.LogicalCardIDtoRoutingMode(logicalCardID,routingMode,crate); 
    
    if (routingMode==0){     

      // Electrons
      m_scRouting.VHDCItoEMU(eIsoRank,eIsoCardId,eIsoRegionId,
                             eNonIsoRank,eNonIsoCardId,eNonIsoRegionId, 
                             MIPbits,Qbits,VHDCI);

      // Make collections
      for (int i=0; i<4; i++){
        em->push_back(L1CaloEmCand(eIsoRank[i],eIsoRegionId[i],eIsoCardId[i],crate,true));
        em->push_back(L1CaloEmCand(eNonIsoRank[i],eNonIsoRegionId[i],eNonIsoCardId[i],crate,false));
      }

      // Debug info
      LogDebug("Electrons") << "Crate=" << crate << " LogicalCardID=" << logicalCardID << " Event=" << eventNumber << endl;;
      for (int i=0; i<4; i++){
        LogDebug("Electrons") << "i=" << i 
                              << " IsoEmRank=" << eIsoRank[i]
                              << " IsoEmCardId=" << eIsoCardId[i]
                              << " IsoEmRegionId=" << eIsoRegionId[i]
                              << " NonIsoRank=" << eNonIsoRank[i]
                              << " NonIsoCardId=" << eNonIsoCardId[i]
                              << " NonIsoRegionId=" << eNonIsoRegionId[i] << endl;
      }  
    } else {
      // Regions not coded right now so throw an exception
      throw cms::Exception("SourceCardtextToRctDigiError")
        << "SourceCardTextToRctDigi::produce : "
        << " can't handle routing mode=" << routingMode << " (yet!)" << endl;
    }
  }

  iEvent.put(em);
  iEvent.put(rgn);
}

