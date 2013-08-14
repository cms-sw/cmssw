// -*- C++ -*-
//
// Package:    SourceCardTextToRctDigi
// Class:      SourceCardTextToRctDigi
// 
/**\class SourceCardTextToRctDigi SourceCardTextToRctDigi.cc L1Trigger/TextToDigi/src/SourceCardTextToRctDigi.cc

Description: Input text file to be loaded into the source cards and output RCT digis for pattern tests. 

*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Mar  9 19:11:51 CET 2007
// $Id: SourceCardTextToRctDigi.cc,v 1.8 2010/08/06 20:24:40 wmtan Exp $
//
//

#include "SourceCardTextToRctDigi.h"
#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "FWCore/MessageLogger/interface/MessageLogger.h" // Logger

using namespace edm;
using namespace std;

// Set constants
const static unsigned NUM_LINES_PER_EVENT = 63;
const static int NUM_RCT_CRATES = 18;

SourceCardTextToRctDigi::SourceCardTextToRctDigi(const edm::ParameterSet& iConfig):
  m_textFileName(iConfig.getParameter<std::string>("TextFileName")),
  m_fileEventOffset(iConfig.getParameter<int>("fileEventOffset")),
  m_nevt(0)
{
  // Produces collections
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();

  // Open the input file
  m_file.open(m_textFileName.c_str(),std::ios::in);

  if(!m_file.good())
    {
      throw cms::Exception("SourceCardTextToRctDigiTextFileOpenError")
        << "SourceCardTextToRctDigi::SourceCardTextToRctDigi : "
        << " couldn't open the file " << m_textFileName << " for reading" << std::endl;
    }

  // Make a SC routing object
  SourceCardRouting m_scRouting; 

}

SourceCardTextToRctDigi::~SourceCardTextToRctDigi()
{
  // Close the input file
  m_file.close();
}

/// Append empty digi collection
void SourceCardTextToRctDigi::putEmptyDigi(edm::Event& iEvent) {
  std::auto_ptr<L1CaloEmCollection> em (new L1CaloEmCollection);
  std::auto_ptr<L1CaloRegionCollection> rgn (new L1CaloRegionCollection);
  for (int i=0; i<NUM_RCT_CRATES; i++){  
    for (int j=0; j<4; j++) {
      em->push_back(L1CaloEmCand(0, i, true));
      em->push_back(L1CaloEmCand(0, i, false));
    }
    for (int j=0; j<14; j++)
      rgn->push_back(L1CaloRegion(0,false,false,false,false,i,j/2,j%2));
    for (unsigned j=0; j<8; j++)
      rgn->push_back(L1CaloRegion(0,true,i,j));
  }
  iEvent.put(em);
  iEvent.put(rgn);
}

// ------------ method called to produce the data  ------------
void SourceCardTextToRctDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Skip event if required
  if (m_nevt < m_fileEventOffset){
    //    std::string tmp;
    // for (unsigned i=0;i<NUM_LINES_PER_EVENT;i++){
    //getline(m_file,tmp);
    //}
    putEmptyDigi(iEvent);
    m_nevt++;
    return;
  } else if (m_nevt==0 && m_fileEventOffset<0) {
    //skip first fileEventOffset input events
    std::string tmp; 
    for(int i=0;i<abs(m_fileEventOffset); i++)
      for (unsigned line=0; line<NUM_LINES_PER_EVENT; line++)  
	if(!getline(m_file,tmp))
	  {
	    throw cms::Exception("SourceCardTextToRctDigiTextFileReadError")
	      << "SourceCardTextToRctDigi::produce() : "
	      << " couldn't read from the file " << m_textFileName << std::endl;
	  }
  }
  

  // New collections
  std::auto_ptr<L1CaloEmCollection> em (new L1CaloEmCollection);
  std::auto_ptr<L1CaloRegionCollection> rgn (new L1CaloRegionCollection);

  // General variables  
  unsigned long VHDCI[2][2];
  int routingMode;
  int crate;
  std::string dataString; 
  unsigned short eventNumber;
  unsigned short logicalCardID;

  // Arrays to hold electron variables
  unsigned short eIsoRank[18][4];
  unsigned short eIsoCardId[18][4];
  unsigned short eIsoRegionId[18][4];
  unsigned short eNonIsoRank[18][4];
  unsigned short eNonIsoCardId[18][4];
  unsigned short eNonIsoRegionId[18][4];

  // Arrays to hold region variables
  unsigned short RC[18][7][2];
  unsigned short RCof[18][7][2];
  unsigned short RCtau[18][7][2];
  unsigned short MIPbits[18][7][2];
  unsigned short Qbits[18][7][2];
  unsigned short HF[18][4][2];
  unsigned short HFQ[18][4][2];

  // Check we're not at the end of the file
  if(m_file.eof())
    {
      throw cms::Exception("SourceCardTextToRctDigiTextFileReadError")
        << "SourceCardTextToRctDigi::produce : "
        << " unexpected end of file " << m_textFileName << std::endl;
    }      
  
  int thisEventNumber=-1;  
  // Read in file one line at a time 
  for (unsigned line=0; line<NUM_LINES_PER_EVENT; line++){  

    if(!getline(m_file,dataString))
    {
      throw cms::Exception("SourceCardTextToRctDigiTextFileReadError")
        << "SourceCardTextToRctDigi::SourceCardTextToRctDigi : "
        << " couldn't read from the file " << m_textFileName << std::endl;
    }   

    // Convert the string to useful info
    m_scRouting.STRINGtoVHDCI(logicalCardID,eventNumber,dataString,VHDCI);
    
    // Check crossing number
    if(line!=0) assert(eventNumber==thisEventNumber);
    thisEventNumber = eventNumber;

    // Are we looking at electrons or regions
    m_scRouting.LogicalCardIDtoRoutingMode(logicalCardID,routingMode,crate); 
    
    if (routingMode==0){     

      // Electrons
      m_scRouting.VHDCItoEMU(eIsoRank[crate],eIsoCardId[crate],eIsoRegionId[crate],
                             eNonIsoRank[crate],eNonIsoCardId[crate],eNonIsoRegionId[crate], 
                             MIPbits[crate],Qbits[crate],VHDCI);

    } else if (routingMode==1) {

      // Regions
      m_scRouting.VHDCItoRC56HF(RC[crate],RCof[crate],RCtau[crate],HF[crate],HFQ[crate],VHDCI);

    } else if (routingMode==2) {

      // Regions
      m_scRouting.VHDCItoRC012(RC[crate],RCof[crate],RCtau[crate],VHDCI);

    } else if (routingMode==3) {

      // Regions
      m_scRouting.VHDCItoRC234(RC[crate],RCof[crate],RCtau[crate],RC[crate+9],RCof[crate+9],RCtau[crate+9],VHDCI);

    } else {
      // Something went wrong
      throw cms::Exception("SourceCardtextToRctDigiError")
        << "SourceCardTextToRctDigi::produce : "
        << " unknown routing mode=" << routingMode << std::endl;
    }
  }

  // Make RCT digis
  for (crate=0; crate<NUM_RCT_CRATES; crate++){

    // Make EM collections
    for (int i=0; i<4; i++){
      em->push_back(L1CaloEmCand(eIsoRank[crate][i],eIsoRegionId[crate][i],eIsoCardId[crate][i],crate,true,i,0));
      em->push_back(L1CaloEmCand(eNonIsoRank[crate][i],eNonIsoRegionId[crate][i],eNonIsoCardId[crate][i],crate,false,i,0));
    }
    
    // Make region collections
    for (int i=0; i<7; i++){// Receiver card
      for (int j=0; j<2; j++){// Region
        rgn->push_back(L1CaloRegion::makeHBHERegion(RC[crate][i][j],RCof[crate][i][j],RCtau[crate][i][j],MIPbits[crate][i][j],Qbits[crate][i][j],crate,i,j));
      }
    }
    
    // Make HF region collections
    for (int i=0; i<4; i++){// Eta bin
      for (int j=0; j<2; j++){// HF0, HF1
        rgn->push_back(L1CaloRegion::makeHFRegion(HF[crate][i][j],HFQ[crate][i][j],crate,i+(4*j)));// region=eta+4*phi for eta 0-3 
      }
    }
  }

  // Debug info
  for (L1CaloEmCollection::const_iterator iem=em->begin(); iem!=em->end(); iem++){
    LogDebug("Electrons") << (*iem);
  }
  
  for (L1CaloRegionCollection::const_iterator irgn=rgn->begin(); irgn!=rgn->end(); irgn++){
      LogDebug("HFRegions") << (*irgn);
  }

  iEvent.put(em);
  iEvent.put(rgn);

  m_nevt++;
}

