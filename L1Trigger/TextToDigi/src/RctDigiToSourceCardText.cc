// -*- C++ -*-
//
// Package:    RctDigiToSourceCardText
// Class:      RctDigiToSourceCardText
// 
/**\class RctDigiToSourceCardText RctDigiToSourceCardText.cc L1Trigger/TextToDigi/src/RctDigiToSourceCardText.cc

 Description: Input RCT digis and output GCT text file to be loaded into the source cards for pattern tests. 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Feb 16 14:52:19 CET 2007
// $Id$
//
//

#include "L1Trigger/TextToDigi/src/RctDigiToSourceCardText.h"
#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "FWCore/MessageLogger/interface/MessageLogger.h" // Logger

using namespace edm;
using namespace std;

// Set constant
const static unsigned NUM_RCT_CRATES = 18;

RctDigiToSourceCardText::RctDigiToSourceCardText(const edm::ParameterSet& iConfig):
  m_rctInputLabel(iConfig.getParameter<edm::InputTag>("RctInputLabel")),
  m_textFileName(iConfig.getParameter<std::string>("TextFileName")),
  m_nevt(0)
{
  // Open the output file
  m_file.open(m_textFileName.c_str(),ios::out);

  if(!m_file.good())
  {
    throw cms::Exception("RctDigiToSourceCardTextFileOpenError")
      << "RctDigiToSourceCardText::RctDigiToSourceCardText : "
      << " couldn't open the file " + m_textFileName + " for writing" << endl;
  }

  // Make a SC routing object
  SourceCardRouting m_scRouting; 
}


RctDigiToSourceCardText::~RctDigiToSourceCardText()
{
  // Close the output file
  m_file.close();
}

// ------------ method called to for each event  ------------
void RctDigiToSourceCardText::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get the RCT data
  Handle<L1CaloEmCollection> em;
  Handle<L1CaloRegionCollection> rgn;
  iEvent.getByLabel(m_rctInputLabel, em);
  iEvent.getByLabel(m_rctInputLabel, rgn);

  // Have to arrange digis into arrays for each RCT crate
  for (int crate=0; crate<NUM_RCT_CRATES; crate++){

    // Arrays etc. 
    unsigned short logicalCardID;
    unsigned short eIsoRank[4]={0,0,0,0};
    unsigned short eIsoCardId[4]={0,0,0,0};
    unsigned short eIsoRegionId[4]={0,0,0,0};
    unsigned short eNonIsoRank[4]={0,0,0,0};
    unsigned short eNonIsoCardId[4]={0,0,0,0};
    unsigned short eNonIsoRegionId[4]={0,0,0,0};
    unsigned short MIPbits[7]={0,0,0,0,0,0,0};
    unsigned short Qbits[7]={0,0,0,0,0,0,0};
    string dataString;

    unsigned numIsoEM=0;
    unsigned numNonIsoEM=0;
    
    // Fill electrons
    for (L1CaloEmCollection::const_iterator iem=em->begin(); iem!=em->end(); iem++){
      if (iem->rctCrate()==crate){
        if (iem->isolated()){
          eIsoRank[numIsoEM]=iem->rank();
          eIsoCardId[numIsoEM]=iem->rctCard();
          eIsoRegionId[numIsoEM]=iem->rctRegion();
          numIsoEM++;
        } else {
          eNonIsoRank[numIsoEM]=iem->rank();
          eNonIsoCardId[numIsoEM]=iem->rctCard();
          eNonIsoRegionId[numIsoEM]=iem->rctRegion();
          numNonIsoEM++;
        }
      }
    }

    // Logical Card ID = Source Card number
    int RoutingMode = 0;
    m_scRouting.RoutingModetoLogicalCardID(logicalCardID,RoutingMode,crate);

    // Debug info
    LogDebug("Electrons") << "Crate=" << crate << " LogicalCardID=" << logicalCardID << " Event=" << m_nevt << endl;;
    for (int i=0; i<4; i++){
      LogDebug("Electrons") << "i=" << i 
                            << " IsoEmRank=" << eIsoRank[i]
                            << " IsoEmCardId=" << eIsoCardId[i]
                            << " IsoEmRegionId=" << eIsoRegionId[i]
                            << " NonIsoRank=" << eNonIsoRank[i]
                            << " NonIsoCardId=" << eNonIsoCardId[i]
                            << " NonIsoRegionId=" << eNonIsoRegionId[i] << endl;
    }
    
    // Convert electrons to SC format
    m_scRouting.EMUtoSTRING(logicalCardID,
                            m_nevt,
                            eIsoRank,
                            eIsoCardId,
                            eIsoRegionId,
                            eNonIsoRank,
                            eNonIsoCardId,
                            eNonIsoRegionId,
                            MIPbits,
                            Qbits,
                            dataString);

    // Write electrons
    m_file << dataString;


//     // Fill regions
//     for (L1CaloRegionCollection::const_iterator irgn=rgn->begin(); irgn!=rgn->end(); irgn++){

//     }  

// //RC arrays are RC[receiver card number<7][region<2]
// //HF arrays are HF[eta<4][HF region<2]
//     void RC56HFtoSTRING(	unsigned short &logicalCardID,
// 			unsigned short &eventNumber,
// 			unsigned short (&RC)[7][2],
// 			unsigned short (&RCof)[7][2],
// 			unsigned short (&RCtau)[7][2],
// 			unsigned short (&HF)[4][2],
// 			unsigned short (&HFQ)[4][2],
// 			std::string &dataString	);

// //RC arrays are RC[receiver card number<7][region<2]
//     void RC012toSTRING(	unsigned short &logicalCardID,
// 			unsigned short &eventNumber,
// 			unsigned short (&RC)[7][2],
// 			unsigned short (&RCof)[7][2],
// 			unsigned short (&RCtau)[7][2],
// 			std::string &dataString	);

// //RC arrays are RC[receiver card number<7][region<2]
//     void RC234toSTRING(	unsigned short &logicalCardID,
// 			unsigned short &eventNumber,
// 			unsigned short (&RC)[7][2],
// 			unsigned short (&RCof)[7][2],
// 			unsigned short (&RCtau)[7][2],
// 			unsigned short (&sisterRC)[7][2],
// 			unsigned short (&sisterRCof)[7][2],
// 			unsigned short (&sisterRCtau)[7][2],
// 			std::string &dataString	);

  }
  m_nevt++;
}


