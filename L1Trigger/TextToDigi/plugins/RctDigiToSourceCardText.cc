// -*- C++ -*-
//
// Package:    RctDigiToSourceCardText
// Class:      RctDigiToSourceCardText
// 
/**\class RctDigiToSourceCardText RctDigiToSourceCardText.cc L1Trigger/TextToDigi/src/RctDigiToSourceCardText.cc

Description: Input RCT digis and output GCT text file to be loaded into the source cards for pattern tests. 

*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Feb 16 14:52:19 CET 2007
// $Id: RctDigiToSourceCardText.cc,v 1.4 2010/08/06 20:24:40 wmtan Exp $
//
//

#include "RctDigiToSourceCardText.h"
#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "FWCore/MessageLogger/interface/MessageLogger.h" // Logger

using namespace edm;
using namespace std;

// Set constant
const static int NUM_RCT_CRATES = 18;

RctDigiToSourceCardText::RctDigiToSourceCardText(const edm::ParameterSet& iConfig):
  m_rctInputLabel(iConfig.getParameter<edm::InputTag>("RctInputLabel")),
  m_textFileName(iConfig.getParameter<std::string>("TextFileName")),
  m_nevt(0)
{
  // Open the output file
  m_file.open(m_textFileName.c_str(),std::ios::out);

  if(!m_file.good())
    {
      throw cms::Exception("RctDigiToSourceCardTextFileOpenError")
	<< "RctDigiToSourceCardText::RctDigiToSourceCardText : "
	<< " couldn't open the file " << m_textFileName << " for writing" << std::endl;
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

  // General variables
  int RoutingMode;
  unsigned short logicalCardID;
  std::string dataString;

  // Arrays to hold electron variables
  unsigned short eIsoRank[18][4]={{0}};
  unsigned short eIsoCardId[18][4]={{0}};
  unsigned short eIsoRegionId[18][4]={{0}};
  unsigned short eNonIsoRank[18][4]={{0}};
  unsigned short eNonIsoCardId[18][4]={{0}};
  unsigned short eNonIsoRegionId[18][4]={{0}};

  // Fill electrons
  unsigned numIsoEM[18]={0};
  unsigned numNonIsoEM[18]={0};

  for (L1CaloEmCollection::const_iterator iem=em->begin(); iem!=em->end(); iem++){
    if (iem->isolated()){
      eIsoRank[iem->rctCrate()][numIsoEM[iem->rctCrate()]]=iem->rank();
      eIsoCardId[iem->rctCrate()][numIsoEM[iem->rctCrate()]]=iem->rctCard();
      eIsoRegionId[iem->rctCrate()][numIsoEM[iem->rctCrate()]]=iem->rctRegion();
      numIsoEM[iem->rctCrate()]++;
    } else {
      eNonIsoRank[iem->rctCrate()][numNonIsoEM[iem->rctCrate()]]=iem->rank();
      eNonIsoCardId[iem->rctCrate()][numNonIsoEM[iem->rctCrate()]]=iem->rctCard();
      eNonIsoRegionId[iem->rctCrate()][numNonIsoEM[iem->rctCrate()]]=iem->rctRegion();
      numNonIsoEM[iem->rctCrate()]++;
    }
    // Debug info
    LogDebug("Electrons") << "Rank=" << iem->rank()
                          << " Card=" << iem->rctCard()
                          << " Region=" << iem->rctRegion()
                          << " Crate=" << iem->rctCrate()
                          << " Isolated=" << iem->isolated();
  }

  // Arrays to hold region variables
  unsigned short RC[18][7][2]={{{0}}};
  unsigned short RCof[18][7][2]={{{0}}};
  unsigned short RCtau[18][7][2]={{{0}}};
  unsigned short HF[18][4][2]={{{0}}};
  unsigned short HFQ[18][4][2]={{{0}}};
  unsigned short MIPbits[18][7][2]={{{0}}};
  unsigned short Qbits[18][7][2]={{{0}}};  

  // Fill regions
  for (L1CaloRegionCollection::const_iterator irgn=rgn->begin(); irgn!=rgn->end(); irgn++){
    if (irgn->id().isHf()){
      HF[irgn->rctCrate()][irgn->id().rctEta()-7][irgn->id().rctPhi()]=irgn->et();
      HFQ[irgn->rctCrate()][irgn->id().rctEta()-7][irgn->id().rctPhi()]=irgn->fineGrain();	
      // Debug info
      LogDebug("HFRegions") << "Et=" << irgn->et()
                            << " FineGrain=" << irgn->fineGrain()
                            << " Eta=" << irgn->id().rctEta()
                            << " Phi=" << irgn->id().rctPhi()
                            << " Crate=" << irgn->rctCrate();
    } else {
      RC[irgn->rctCrate()][irgn->rctCard()][irgn->rctRegionIndex()]=irgn->et();		
      RCof[irgn->rctCrate()][irgn->rctCard()][irgn->rctRegionIndex()]=irgn->overFlow();			
      RCtau[irgn->rctCrate()][irgn->rctCard()][irgn->rctRegionIndex()]=irgn->tauVeto();
      MIPbits[irgn->rctCrate()][irgn->rctCard()][irgn->rctRegionIndex()]=irgn->mip();
      Qbits[irgn->rctCrate()][irgn->rctCard()][irgn->rctRegionIndex()]=irgn->quiet();
      // Debug info
      LogDebug("Regions") << "Et=" << irgn->et()
                          << " OverFlow=" << irgn->overFlow()
                          << " tauVeto=" << irgn->tauVeto()
                          << " mip=" << irgn->mip()
                          << " quiet=" << irgn->quiet()
                          << " Card=" << irgn->rctCard()
                          << " Region=" << irgn->rctRegionIndex()
                          << " Crate=" << irgn->rctCrate();
    }
  }

  for (int crate=0; crate<NUM_RCT_CRATES; crate++){

    // Logical Card ID = Source Card number
    RoutingMode=0;
    m_scRouting.RoutingModetoLogicalCardID(logicalCardID,RoutingMode,crate);

    // Convert electrons to SC format
    m_scRouting.EMUtoSTRING(logicalCardID,
			    m_nevt,
			    eIsoRank[crate],
			    eIsoCardId[crate],
			    eIsoRegionId[crate],
			    eNonIsoRank[crate],
			    eNonIsoCardId[crate],
			    eNonIsoRegionId[crate],
			    MIPbits[crate],
			    Qbits[crate],
			    dataString);

    // Write electrons
    m_file << dataString;

    // Logical Card ID = Source Card number
    RoutingMode=1;
    m_scRouting.RoutingModetoLogicalCardID(logicalCardID,RoutingMode,crate);
 
    // Convert regions to SC format
    m_scRouting.RC56HFtoSTRING(logicalCardID,
                               m_nevt,
                               RC[crate],
                               RCof[crate],
                               RCtau[crate],
                               HF[crate],
                               HFQ[crate],
                               dataString);
  
    // Write regions
    m_file << dataString;

    // Logical Card ID = Source Card number  		
    RoutingMode=2;
    m_scRouting.RoutingModetoLogicalCardID(logicalCardID,RoutingMode,crate);

    // Convert regions to SC format  
    m_scRouting.RC012toSTRING(logicalCardID,
                              m_nevt,
                              RC[crate],
                              RCof[crate],
                              RCtau[crate],
                              dataString);

    // Write regions
    m_file << dataString;

    // This is to 9 only as this is the shared source card
    if (crate<9){ 
      // Logical Card ID = Source Card number
      RoutingMode=3;					
      m_scRouting.RoutingModetoLogicalCardID(logicalCardID,RoutingMode,crate);
  
      // Convert regions to SC format    
      m_scRouting.RC234toSTRING(logicalCardID,
                                m_nevt,
                                RC[crate],
                                RCof[crate],
                                RCtau[crate],
                                RC[crate+9],
                                RCof[crate+9],
                                RCtau[crate+9],
                                dataString);
    
      // Write regions
      m_file << dataString;
    }
    
  }
  
  // Force write to file 
  m_file << flush;

  m_nevt++;
}

