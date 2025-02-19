// -*- C++ -*-
//
// Package:    GctFibreAnalyzer
// Class:      GctFibreAnalyzer
// 
/**\class GctFibreAnalyzer GctFibreAnalyzer.cc L1Trigger/L1GctAnalzyer/src/GctFibreAnalyzer.cc

Description: Analyzer individual fibre channels from the source card.

*/
//
// Original Author:  Alex Tapper
//         Created:  Thu Jul 12 14:21:06 CEST 2007
// $Id: GctFibreAnalyzer.cc,v 1.17 2011/01/19 07:32:18 elmer Exp $
//
//

#include "FWCore/MessageLogger/interface/MessageLogger.h" // Logger
#include "FWCore/Utilities/interface/Exception.h" // Exceptions

// Include file
#include "L1Trigger/L1GctAnalyzer/interface/GctFibreAnalyzer.h"

// Data format
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

GctFibreAnalyzer::GctFibreAnalyzer(const edm::ParameterSet& iConfig):
  m_fibreSource(iConfig.getUntrackedParameter<edm::InputTag>("FibreSource")),
  m_doLogicalID(iConfig.getUntrackedParameter<bool>("doLogicalID")),
  m_doCounter(iConfig.getUntrackedParameter<bool>("doCounter")),
  m_numZeroEvents(0),
  m_numInconsistentPayloadEvents(0),
  m_numConsistentEvents(0)
{
}

GctFibreAnalyzer::~GctFibreAnalyzer()
{
  edm::LogInfo("Zero Fibreword events") << "Total number of events with zero fibrewords: " << m_numZeroEvents;
  edm::LogInfo("Inconsistent Payload events") << "Total number of events with inconsistent payloads: " << m_numInconsistentPayloadEvents;
  if(m_doCounter){edm::LogInfo("Successful events") << "Total number of Successful events: " << m_numConsistentEvents;}
}

void GctFibreAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  Handle<L1GctFibreCollection> fibre;
  iEvent.getByLabel(m_fibreSource,fibre);

  bool bc0= false;
  int flag_for_zeroes = 0;
  int flag_for_inconsistent_events = 0;
  unsigned int flag_for_consistency = 0;
  int flag_for_consistent_events = 0;

  for (L1GctFibreCollection::const_iterator f=fibre->begin(); f!=fibre->end(); f++){
	
    if (f->data()!=0)
      {
        if(m_doCounter) 
          {
            if(f==fibre->begin()) {flag_for_consistency = f->data();}
            else if(flag_for_consistency == f->data()){flag_for_consistent_events++;}
          }

        // Check for corrupt fibre data
        if (!CheckFibreWord(*f)){
          edm::LogInfo("GCT fibre data error") << "Missing phase bit (clock) in fibre data " << (*f);
        }
    
        // Check for BC0
        if (CheckForBC0(*f) && (f==fibre->begin())) {
          bc0=true;
        }

        // Check for mismatch between fibres
        if ((bc0 && !CheckForBC0(*f)) ||
            (!bc0 && CheckForBC0(*f))){
          edm::LogInfo("GCT fibre data error") << "BC0 mismatch in fibre data " << (*f);
        }

        // Check logical ID pattern
        if (m_doLogicalID) CheckLogicalID(*f);

        // Check counter pattern
        if (m_doCounter) CheckCounter(*f);

        flag_for_zeroes = 1;	
      }
    else 
      {
	//this basically checks that all the data is 0s by the time it gets to the last iteration
        if(flag_for_zeroes == 0 && f==(fibre->end()-1)) {m_numZeroEvents++;}
	//if the zero flag is set to 1 and it managed to find its way into here then something is wrong!
        if(flag_for_zeroes == 1) {flag_for_inconsistent_events++;}
      }

  }
  //check for inconsistent events i.e. those with one(or more) zeroes, and the rest data
  if(flag_for_inconsistent_events != 0) {m_numInconsistentPayloadEvents++;}
  //check for consistent events with the counter
  if(m_doCounter && flag_for_consistent_events != 0) {m_numConsistentEvents++;}
}

bool GctFibreAnalyzer::CheckForBC0(const L1GctFibreWord fibre)
{
  // Check for BC0 on this event
  if (fibre.data() & 0x8000){
    return true;
  } else {
    return false;
  }
}

bool GctFibreAnalyzer::CheckFibreWord(const L1GctFibreWord fibre)
{
  // Check that the phase or clock bit (MSB bit on cycle 1) is set as it should be
  if (fibre.data() & 0x80000000){
    return true;
  } else {
    return false;
  }
}

void GctFibreAnalyzer::CheckCounter(const L1GctFibreWord fibre)
{
  // Remove MSB from both cycles
  int cycle0Data, cycle1Data;
  
  cycle0Data = fibre.data() & 0x7FFF;
  cycle1Data = (fibre.data() >> 16) & 0x7FFF;

  // Check to see if fibre numbers are consistent
  if ((cycle0Data+1)!=cycle1Data){
    edm::LogInfo("GCT fibre data error") << "Fibre data not incrementing in cycles 0 and 1 "
                                         << " Cycle 0 data=" << cycle0Data
                                         << " Cycle 1 data=" << cycle1Data
                                         << " " << fibre;      
  }  
  
  // For now just write out the data
  edm::LogInfo("GCT fibre counter data") << " Fibre data: cycle0=" << cycle0Data 
                                         << " cycle1=" << cycle1Data
                                         << " " << fibre;
}

void GctFibreAnalyzer::CheckLogicalID(const L1GctFibreWord fibre)
{
  //added by Jad Marrouche, May 08

  unsigned ref_jf_link[] = {1,2,3,4,1,2,3,4};
  //this array lists the link number ordering we expect from the 3 JFs in positive eta
  //below, we modify indices 2 and 3 from 3,4 to 1,2 to represent negative eta

  unsigned ref_eta0_link[] = {3,4,3,4,3,4};
  //this array lists the link number ordering we expect from the ETA0

  unsigned ref_jf_type[] = {2,2,3,3,1,1,1,1};
  //this array lists the SC_type ordering we expect for the JFs

  unsigned ref_eta0_type[] = {2,2,2,2,2,2};
  //this array lists the SC_type ordering we expect for the ETA0 (for consistency)

  int eta_region, rct_phi_region, leaf_phi_region, jf_type, elec_type, local_source_card_id, source_card_id_read, source_card_id_expected;

  // Check that data in cycle 0 and cycle 1 are equal
  if ((fibre.data()&0x7FFF)!=((fibre.data()&0x7FFF0000)>>16)){
    edm::LogInfo("GCT fibre data error") << "Fibre data different on cycles 0 and 1 " << fibre;
  }

  //fibre.block() gives 0x90c etc
  //fibre.index() runs from 0 to 6/8

  if((fibre.block() >> 10) & 0x1 ) 
    {
      eta_region = 0;		//negative eta region 		
      ref_jf_link[2] = 1; //modify indices to represent neg_eta fibre mapping
      ref_jf_link[3] = 2;	
    } 	
  else eta_region = 1;	//positive eta region 

  if(((fibre.block() >> 8) & 0x7) == 0 || ((fibre.block() >> 8) & 0x7) == 4)	//i.e. electron leaf cards
    {
	
      if((fibre.block() & 0xFF)==0x04)		elec_type=1;
      else if((fibre.block() & 0xFF)==0x84)	elec_type=0;
      else throw cms::Exception("Unknown GCT fibre data block ") << fibre.block(); //else something screwed up   
	
      rct_phi_region = (fibre.index() / 3) + (4*elec_type);

      local_source_card_id = (4*eta_region);
	
      source_card_id_expected = (8 * rct_phi_region) + local_source_card_id;
	
      source_card_id_read = (fibre.data() >> 8) & 0x7F;
		
      if(source_card_id_expected != source_card_id_read ) 
        {
          edm::LogInfo("GCT fibre data error") << "Electron Source Card IDs do not match "  
                                               << "Expected ID = " << source_card_id_expected
                                               << " ID read from data = " << source_card_id_read
                                               << " " << fibre; //screwed up
        }

      if( (fibre.data() & 0xFF) != (unsigned int)(2 + fibre.index()%3))
        {
          edm::LogInfo("GCT fibre data error") << "Electron Fibres do not match "  
                                               << "Expected Fibre = " << (2 + fibre.index()%3)
                                               << " Fibre read from data = " << (fibre.data() & 0xFF)
                                               << " " << fibre; //screwed up
        }


    }
  else	//i.e. jet leaf cards
    {
      //the reason we use these values for eta_region is so it is easy to add 4 to the local source card ID
      //remember that 0x9.. 0xA.. and 0xB.. are +ve eta block headers
      //whereas 0xD.., 0xE.. and 0xF.. are -ve eta
      //can distinguish between them using the above mask and shift

      if((fibre.block() & 0xFF)==0x04)		jf_type=1;	//JF2
      else if((fibre.block() & 0xFF)==0x0C)	jf_type=2;	//JF3
      else if((fibre.block() & 0xFF)==0x84)	jf_type=-1;	//ETA0
      else if((fibre.block() & 0xFF)==0x8C)	jf_type=0;	//JF1
      else throw cms::Exception("Unknown GCT fibre data block ") << fibre.block(); //else something screwed up   

      //edm::LogInfo("JF Type Info") << "JF TYPE = " << jf_type << " block = " << fibre;	//INFO ONLY

      leaf_phi_region = ((fibre.block() >> 8) & 0x7)-1;		//0,1,2,3,4,5 for leaf cards
      if(eta_region == 0) leaf_phi_region--;		//need to do this because block index goes 9.. A.. B.. D.. E.. F.. - 8 and C are reserved for electron leafs which are dealt with above
      if(leaf_phi_region <0 || leaf_phi_region >5) throw cms::Exception("Unknown Leaf Card ") << fibre.block(); 
      //throw exception if number is outside 0-5 which means something screwed up
      //if(leaf_phi_region <0 || leaf_phi_region >5) edm::LogInfo("GCT fibre data error") << "Unknown leaf card " << fibre;
      //write to logger if number is outside 0-5 which means something screwed up

      if(jf_type == -1)
        {
          //in this case fibre.index() runs from 0-5
          //JF1 comes first, followed by JF2 and JF3
	
          if(fibre.index() <=5 )	//the compiler warning is because fibre.index() is unsigned int and hence is always >=0
            {
              rct_phi_region = ( (8 + ((leaf_phi_region%3)*3) + (fibre.index() / 2) ) % 9);
              //fibre.index()/2 will give 0 for 0,1 1 for 2,3 and 2 for 4,5
		
              //local_source_card_id = ref_eta0_type[ fibre.index() ] + (4 * (1 % eta_region));
              //take the ones complement of the eta_region because this is the shared part (i.e. other eta0 region)
              //this is done by (1 % eta_region) since 1%0 = 1 and 1%1=0
			  //FLAWED - since 1%0 is a floating point exception you idiot!

			  local_source_card_id = ref_eta0_type[ fibre.index() ] + (4 + (eta_region * -4));
			  //this gives what you want - adds 4 when eta_region = 0 (neg) and adds 0 when eta_region = 1 (pos)
	
              source_card_id_expected = (8 * rct_phi_region) + local_source_card_id;
              //from GCT_refdoc_v2_2.pdf
		
              source_card_id_read = (fibre.data() >> 8) & 0x7F;
		
              if(source_card_id_expected != source_card_id_read ) 
                {
                  edm::LogInfo("GCT fibre data error") << "ETA0 Source Card IDs do not match "  
                                                       << "Expected ID = " << source_card_id_expected
                                                       << " ID read from data = " << source_card_id_read
                                                       << " " << fibre; //screwed up
                }

              if( (fibre.data() & 0xFF) != ref_eta0_link[fibre.index()])
                {
                  edm::LogInfo("GCT fibre data error") << "ETA0 Fibres do not match "  
                                                       << "Expected Fibre = " << ref_eta0_link[fibre.index()]
                                                       << " Fibre read from data = " << (fibre.data() & 0xFF)
                                                       << " " << fibre; //screwed up
                }
            }
          else edm::LogInfo("GCT fibre data error") << "ETA0 Fibre index out of bounds " << fibre;
          //edm::LogInfo("Fibre Index Info") << "ETA0 Fibre index = " << fibre.index();
        }


      if(jf_type >=0) 
        {
          if(fibre.index() <=7 )
            {
              rct_phi_region = ( (8 + ((leaf_phi_region%3)*3) + jf_type ) % 9);		//see table below

              /*
		Leaf Card	|	RCT crate	|	Jet Finder
		___________________________________________
		LC3	|	LC0	|	17	|	8	|	JF1
                |		|	9	|	0	|	JF2
                |		|	10	|	1	|	JF3
		___________________________________________
		LC4	|	LC1	|	11	|	2	|	JF1
                |		|	12	|	3	|	JF2
                |		|	13	|	4	|	JF3
		___________________________________________
		LC5	|	LC2	|	14	|	5	|	JF1
                |		|	15	|	6	|	JF2
                |		|	16	|	7	|	JF3
		___________________________________________
		The phase results in the 17/8 being at the top
		This can be adjusted as necessary by changing
		the number 8 added before modulo 9 operation
              */

              local_source_card_id = ref_jf_type[ fibre.index() ] + (4 * eta_region);

              //since the SC sharing scheme renumbers SC 7 as SC3:
              if(local_source_card_id == 7) local_source_card_id = 3;
              //there is probably a more elegant way to do this

              source_card_id_expected = (8 * rct_phi_region) + local_source_card_id;

              source_card_id_read = (fibre.data() >> 8) & 0x7F;

              if(source_card_id_expected != source_card_id_read ) 
                {
                  edm::LogInfo("GCT fibre data error") << "Source Card IDs do not match "  
                                                       << "Expected ID = " << source_card_id_expected
                                                       << " ID read from data = " << source_card_id_read
                                                       << " " << fibre; //screwed up
                }

              if( (fibre.data() & 0xFF) != ref_jf_link[fibre.index()])
                {
                  edm::LogInfo("GCT fibre data error") << "Fibres do not match "  
                                                       << "Expected Fibre = " << ref_jf_link[fibre.index()]
                                                       << " Fibre read from data = " << (fibre.data() & 0xFF)
                                                       << " " << fibre; //screwed up
                }
            }
          else edm::LogInfo("GCT fibre data error") << "Fibre index out of bounds " << fibre;
        }

    }


}
