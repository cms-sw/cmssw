#include "TFile.h"
#include "TTree.h"
#include "IORawData/Ecal2004TBInputService/interface/Ecal2004TBSource.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawCrystal.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawHodo.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawPn.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTower.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawHeader.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTriggerChannel.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawAdc2249.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawScaler.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTdcTriggers.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTdcInfo.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTpgChannel.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawLaserPulse.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawPattern.h"
#include "IORawData/Ecal2004TBInputService/interface/TRunInfo.h"
#include "IORawData/Ecal2004TBInputService/interface/H4Geom.h"

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>
#include <DataFormats/EcalDetId/interface/EcalSubdetector.h>
#include <TBDataFormats/EcalTBObjects/interface/EcalTBCollections.h>

#include "FWCore/PluginManager/interface/PluginCapabilities.h"

#include <iostream>
#include <stdlib.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

// Data structure to relate hodoscopes fiber numbers in the layers with the channel number of the photomultiplier.
struct hodo_fibre_index 
{
  int nfibre;
  int ndet;
};

// nHodoscopes = 2; nFibres = 64 
const static struct hodo_fibre_index hodoFibreMap[2][64] = {
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


Ecal2004TBSource::Ecal2004TBSource(const edm::ParameterSet & pset, edm::InputSourceDescription const& desc) : 
  edm::ExternalInputSource(pset,desc)
{
  m_tree=0;
  m_fileCounter=-1;
  m_file=0;
  m_runInfo=0;
  m_i=0;


  produces<EBDigiCollection>();
  produces<EcalPnDiodeDigiCollection>();         
  produces<EcalTBHodoscopeRawInfo>();
  produces<EcalTBTDCRawInfo>();
  produces<EcalTBEventHeader>();
  produces<EcalTrigPrimDigiCollection>();
}


void Ecal2004TBSource::openFile(const std::string& filename) {
  if (m_file!=0) 
    {
      m_file->Close();
      m_file=0;
      m_tree=0;
      m_runInfo=0;
    }
  
  m_file=TFile::Open(filename.c_str());

  if (m_file==0) 
    {
      edm::LogError("Ecal2004TBSourceError") << "Unable to open the file " << filename ;
      m_tree=0;
      m_runInfo=0;
      return;
    } 

  // run info
  m_runInfo = (TRunInfo*)m_file->Get("TRunInfo");
  if (m_runInfo==0) 
    {
      m_file->Close();
      m_file=0;
      edm::LogError("Ecal2004TBSourceError") << "Unable to find RunInfo" ;
      return;
    }

  // charging the tree: different names for different types of run
  if (m_runInfo->GetRunType() == 1)      { m_tree=(TTree*)m_file->Get("T01"); }   // beam run
  else if (m_runInfo->GetRunType() == 3) { m_tree=(TTree*)m_file->Get("T13"); }   // laser run
  else if (m_runInfo->GetRunType() == 5) { m_tree=(TTree*)m_file->Get("T11"); }   // pedestal run
  else { edm::LogError("Ecal2004TBSourceError") << "Unknown run type: " << m_runInfo->GetRunType() ; return; } 

  if (m_tree==0) 
    {
      m_file->Close();
      m_file=0;
      edm::LogError("Ecal2004TBSourceError") << "Unable to find CMSRAW tree" ;
      return;
    }
  
  edm::LogInfo("Ecal2004TBSourceInfo") << "Opening '" << filename << "' with " << m_tree->GetEntries() << " events\n";
  
  // charging branches
  m_tree->SetMakeClass(1);  
  TObjArray* lb=m_tree->GetListOfBranches();
  n_towers=0;
  n_pns=0;
  n_pnIEs=0;
  
  m_eventHeader=0;

  for (int i=0;i<maxTowers;i++)
    m_towers[i]=0;
  for (int i=0;i<maxPns;i++)
    m_pns[i]=0;       
  for (int i=0;i<maxPnIEs;i++)
    m_pnIEs[i]=0;       

  m_hodo=0;
  m_tdcInfo=0;
  m_tpg=0;
  
  for (int i=0; i<lb->GetSize(); i++) 
    {
      TBranch* b=(TBranch*)lb->At(i);
      if (b==0) continue;
      string branchName(b->GetName());

      if (!strcmp(b->GetClassName(),"TRawHeader")) 
	{
	  if ( b->GetEntries() ) 
	    { 
	      b->SetAddress(&m_eventHeader);
	    } 
	}
      else if (!strcmp(b->GetClassName(),"TRawTower")) 
	{
	  if ( b->GetEntries() ) 
	    { 

	      b->SetAddress(&(m_towers[n_towers]));
	      for(unsigned int j = 0; j < branchName.size(); j++) 
		{
		  if (isdigit(branchName.at(j)))
		    {
		      int branchNumber = atoi(branchName.substr(j,branchName.size()).c_str());
		      towerNumbers[n_towers]=branchNumber;
		      break;
		    }
		}
	      n_towers++;
	    }
	}
      else if ( (!strcmp(b->GetClassName(),"TRawPn")) && (!strncmp(b->GetName(),"BRawPn",6)) && (strncmp(b->GetName(),"BRawPnIE",8)) )
	{
	  if ( b->GetEntries() ) 
	    { 

	      b->SetAddress(&(m_pns[n_pns]));
	      for(unsigned int j = 0; j < branchName.size(); j++) 
		{
		  if (isdigit(branchName.at(j)))
		    {
		      int branchNumber = atoi(branchName.substr(j,branchName.size()).c_str());
		      pnNumbers[n_pns]=branchNumber;
		      break;
		    }
		}
	      n_pns++;
	    }
	}
      else if ( (!strcmp(b->GetClassName(),"TRawPn")) && (!strncmp(b->GetName(),"BRawPnIE",8)) )
	{
	  if ( b->GetEntries() ) 
	    { 

	      b->SetAddress(&(m_pnIEs[n_pnIEs]));
	      for(unsigned int j = 0; j < branchName.size(); j++) 
		{
		  if (isdigit(branchName.at(j)))
		    {
		      int branchNumber = atoi(branchName.substr(j,branchName.size()).c_str());
		      pnIENumbers[n_pnIEs]=branchNumber;
		      break;
		    }
		}
	      n_pnIEs++;
	    }
	}
      else if (!strcmp(b->GetClassName(),"TRawHodo")) 
	{
	  if ( b->GetEntries() ) 
	    {
	      b->SetAddress(&m_hodo);
	    } 
	}
      else if (!strcmp(b->GetClassName(),"TRawTdcInfo")) 
	{
	  if ( b->GetEntries() ) 
	    {

	      b->SetAddress(&m_tdcInfo);
	    } 
	}
      else if (!strcmp(b->GetClassName(),"TRawTpgChannel")) 
	{
	  if ( b->GetEntries() ) 
	    {
	      b->SetAddress(&m_tpg);
	    } 
	}
    }
  edm::LogInfo("Ecal2004TBSourceInfo") << "---------------- Branches charged -----------------" ;
}


void Ecal2004TBSource::setRunAndEventInfo() {

  while (m_tree==0 || m_i==m_tree->GetEntries()) 
    {
      m_fileCounter++;
      if (m_file!=0) 
	{
	  m_file->Close();
	  m_file=0; 
	  m_tree=0;
	  m_runInfo=0;
	}
      if (m_fileCounter>=int(fileNames().size())){ edm::LogError("Ecal2004TBSourceError") << "problem with the file, exit" ;  return; } // nothing good
      openFile(fileNames()[m_fileCounter]);
    }

  if (m_tree==0 || m_i==m_tree->GetEntries()) { edm::LogError("Ecal2004TBSourceError") << "problem with the tree, exit" ; return; } //nothing good

  m_tree->GetEntry(m_i);
  m_i++;


  if ( m_runInfo ){ setRunNumber(m_runInfo->GetRunNum()); }
  // if ( m_eventHeader ){ setEventNumber(m_eventHeader->GetEvtNum()); }     // to solve 
  setEventNumber(m_i);
  if (m_i%100 == 0)
    edm::LogInfo("Ecal2004TBSourceInfo") << m_i << " events read" ;
}


bool Ecal2004TBSource::produce(edm::Event& e) {

  if (m_tree==0) return false;

  // create the collection
  auto_ptr<EBDigiCollection> productEb(new EBDigiCollection);
  auto_ptr<EcalPnDiodeDigiCollection> productEPn(new EcalPnDiodeDigiCollection);
  auto_ptr<EcalTBHodoscopeRawInfo> productHodo(new EcalTBHodoscopeRawInfo());         
  auto_ptr<EcalTBTDCRawInfo> productTdc(new EcalTBTDCRawInfo());                      
  auto_ptr<EcalTBEventHeader> productHeader(new EcalTBEventHeader());                      
  auto_ptr<EcalTrigPrimDigiCollection> productTpg(new EcalTrigPrimDigiCollection());

  // supermodule geometry
  H4Geom geom;

  // towers/crystals data
  for(int tnum=0; tnum < n_towers; tnum++)   
    {
      if (m_towers[tnum])
	{
	  for (int xnum = 0; xnum < maxXtals; xnum++) 
	    {
	      int smXtal  = geom.getSMCrystalNumber(towerNumbers[tnum],xnum);
	      TRawCrystal *myRawCrystal = m_towers[tnum]->GetCrystal(xnum); 	  
	      int SamplesNum = myRawCrystal->GetNSamples();
	      if (SamplesNum != 0)
		{	  
                  productEb->push_back( EBDetId(1, smXtal,EBDetId::SMCRYSTALMODE) );
		  EBDataFrame theFrame( productEb->back() );
		  theFrame.setSize(SamplesNum);
		  
		  for (int sample=0; sample<SamplesNum; sample++)   
		    {
		      int mySample = myRawCrystal->GetSample(sample); 
		      int adcValue = (mySample & 0xFFF);       
		      int gain = (mySample & 0x3000) >> 12;
		      theFrame.setSample(sample, EcalMGPASample(adcValue,gain));		 
		    }
		  
		}
	    }
	} // loop over crystals in tower
    } // loop over towers
  
  


  // pn data
  if ( n_pns < 10 ){ edm::LogError("Ecal2004TBSourceError") << "WARNING, PNs number < 10" ; }
  for(int pnnum=0; pnnum < n_pns; pnnum++)   
    {
      if (m_pns[pnnum])
	{
	  int PnSamplesNum = m_pns[pnnum]->GetNSamples();
	  if ((PnSamplesNum != 50) && (m_runInfo->GetRunType() != 1)){ edm::LogError("Ecal2004TBSourceError") << "WARNING, laser or pedestal run with PnSamples != 50" ; } 
	  
	  if (PnSamplesNum != 0)
	    {
	      EcalPnDiodeDetId PnId(1, 1, pnnum+1);      // 1st: subDetectorId: EB (1) ,EE (2)
	      // 2nd: DCCId relative to SubDetector. In barrel it is the SupermoduleId from 1-36
	      // 3rd: PnId, in barrel from 1-10
	      EcalPnDiodeDigi thePnDigi(PnId);
	      thePnDigi.setSize(PnSamplesNum);
	      
	      for (int pnsample=0; pnsample<PnSamplesNum; pnsample++)   
		{
		  int myPnSample = m_pns[pnnum]->GetSample(pnsample); 
		  thePnDigi.setSample(pnsample, EcalFEMSample((uint16_t)myPnSample));
		}
	      
	      productEPn->push_back(thePnDigi);
	    }
	}
    }
      
      
  
  // hodoscopes information
  if ( (m_runInfo->GetRunType() == 1) && (m_hodo) )   
    {
      int lenght = m_hodo->GetLen();

      if ( lenght <= 0 ) { edm::LogError("Ecal2004TBSourceError") << "WARNING, hodoscope lenght <= 0" ; }
      if ( lenght != hodoRawLen*nHodoPlanes ) { edm::LogError("Ecal2004TBSourceError") << "WARNING, hodoscope raw data corrupted! Len = " << lenght ; }
      
      if ( lenght > 0 && lenght == hodoRawLen*nHodoPlanes ) 
	{ 
	      // Decoding of raw data into integer array hits                                  
	  for (int ipl=0; ipl<nHodoPlanes; ipl++) 
	    {	      
	      int detType = 1;       // new mapping for electronics channels  
	      
	      for (int fib=0; fib<nHodoFibres; fib++) { hodoHits[ipl][fib] = 0; }            
	      
	      int ch=0;
	      for(int j=0; j<hodoRawLen; j++) 
		{
		  int word=m_hodo->GetValue(j+ipl*hodoRawLen)&0xffff;
		  for(int i=1; i<0x10000; i<<=1) 
		    {
		      if ( word & i ) 
			{
			  // map electronics channel to No of fibre
			  hodoHits[ipl][hodoFibreMap[detType][ch].nfibre - 1]++;
			}
		      ch ++;
		    }
		} 
	    }

	  // building the hodo infos (returning decoded hodoscope hits information)
	  productHodo->setPlanes((unsigned int)nHodoPlanes);
	  for (int ipl = 0; ipl < nHodoPlanes; ipl++) 
	    {	      	  
	      EcalTBHodoscopePlaneRawHits theHodoPlane;
	      theHodoPlane.setChannels((unsigned int)nHodoFibres);
	      for (int fib = 0; fib < nHodoFibres; fib++){ theHodoPlane.setHit((unsigned int)fib, (bool)hodoHits[ipl][fib]); }
	      productHodo->setPlane((unsigned int)ipl, theHodoPlane);
	    }
	} // lenght ok
    } // beam run
  



  // tdc information
  if ( m_tdcInfo ) 
    {
      int nTdcSamples = m_tdcInfo->GetNValue();
      productTdc->setSize((unsigned int)nTdcSamples);

      for (int mySample = 0; mySample<nTdcSamples; mySample++)
	{
	  EcalTBTDCSample theTdcSample(1,(unsigned int)(m_tdcInfo->GetValue(mySample)));
	  productTdc->setSample((unsigned int)mySample, theTdcSample);
	}
   } // tdc ok


  
  // tpg information
  if ( m_tpg ) 
    {   
      int nTpgSamples = m_tpg->GetLen(); //20 clock samples per Tower +  2 headers 
      int nTowers = (nTpgSamples-2)/20; //Only 10 TT read in 2004 Data
      for (int itt=0;itt<nTowers;itt++) 
	{
	  //	  LogDebug("Ecal2004TBSourceDebug") << "+++++++++++++++++++++++++" ; 
	  int startingIndex = ( (int)floor((double)itt/10.) + 1)*2 +
	    itt*20;
	  int ieta=itt/H4Geom::kTowersInPhi+1;
	  int iphi=4-(itt%4); //Reverse phi direction w.r.t SM numbering 
	  //	  LogDebug("Ecal2004TBSourceDebug") << "Tower " << ieta << " " << iphi ;

 	  EcalTrigTowerDetId ttId(1,EcalBarrel,ieta,iphi,EcalTrigTowerDetId::SUBDETIJMODE);
 	  EcalTriggerPrimitiveDigi theTpgDigi(ttId);
	  theTpgDigi.setSize(20);
	  for (int iS = 0; iS < 20; iS++)
	    {
	      int sample = m_tpg->GetValue(startingIndex + iS);
	      //	      LogDebug("Ecal2004TBSourceDebug") << "Value " << iS << " " << (int)((double)(((unsigned int)sample & (unsigned int)0x3FF))/(double)(0x3FF)*0xFF)
	      //			<< " FGVB " << ( (sample & 0x400) >> 10 )
	      //			<< " GFB " <<  ( sample & 0x800 ) 
	      //			<< std ::endl;
	      //In 2004 et was encoded using 10 bit. Here we saturate above 0xFF 
	      //	      int et=(int)((double)(((unsigned int)sample & (unsigned int)0x3FF))/(double)(0x3FF)*0xFF);
	      int et = (int) ((unsigned int)sample & (unsigned int)0x3FF);
	      if (et > 0xFF)
		et=0xFF;
	      EcalTriggerPrimitiveSample theTpgSample(et,(sample & 0x400) >> 10,0); 
 	      theTpgDigi.setSample(iS, theTpgSample);
	    }
	  //	  LogDebug("Ecal2004TBSourceDebug") << theTpgDigi ;
	  productTpg->push_back(theTpgDigi);
	  //	  LogDebug("Ecal2004TBSourceDebug") << "+++++++++++++++++++++++++" ; 
	}
    }

  
  // Event Header
  //   if ( m_eventHeader )
  //     {
  //       productHeader->setEventNumber(m_eventHeader->GetEvtNum());
  //       productHeader->setBurstNumber(m_eventHeader->GetBurstNum());
  //       productHeader->setTriggerMask(m_eventHeader->GetTrigMask());
  //       productHeader->setDate(m_eventHeader->GetDate());
  //       productHeader->setCrystalInBeam(EBDetId(1,m_eventHeader->GetXtal(),EBDetId::SMCRYSTALMODE));
  //       productHeader->setThetaTableIndex(m_eventHeader->GetThetaTableIndex());
  //       productHeader->setPhiTableIndex(m_eventHeader->GetPhiTableIndex());
  //       productHeader->setLightIntensity((m_eventHeader->GetLightIntensityIndex() & 0xFFFF));
  //       productHeader->setLaserType(((m_eventHeader->GetLightIntensityIndex() & 0xFFFF0000) >> 16)) ;
  //     } // event header ok	
  
  // putting into the event
  e.put(productEb);
  e.put(productEPn);
  e.put(productHodo);
  e.put(productTdc);
  e.put(productTpg);
  e.put(productHeader);

  return true;

}

