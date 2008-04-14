// -*- C++ -*-
//
// Package:    RPCVHDLConeMaker
// Class:      RPCVHDLConeMaker
// 
/**\class RPCVHDLConeMaker RPCVHDLConeMaker.cc src/RPCVHDLConeMaker/src/RPCVHDLConeMaker.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Artur Kalinowski
//         Created:  Tue Aug  1 13:54:56 CEST 2006
// $Id: RPCVHDLConeMaker.cc,v 1.2 2007/06/08 08:43:59 fruboes Exp $
//
//


// system include files
#include <memory>
#include <fstream>
#include <iomanip>
#include <ctime>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// class decleration
//
#include "L1Trigger/RPCTrigger/interface/RPCVHDLConeMaker.h"
#include "L1Trigger/RPCTrigger/interface/RPCRingFromRolls.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
//#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"

#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

//
// constructors and destructor
//
RPCVHDLConeMaker::RPCVHDLConeMaker(const edm::ParameterSet& iConfig){

  ::putenv("CORAL_AUTH_USER konec");
  ::putenv("CORAL_AUTH_PASSWORD konecPass");

  RPCLinksDone  = false;

  minTower = iConfig.getParameter<int>("minTower");
  maxTower = iConfig.getParameter<int>("maxTower");

  minSector = iConfig.getParameter<int>("minSector");
  maxSector = iConfig.getParameter<int>("maxSector");

  patternsPath = iConfig.getParameter<std::string>("patternsPath"); 
  conesPath = iConfig.getParameter<std::string>("conesPath");

}


RPCVHDLConeMaker::~RPCVHDLConeMaker(){}


//
// member functions
//


void RPCVHDLConeMaker::initRPCLinks(const edm::EventSetup& iSetup){

  using namespace edm;
  using namespace std;

  if(!RPCLinksDone) RPCLinksDone  = true;
  else return;
 //Open the cabling database
//  edm::ESHandle<RPCReadOutMapping> map;
//  iSetup.get<RPCReadOutMappingRcd>().get(map);


   edm::ESHandle<RPCEMap> nmap;
   iSetup.get<RPCEMapRcd>().get(nmap);
   const RPCEMap* eMap=nmap.product();
   edm::ESHandle<RPCReadOutMapping>  map = eMap->convert();

  LogInfo("") << "version: " << map->version() << endl;

 // Build the trigger linksystem geometry;
  if (!theLinksystem.isGeometryBuilt()){
    edm::LogInfo("RPC") << "Building RPC links map for a RPCTrigger";
    edm::ESHandle<RPCGeometry> rpcGeom;
    iSetup.get<MuonGeometryRecord>().get( rpcGeom );     
    theLinksystem.buildGeometry(rpcGeom);
    edm::LogInfo("RPC") << "RPC links map for a RPCTrigger built";
  } 
 aLinks=theLinksystem.getLinks();
}

// ------------ method called to produce the data  ------------
void
RPCVHDLConeMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

   using namespace edm;
   using namespace std;

   //initRPCLinks(iSetup);

    writeLogCones(minTower,maxTower,minSector,maxSector,iSetup);

   //writeLogConesForTB(12,iSetup);
   return;
}

void RPCVHDLConeMaker::writeLogCones(int towMin, int towMax, int secMin, int secMax,
				     const edm::EventSetup& iSetup){

  using namespace edm;
  using namespace std;


  for(int iTow=towMin;iTow<=towMax;iTow++){
    for(int iSec=secMin;iSec<=secMax;iSec++){


      string fileName=conesPath + "/pac_t";
      char text[100];
      sprintf(text,"%d",iTow);
      fileName.append(text);
      sprintf(text,"_sec%d",iSec);
      fileName.append(text);
      fileName+=".vhd"; 
      string fileName1=patternsPath;
      //string fileName1="patterns/testPatterns/pacPat_";
      //sprintf(text,"t%d.vhd",iTow);
      sprintf(text,"/t%dsc0.vhdl",abs(iTow));
      fileName1.append(text);
      std::ofstream out(fileName.c_str());
      writeHeader(iTow,iSec,out);
      writeConesDef(iTow,iSec,out,iSetup);
      //writeQualityDef(out);  
      //writePatternsDef(out);
      out.close();
      //      
      string command = "cat "+ fileName +  " " + fileName1;
      command+=" > tmp.out";
      system(command.c_str());
      command = "mv tmp.out ";
      command+=fileName;
      system(command.c_str());
      //
            
      std::ofstream out1(fileName.c_str(),ios::app);
      out1<<std::endl<<"end RPC_PAC_patt;"<<std::endl;
      //writeSorterDef(out1);
      out1.close();
      
    }
  }
}

void RPCVHDLConeMaker::writeHeader(int aTower, int aSector, std::ofstream & out){

  using namespace edm;
  using namespace std;

  
  int maxPAC = 12;
  //int maxPAC = 2;

  bool begin = true;

  //Get current time
  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  asctime (timeinfo);
  //
  out<<"-- version "<<asctime(timeinfo);
  out<<"-- file generated by RPCVHDLConeMaker"<<endl;
  out<<"library ieee;"<<endl;
  out<<"use ieee.std_logic_1164.all;"<<endl;
  out<<"use work.std_logic_1164_ktp.all;"<<endl;
  out<<"use work.RPC_PAC_def.all;"<<endl;
  out<<""<<endl;
  out<<"package RPC_PAC_patt is"<<endl;
  out<<""<<endl;
  out<<"constant TT_EPACS_COUNT         :natural := ";
  //out<<0;
  out<<maxPAC;
  out<<";"<<endl;
  out<<"constant TT_TPACS_COUNT         :natural := ";
  out<<0;
  //out<<maxPAC;
  out<<";"<<endl;
  out<<"constant TT_REF_GROUP_NUMBERS   :natural := 1;"<<endl;
  out<<"constant TT_GBSORT_INPUTS       :natural := "
     <<maxPAC<<";"<<endl;
  out<<""<<endl;
  out<<"constant PACLogPlainsDecl       :TPACLogPlainsDecl := ("<<endl;
  out<<"  --PAC_INDEX"<<endl;
  out<<"  --|   PAC_MODEL"<<endl;
  out<<"  --|   |      logplane 1 size .........logplane 6 size"<<endl;
  //
    for(int iPAC=0;iPAC<maxPAC;iPAC++){
        int size[6];
        for (int i =0; i <6; ++i){
          size[i]=RPCRingFromRolls::m_LOGPLANE_SIZE[abs(aTower)][i];
          if (size[i]==0) size[i]=1;
        } 
	if(!begin) out<<",";
	else begin = false;
	out<<"   ("
	   <<iPAC
	   <<",  E, (  "
	   <<size[0]<<", "
	   <<size[1]<<", "
	   <<size[2]<<", "
	   <<size[3]<<", "
	   <<size[4]<<", "
	   <<size[5]
	   <<"))";		     
	out<<"-- "<<endl;
    }
//
    out<<");"<<endl<<endl;
  out<<"--PAC_INDEX PAC_MODEL Tower LogSector LogSegment"<<endl;
    for(int iPAC=0;iPAC<maxPAC;iPAC++){
       out<<"--    "
	 <<iPAC
	 <<"        E"<<"\t"
	 <<aTower<<"\t"
	 <<aSector<<"\t"
	 <<iPAC<<"\t"<<endl;
    }

  out<<""<<endl;
  out<<""<<endl;
  out<<"constant LogPlainConn           :TLogPlainConn := ("<<endl;
  out<<"  --PAC_INDEX   Logplane        LinkChannel     LinkLeftBit"<<endl;
  out<<"  --| PAC_MODEL |       Link    |       LogPlaneLeftBit"<<endl;
  out<<"  --|      |    |       |       |       |       |       LinkBitsCount"<<endl;
  out<<"  --------------------------------------------------------------"<<endl;


}


void RPCVHDLConeMaker::writeConesDef(int iTower, int iSec, std::ofstream & out, const edm::EventSetup& iSetup){


  int dccInputChannel = getDCCNumber(iTower,iSec);

  using namespace edm;
  using namespace std;

 
  int minPAC = iSec*12;
  int maxPAC = minPAC+11;
  //int maxPAC = minPAC+1;

  /*
  //Open the cabling database
  edm::ESHandle<RPCReadOutMapping> map;
  iSetup.get<RPCReadOutMappingRcd>().get(map);
  //LogInfo("") << "version: " << map->version() << endl;
  */
   edm::ESHandle<RPCEMap> nmap;
   iSetup.get<RPCEMapRcd>().get(nmap);
   const RPCEMap* eMap=nmap.product();
   edm::ESHandle<RPCReadOutMapping>  map = eMap->convert();


  // Build the trigger linksystem geometry;
  if (!theLinksystem.isGeometryBuilt()){
    edm::LogInfo("RPC") << "Building RPC links map for a RPCTrigger";
    edm::ESHandle<RPCGeometry> rpcGeom;
    iSetup.get<MuonGeometryRecord>().get( rpcGeom );     
    theLinksystem.buildGeometry(rpcGeom);
    edm::LogInfo("RPC") << "RPC links map for a RPCTrigger built";
  } 
  RPCRingFromRolls::RPCLinks aLinks=theLinksystem.getLinks();
  
  bool beg = true;

  for(int iCone=minPAC;iCone<=maxPAC;iCone++){
    for(int iPlane=1;iPlane<7;iPlane++){
      RPCRingFromRolls::RPCLinks::const_iterator CI= aLinks.begin();
      for(;CI!=aLinks.end();CI++){
	RPCRingFromRolls::stripCords aCoords = CI->first;
	RPCRingFromRolls::RPCConnectionsVec aConnVec = CI->second;
	RPCRingFromRolls::RPCConnectionsVec::const_iterator aConnCI = aConnVec.begin();
	RPCDetId aId(aCoords.m_detRawId);
	for(;aConnCI!=aConnVec.end();aConnCI++){
	  if(aConnCI->m_tower==iTower && 
	     aConnCI->m_PAC==iCone &&
	     aConnCI->m_logplane==iPlane){
	    ////////////////////
	    LinkBoardElectronicIndex a;
	    std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> linkStrip = 
	      std::make_pair(a, LinkBoardPackedStrip(0,0));
            std::pair<int,int> stripInDetUnit(aCoords.m_detRawId, aCoords.m_stripNo);
	    std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> > aVec = map->rawDataFrame( stripInDetUnit);
	    std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> >::const_iterator CI;
            if( aCoords.m_stripNo <0) continue;
            if( aCoords.m_isVirtual) continue;
	    for(CI=aVec.begin();CI!=aVec.end();CI++){
	      if(CI->first.dccInputChannelNum==dccInputChannel){
	      linkStrip = *CI;
	      //break;
	      }
	    }
	    if(linkStrip.second.packedStrip()==-17) { 
                    cout<<" Problem: "<<aCoords.m_detRawId<<" "<<aCoords.m_stripNo<<" "<<RPCDetId(aCoords.m_detRawId)<<endl; continue;}	  	   
	    else{	    
	      if(iPlane==33)
		LogInfo("")<<"("<<iCone<<",\t E, \t"<<iPlane<<"\t"
		    <<linkStrip.first.tbLinkInputNum<<"\t"
		    <<linkStrip.first.lbNumInLink<<"\t"
		    <<aConnCI->m_posInCone<<"\t "
		    <<linkStrip.second.packedStrip()<<" \t"
		    <<1<<") --"<<aId<<endl;	      	      	      
	      if(!beg)out<<",";
	      else beg = false;
	      out<<"("<<iCone-minPAC<<",\t E, \t"<<iPlane-1<<",\t"
		 <<linkStrip.first.tbLinkInputNum<<",\t"
		 <<linkStrip.first.lbNumInLink<<",\t"
		 <<aConnCI->m_posInCone<<",\t "
		 <<linkStrip.second.packedStrip()<<", \t";
	      out<<1<<") --"<<aId<<endl;	      	      	      
	    }
	  }
	}
      }
    }
  }
  out<<");"<<endl;
}

void RPCVHDLConeMaker::writeQualityDef(std::ofstream & out){

  using namespace std;

  out<<endl;
  
  out<<"constant PACCellQuality :TPACCellQuality := ("<<endl;
  //out<<"(0,\"010000\",7),"<<endl;
  //out<<"(0,\"001111\",0)"<<endl;
  out<<"(0,\"111111\",3),"<<endl;
  out<<"(0,\"111110\",2),"<<endl;
  out<<"(0,\"111101\",2),"<<endl;
  out<<"(0,\"111011\",2),"<<endl;
  out<<"(0,\"110111\",2),"<<endl;
  out<<"(0,\"101111\",2),"<<endl;
  out<<"(0,\"011111\",2),"<<endl;
  out<<"(0,\"111100\",1),"<<endl;
  out<<"(0,\"111010\",1),"<<endl;
  out<<"(0,\"110110\",1),"<<endl;
  out<<"(0,\"101110\",1),"<<endl;
  out<<"(0,\"011110\",1),"<<endl;
  out<<"(0,\"111001\",1),"<<endl;
  out<<"(0,\"110101\",1),"<<endl;
  out<<"(0,\"101101\",1),"<<endl;
  out<<"(0,\"011101\",1),"<<endl;
  out<<"(0,\"110011\",1),"<<endl;
  out<<"(0,\"101011\",1),"<<endl;
  out<<"(0,\"011011\",1),"<<endl;
  out<<"(0,\"100111\",1),"<<endl;
  out<<"(0,\"010111\",1),"<<endl;
  out<<"(0,\"001111\",1)"<<endl;
  out<<");"<<endl<<endl;
  
}


void RPCVHDLConeMaker::writePatternsDef(std::ofstream & out){

  using namespace edm;
  using namespace std;


  ///
  int aTower = 4;
  int minPAC = 0;
  int maxPAC = 1;
  ///

  int iTower = aTower;

  out<<"constant PACPattTable		:TPACPattTable := ("<<endl;
  out<<"-- PAC_INDEX"<<endl;
  out<<"-- | PAC_MODEL"<<endl;
  out<<"-- | |   Ref Group Index"<<endl;
  out<<"-- | |   |  Qualit Tab index"<<endl;
  out<<"-- | |   |  |  Plane1    Plane2   Plane3    Plane4     Plane5     Plane6  sign code  pat number"<<endl;
  for(int iPAC=minPAC;iPAC<=maxPAC;iPAC++){
    for(int i=0;i<8;i++){
      out<<"("<<iPAC<<", T, 0, 0, (";
      for(int iLogPlane=0;iLogPlane<6;iLogPlane++){
	int strip= (RPCRingFromRolls::m_LOGPLANE_SIZE[iTower][iLogPlane]-8)/2+i;
	out<<"( "<<setw(2)<<strip<<", "<<strip<<")";
	if(iLogPlane<5) out<<", ";
      }
      out<<"),";
      out<<"  1,  "<<i+1<<")";
      if(i!=7 || iPAC<maxPAC-minPAC) out<<", ";
      out<<"--0"<<endl;
    }
  }
  out<<");"<<endl<<endl;
}


void RPCVHDLConeMaker::writeXMLPatternsDef(std::ofstream & out){

  using namespace edm;
  using namespace std;


  ///
  int aTower = 0;
  int minPAC = 0;
  int maxPAC = 0;
  ///

  int iTower = aTower;

  //Get current time
  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  asctime (timeinfo);
  //
  out<<"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>"<<endl;
  out<<"<?xml-stylesheet type=\"text/xsl\" href=\"default.xsl\"?>"<<endl;
  out<<"<pacdef>"<<endl;
  out<<"<date>"<<asctime(timeinfo);
  out<<"</date>"<<endl;
  out<<""<<endl;
  out<<"<descr>-- EfficiencyCut 0.9"<<endl;
  out<<"-- Simple patterns for the MTCC tests."<<endl;
  out<<"</descr>"<<endl;
  out<<endl;
  out<<"<qualitTable>"<<endl;
  out<<"<quality id=\"0\" planes=\"001111\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"010111\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"011011\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"011101\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"011110\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"011111\" val=\"2\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"100111\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"101011\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"101101\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"101110\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"101111\" val=\"2\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"110011\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"110101\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"110110\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"110111\" val=\"2\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"111001\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"111010\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"111011\" val=\"2\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"111100\" val=\"1\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"111101\" val=\"2\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"111110\" val=\"2\"/>"<<endl;
  out<<"<quality id=\"0\" planes=\"111111\" val=\"3\"/>"<<endl;
  out<<"<quality id=\"1\" planes=\"000111\" val=\"0\"/>"<<endl;
  out<<"<quality id=\"1\" planes=\"001011\" val=\"0\"/>"<<endl;
  out<<"<quality id=\"1\" planes=\"001101\" val=\"0\"/>"<<endl;
  out<<"<quality id=\"1\" planes=\"001110\" val=\"0\"/>"<<endl;
  out<<"<quality id=\"1\" planes=\"001111\" val=\"1\"/>"<<endl;
  out<<"</qualitTable>"<<endl;
  out<<endl;
  out<<"<pac logSector=\"0\" logSegment=\"0\" tower=\""<<iTower<<"\">"<<endl;

    for(int i=0;i<8;i++){
      out<<"<pat code=\""<<i+1<<"\" grp=\"0\" num=\"0\" qual=\"0\" sign=\"1\" type=\"T\">"<<endl;
      for(int iLogPlane=0;iLogPlane<6;iLogPlane++){
	int strip= (RPCRingFromRolls::m_LOGPLANE_SIZE[iTower][iLogPlane]-8)/2+i;
	out<<"<str Pl=\""<<iLogPlane
	   <<"\" f=\""<<strip
	   <<"\" t=\""<<strip<<"\"/>"<<endl;
      }
      out<<"</pat>"<<endl;
    }
    out<<"</pac>"<<endl;
    out<<endl;
    out<<"</pacdef>"<<endl;
    
}


void RPCVHDLConeMaker::writeSorterDef(std::ofstream & out){

  using namespace edm;
  using namespace std;


  ///
  int aTower = 0;
  int minPAC = 0;
  int maxPAC = 2;
  //int maxPAC = 12;
  ///

  //out<<" "<<endl;
  out<<"\n \n constant GBSortDecl		:TGBSortDecl := ("<<endl;  
  out<<"--PAC_INDEX"<<endl;  
  out<<"--|   PAC_MODEL"<<endl;  
  out<<"--|	 |   GBSORT_INPUT_INDEX"<<endl;  
  for(int i=0;i<(maxPAC-minPAC);i++){
    out<<" ("<<i<<",\t T,\t"<<i<<")";
    if(i<(maxPAC-minPAC)-1) out<<",";
    out<<endl;
    //out<<",(2,   T,  2)"<<endl;  
  }
  out<<");"<<endl;  
  out<<""<<endl;  
  out<<"end RPC_PAC_patt;"<<endl;  
}


int RPCVHDLConeMaker::getDCCNumber(int iTower, int iSec){

  int tbNumber = 0;
  if(iTower<-12) tbNumber = 0;
  else if(-13<iTower && iTower<-8) tbNumber = 1;
  else if(-9<iTower && iTower<-4) tbNumber = 2;
  else if(-5<iTower && iTower<-1) tbNumber = 3;
  else if(-2<iTower && iTower<2) tbNumber = 4;
  else if(1<iTower && iTower<5) tbNumber = 5;
  else if(4<iTower && iTower<9) tbNumber = 6;
  else if(8<iTower && iTower<13) tbNumber = 7;
  else if(12<iTower) tbNumber = 8;

  int phiFactor = iSec%4;
  return (tbNumber + phiFactor*9); //Count DCC input channel from 1
}
