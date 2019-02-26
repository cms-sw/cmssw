// -*- C++ -*-
//
// Package:    WriteVHDL
// Class:      WriteVHDL
// 
/**\class WriteVHDL WriteVHDL.cc L1TriggerConfig/WriteVHDL/src/WriteVHDL.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Tue Mar 18 15:15:30 CET 2008
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"

#include "CondFormats/DataRecord/interface/L1RPCConeBuilderRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"

#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"



#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"


#include <fstream>
#include <bitset>

//
// class decleration
//

class WriteVHDL : public edm::EDAnalyzer {
   public:
      explicit WriteVHDL(const edm::ParameterSet&);
      ~WriteVHDL();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      int getDCCNumber(int iTower, int iSec);
      int getDCC(int iSec);
      int getTBNumber(int iTower);
      int getDCCNumberFromTB(int tbNumber, int iSec);
      int   m_towerBeg;
      int   m_towerEnd;
      int   m_sectorBeg;
      int   m_sectorEnd;
      std::string m_templateName;  
      std::string m_outdirName;  

      struct TBLoc {
        TBLoc(int tb, int sec) : tbNum(tb), sector(sec) {}; 
        int tbNum;
        int sector;
        bool operator< (const TBLoc &  c2  ) const {
              if (this->tbNum != c2.tbNum )  return (this->tbNum < c2.tbNum);
                  else return (this->sector < c2.sector);
        }
      };

      typedef std::map<TBLoc, std::set<int> > TBInputsMap; /// value contains list of used TB inputs, key localizes TB
      TBInputsMap m_tbInputs;

      typedef std::map<TBLoc, std::map<int, int> > TBInputs3to4Map;
      TBInputs3to4Map m_tbInputs3to4;
      
      struct TDetStrip{
        TDetStrip(int d, int s) : detId(d), strip(s) {};
        int detId;
        int strip;
        bool operator< (const TDetStrip &  c2  ) const {
          if (this->detId != c2.detId )  return (this->detId < c2.detId);
          else return (this->strip < c2.strip);
        }
      };
      
      struct TStripConnection{
        TStripConnection(int t, int l, int p) : tbInput(t), lbInTBInput(l), packedStrip(p) {};
        TStripConnection() : tbInput(-1), lbInTBInput(-1), packedStrip(-1) {};
        int tbInput;
        int lbInTBInput;
        int packedStrip;
      };
      
      typedef std::map<TDetStrip,TStripConnection> TStrip2Con;
      typedef std::map<TBLoc, TStrip2Con > TTB2Con;
      TTB2Con m_4thPlaneConnections;
      

      void writePats(const edm::EventSetup& evtSetup,int tower, int logsector);

      std::string writeVersion();
      
      std::string writeCNT(const edm::EventSetup& iSetup, int tower, int logsector,std::string pacT);

      std::string writePACandLPDef(const edm::EventSetup& iSetup, 
                                   int tower, int logsector, std::string PACt);
      
      std::string writeConeDef(const edm::EventSetup& iSetup, 
                               int tower, int sector, std::string PACt);
      
      std::string writeQualTable(const edm::EventSetup& iSetup, int tower, int sector);
      
      std::string writePatterns(const edm::EventSetup& iSetup,
                                int tower, int sector, std::string PACt);
      
      std::string writeGB(std::string PACt);

      void prepareEncdap4thPlaneConnections(edm::ESHandle<RPCGeometry> geom, edm::ESHandle<RPCReadOutMapping> map);
      // ----------member data ---------------------------
};

WriteVHDL::WriteVHDL(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
   m_towerBeg = iConfig.getParameter<int>("minTower");
   m_towerEnd = iConfig.getParameter<int>("maxTower");

   m_sectorBeg = iConfig.getParameter<int>("minSector");
   m_sectorEnd = iConfig.getParameter<int>("maxSector");
   m_templateName = iConfig.getParameter<std::string>("templateName");
   m_outdirName = iConfig.getParameter<std::string>("outDir");


}


WriteVHDL::~WriteVHDL()
{


}


//
// member functions
//

// ------------ method called to for each event  ------------
void
WriteVHDL::beginJob() 
{}


// ------------ method called once each job just before starting event loop  ------------
/*
XXV -- version comment
XXP -- pac/logplanes def
XXC -- cones
XXQ -- quality table
XXS -- Patterns
XXG -- ghostbuster
*/
void 
WriteVHDL::analyze(const edm::Event& iEvent, const edm::EventSetup& evtSetup)
{

  for (int tw = m_towerBeg; tw <= m_towerEnd; ++tw ){
    for (int sec = m_sectorBeg; sec <=m_sectorEnd; ++sec){
      writePats(evtSetup,tw,sec);
    }
  }


}



void 
WriteVHDL::writePats(const edm::EventSetup& evtSetup,int tower, int logsector) {

  std::ifstream inputfile(m_templateName.c_str());
  std::stringstream fname;
  fname << m_outdirName << "/pac_t" << tower << "_sec" << logsector << ".vhd"; 

  std::ofstream fout( fname.str().c_str() );
  
  // get PAC type
  edm::ESHandle<L1RPCConfig> conf;
  evtSetup.get<L1RPCConfigRcd>().get(conf);
  const L1RPCConfig *rpcconf = conf.product();

  RPCPattern::RPCPatVec::const_iterator it =  rpcconf->m_pats.begin();

  while ( it->getTower()!=std::abs(tower) && it!=rpcconf->m_pats.end() ) ++it;

  if (it==rpcconf->m_pats.end())
    throw cms::Exception("") << " tower not found " << tower << "\n";

  RPCPattern::TPatternType patType =  it->getPatternType();

  std::string pacT = "E";
  if (patType==RPCPattern::PAT_TYPE_T ) pacT = "T";


  


  if(inputfile.fail())  {
    throw cms::Exception("IO") << "Cannot open file: " <<m_templateName << "\n";
  }

  char ch, chNext, chCmd;
  while (!inputfile.eof()) {

    inputfile.get(ch);

    if (!inputfile.eof()) { 

       if ( ch == 'X' ) {
          inputfile.get(chNext);
          if (!inputfile.eof()) {
             if (chNext == 'X'){ //input command
                 inputfile.get(chCmd);
                 if (!inputfile.eof()) {
                   switch (chCmd) {
                      case 'V':
                           fout << writeVersion();
                           break;
                      case 'N':
                           fout << writeCNT(evtSetup, tower, logsector, pacT);
                           break;
                      case 'P':
                           fout << writePACandLPDef(evtSetup, tower, logsector, pacT);
                           break;
                      case 'C':
                           fout << writeConeDef(evtSetup, tower, logsector, pacT);
                           break;
                      case 'Q':
                           fout << writeQualTable(evtSetup, tower, logsector);
                           break;
                      case 'S':
                           fout << writePatterns(evtSetup, tower, logsector, pacT);  
                           break;
                      case 'G':
                           fout << writeGB(pacT);
                           break;
                      default:
                           throw cms::Exception("BadTemplate") << " Unknown command: XX" << chCmd << "\n";
                   } 

                 } else {
                   throw cms::Exception("BadTemplate") << " Problem when reading template \n";
                 }

             } else {
               fout << ch << chNext;
             }
          }

       } else {
          fout << ch;
       }


    }

  }

}

std::string WriteVHDL::writeVersion(){

  std::stringstream ret;
  //Get current time
  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  //asctime (timeinfo);
  
  ret << "-- WriteVHDL " << asctime (timeinfo) << std::endl;
  return ret.str();
}

std::string WriteVHDL::writeCNT(const edm::EventSetup& iSetup, int tower, int sector, std::string pacT){
   
   std::stringstream ret;
   int nT=0, nE=0, refGrps = 0;
   if (pacT == "E") {
      nE=12;
   }
   else if (pacT == "T") {
      nT=12;
   }
   else
      throw cms::Exception("") << "Unknown PAC type \n";
   
   ret << "constant TT_EPACS_COUNT         :natural := " << nE << ";" <<  std::endl;
   ret << "constant TT_TPACS_COUNT         :natural := " << nT << ";" <<  std::endl;
   
   // calculate number of RefGruops used
   if (pacT == "E") {
   
      tower = std::abs(tower);

      edm::ESHandle<L1RPCConfig> conf;
      iSetup.get<L1RPCConfigRcd>().get(conf);

      const RPCPattern::RPCPatVec *pats = &conf.product()->m_pats;
      int ppt = conf.product()->getPPT();
      int segment = 0;

      if (ppt == 1 || ppt == 12) {
        sector = 0;
      }


      const RPCPattern::RPCPatVec::const_iterator itEnd = pats->end();
      RPCPattern::RPCPatVec::const_iterator it;

      for ( int iPAC = 0; iPAC < 12 ; ++iPAC){

        if (ppt == 144 || ppt == 12) segment = iPAC;

        for (it = pats->begin(); it!=itEnd; ++it){

           // select right pac
           if ( it->getTower() != tower ||
                it->getLogSector() != sector ||
                it->getLogSegment() != segment ) continue;
   
           if (it->getPatternType() != RPCPattern::PAT_TYPE_E ) {
             throw cms::Exception("WriteVHDL") 
               << "Expected E type pattern, got different one" << std::endl;
           }
           if (refGrps < it->getRefGroup() ) refGrps = it->getRefGroup();
           
         } // patsIter
      } // segment iter
   } // if type E

   ret  << "constant TT_REF_GROUP_NUMBERS   :natural := " << refGrps + 1 << ";" <<  std::endl;

   
   return ret.str();
}


std::string WriteVHDL::writePACandLPDef(const edm::EventSetup& iSetup, int tower, int logsector,  std::string pacT){

  std::stringstream ret; 

  tower = std::abs(tower);

  // get logplane size
  edm::ESHandle<L1RPCConeBuilder> coneBuilder;
  iSetup.get<L1RPCConeBuilderRcd>().get(coneBuilder);
  
  edm::ESHandle<L1RPCConeDefinition> l1RPCConeDefinition;
  iSetup.get<L1RPCConeDefinitionRcd>().get(l1RPCConeDefinition);

  std::string coma = "";
  for (int seg = 0; seg < 12; ++seg  ){
    ret << coma << "   (" << seg
        << ",  "<<  pacT
        << ", (  ";

    std::string coma1 = ""; 
    for (int lp = 0; lp < 6; ++lp){
      //int size = l1RPCConeDefinition->getLPSizes().at(tower).at(lp);
      int lpSize = -1;
      L1RPCConeDefinition::TLPSizeVec::const_iterator it = l1RPCConeDefinition->getLPSizeVec().begin();
      L1RPCConeDefinition::TLPSizeVec::const_iterator itEnd = l1RPCConeDefinition->getLPSizeVec().end();
      for (;it!=itEnd;++it){
            
        if (it->m_tower != std::abs(tower) || it->m_LP != lp) continue;
        lpSize = it->m_size;
            
      }

      //FIXME
      if (lpSize==-1) {
           throw cms::Exception("getLogStrip") << " lpSize==-1\n";
      }
      
     
      
      if (lpSize == 0 || lpSize == -1) lpSize = 1;
      ret << coma1 << lpSize;
      coma1 = ", ";
    }

    ret << "))--" << std::endl;
    coma = ",";
  }

  ret << ");" <<std::endl;
  return ret.str();
}

std::string WriteVHDL::writeQualTable(const edm::EventSetup& iSetup, int tower, int sector){

  std::stringstream ret;
  
  edm::ESHandle<L1RPCConfig> conf;
  iSetup.get<L1RPCConfigRcd>().get(conf);
  
  const  RPCPattern::TQualityVec *qvec = &conf.product()->m_quals;
  
  bool first = true;
  RPCPattern::TQualityVec::const_iterator it = qvec->begin();
  RPCPattern::TQualityVec::const_iterator itEnd = qvec->end();

  //unsigned int ppt = conf.product()->getPPT();
 // if (ppt == 1) {
    sector = 0;
 // }

  int noOfQualitiesWritten = 0; 

  for (;it!=itEnd;++it) {
     
     // there is only one PACCellQuality for 12 comparators!
     if ( it->m_tower != std::abs(tower) || 
          it->m_logsector!=sector || 
          it->m_logsegment!= 0) continue;
           
     if(first) {
        ret<<" (";
        first = false;
     } else {
        ret <<", "<<std::endl<<" (";
     }

     std::bitset<6> fp(it->m_FiredPlanes) ;
     ret  << (int)it->m_QualityTabNumber <<",\""
//           << (int)it->m_FiredPlanes<<"\","
           << fp.to_string<char,std::char_traits<char>, std::allocator<char> >() <<"\","
           << (int)it->m_QualityValue<<")";
     ++noOfQualitiesWritten;
  }
  
  if (noOfQualitiesWritten == 1 && std::abs(tower) == 9) {


  }

  ret<< ");" <<std::endl <<std::endl;
  
  return ret.str();
}

std::string WriteVHDL::writePatterns(const edm::EventSetup& iSetup, 
                                     int tower, int sector, std::string pacT)
{


  std::stringstream ret;
  
  tower = std::abs(tower);
  
  edm::ESHandle<L1RPCConfig> conf;
  iSetup.get<L1RPCConfigRcd>().get(conf);
  
  const RPCPattern::RPCPatVec *pats = &conf.product()->m_pats;
  int ppt = conf.product()->getPPT();
  int segment = 0;
  
  if (ppt == 1 || ppt == 12) {
    sector = 0;
  }


  const RPCPattern::RPCPatVec::const_iterator itEnd = pats->end();
  RPCPattern::RPCPatVec::const_iterator it;
  int to[6], globalPatNo=0;
  bool firstRun = true;
  
  for ( int iPAC = 0; iPAC < 12 ; ++iPAC){
  
    if (ppt == 144 || ppt == 12) segment = iPAC; 
    
    for (it = pats->begin(); it!=itEnd; ++it){
       
       // select right pac
       if ( it->getTower() != tower ||
            it->getLogSector() != sector ||
            it->getLogSegment() != segment ) continue;

       for (int i = 0; i<6 ; ++i){
          to[i]=it->getStripTo(i)-1;
          if (it->getStripFrom(i)==RPCPattern::m_NOT_CONECTED)
             to[i]=RPCPattern::m_NOT_CONECTED;
       }
       
       if (!firstRun) ret << ",";
       firstRun = false;
       
       int sign =  it->getSign();
       
       if (sign == 0) sign = 1;
       else if (sign == 1) sign = 0;
       else throw cms::Exception("BAD sign") << "Bad sign definition: " << sign << std::endl;
       
       ret   << "( " << iPAC << ", " << pacT
             << ", " << it->getRefGroup()
             << ", " << it->getQualityTabNumber()
             << ",(" // planes start
             << "(" << it->getStripFrom(0) << "," << to[0] << ")"  // pl1
             << ",(" << it->getStripFrom(1) << "," << to[1] << ")" // pl2
             << ",(" << it->getStripFrom(2) << "," << to[2] << ")" // pl3
             << ",(" << it->getStripFrom(3) << "," << to[3] << ")" // pl4
             << ",(" << it->getStripFrom(4) << "," << to[4] << ")" // pl5
             << ",(" << it->getStripFrom(5) << "," << to[5] << ")" // pl6
             << ")"  // planes end
             << ", " << sign
             << ", " << it->getCode()
             << ") -- " << globalPatNo++ << std::endl;
       
    } // patterns iteration
  
  } // segment
  
  ret << ");" <<std::endl<< std::endl;


  return ret.str();
}

std::string WriteVHDL::writeGB(std::string PACt){

  std::stringstream ret;
  bool frun=true;
  
  for(int i = 0; i<12; ++i){
     
     if(frun){
        frun=false;
        ret << "(";
     } else{
        ret << std::endl << ",(";
     }
     
     ret << i <<", " << PACt << ", " << i << ")";

  }
  
  ret << ");" << std::endl;

  return ret.str();
}



void WriteVHDL::prepareEncdap4thPlaneConnections(edm::ESHandle<RPCGeometry> rpcGeom, edm::ESHandle<RPCReadOutMapping> map) {

 static bool jobDone = true;
 if (jobDone) return;
 jobDone = true;

 //std::cout << "prepareEncdap4thPlaneConnections\n " ;
 
 
  // build map of used TB inputs
  for(TrackingGeometry::DetContainer::const_iterator it = rpcGeom->dets().begin();
      it != rpcGeom->dets().end();
      ++it)
  {
    if( dynamic_cast<const RPCRoll* >( *it ) == 0 ) continue;
    RPCRoll const* roll = dynamic_cast< RPCRoll const*>( *it );
    int detId = roll->id().rawId();
    
    for (int strip = 1; strip<= roll->nstrips(); ++strip){
      LinkBoardElectronicIndex a = {0,0,0,0};
      std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> linkStrip =
          std::make_pair(a, LinkBoardPackedStrip(0,0));

      std::pair<int,int> stripInDetUnit(detId, strip);
      std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> > aVec = map->rawDataFrame( stripInDetUnit);
      std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> >::const_iterator CI;

      
      for (int iSec = 0; iSec < 12; ++iSec) {
        int DCC = getDCC(iSec);
        for (int iTB = 1; iTB <8 ; ++iTB) {
          int DCCin = getDCCNumberFromTB(iTB, iSec);
          int ncons = 0;
          for(CI=aVec.begin();CI!=aVec.end();++CI){
            if(CI->first.dccInputChannelNum==DCCin &&  CI->first.dccId == DCC ) {
              linkStrip = *CI;
              ++ncons;
            }
          }
          
          if (ncons > 1) std::cout << "Problem: more then one connection for given TB\n";
          if (ncons == 1) {
            TBLoc loc(iTB, iSec);
            int tbInput = linkStrip.first.tbLinkInputNum;
            //int lbInTBInput = linkStrip.first.lbNumInLink;
            m_tbInputs[loc].insert(tbInput);
            /*std::cout << " Found con for TC" << iSec << " TB" << iTB
                << " - " << tbInput
                << std::endl;*/

          }
        } // tb iter
      } // TC iter
    } // strip iter
  } // roll iter

  /*
  for(auto const& t : m_tbInputs )
  {
    std::cout 
        << "TB" << t.first.tbNum
        << " SEC" <<  t.first.sector
        << std::endl
        << "      ";
    for(auto const& input : t.second ){
      std::cout << " " << input; 
    }
    std::cout << std::endl;
  }
  */

 
 for(TrackingGeometry::DetContainer::const_iterator it = rpcGeom->dets().begin();
     it != rpcGeom->dets().end();
     ++it)
 {
   RPCRoll const* roll = dynamic_cast< RPCRoll const*>( *it );
   if( roll == 0 ) continue;
   RPCDetId d = roll->id();
   if ( std::abs(d.region()) !=1) continue;
   if (d.station()!=4 && d.station() != -4) continue;
   int ss = 3;
   if  ( d.station() < 0) ss = -3;

   RPCDetId matching4stDetId(d.region(), d.ring(), ss, d.sector(), d.layer(), d.subsector(), d.roll() );

   //std::cout << d << std::endl;
   int chamberMatches = 0;
   // check if 3d plane matches 4th plane
   for(TrackingGeometry::DetContainer::const_iterator it3 = rpcGeom->dets().begin();
             it3 != rpcGeom->dets().end();
                  ++it3)
   {
     RPCRoll const* roll3 = dynamic_cast< RPCRoll const*>( *it3 );
     if( roll3 == 0 ) continue;
     RPCDetId d3 = roll3->id().rawId();
     if (d3 != matching4stDetId ) continue;
     /*
     if (roll->id().rawId() == 637637354) {
       std::cout << "Match found for:  " <<  roll->id().rawId() 
           << " " <<d3 
           << "\n " << d << std::endl; 
   }*/
     ++chamberMatches;
     //std::cout << "    " << d3 << std::endl;

     if (roll->nstrips() != roll3->nstrips() ) {
       std::cout << d << " Strips differ\n " ;
     } else { // check phi pos
       LocalPoint c = roll->centreOfStrip(roll->nstrips()/2);
       LocalPoint c4 = roll3->centreOfStrip(roll3->nstrips()/2);
       GlobalPoint g = roll->toGlobal(c);
       GlobalPoint g4 = roll3->toGlobal(c4);
       if ( std::abs( g.phi() - g4.phi() ) > 0.01 ) std::cout << d << " " << g.phi() << " " << g4.phi() << std::endl;
     }
     
     
     // Prepare 4th plane cons
     int detId3 = roll3->id().rawId();

     for (int strip = 1; strip<= roll3->nstrips(); ++strip){
       LinkBoardElectronicIndex a = {0,0,0,0};
       std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> linkStrip = std::make_pair(a, LinkBoardPackedStrip(0,0));

       std::pair<int,int> stripInDetUnit(detId3, strip);
       std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> > aVec = map->rawDataFrame( stripInDetUnit);
       std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> >::const_iterator CI;
       
       /*
       if (roll->id().rawId() == 637637354 && strip == 8) {
         std::cout << "Enter\n";
        }*/
       
       
       for(CI=aVec.begin();CI!=aVec.end();++CI){
         

         
         // reverse map a DCC connection to a TB
         int DCCin = CI->first.dccInputChannelNum;
         int DCC =   CI->first.dccId;
         int tb=-1, tc=-1;


         
         int tbInput = -1;
         int lbInTBInput = -1;
         int packedStrip = -1; 
         for (int iSec = 0; iSec < 12; ++iSec){
           
           int matches = 0;
           for (int iTB = 1; iTB < 8; ++iTB){
             if (DCCin == getDCCNumberFromTB(iTB, iSec) ){
               if ( DCC == getDCC(iSec) ){
                 ++matches;
                 tb = iTB;
                 tc = iSec;
                 tbInput = linkStrip.first.tbLinkInputNum;
                 lbInTBInput = linkStrip.first.lbNumInLink;
                 packedStrip = linkStrip.second.packedStrip();
               }
             }
           } // TB iter
           if (matches > 1) {
             std::cout << "Found more than one match!\n";
           }
           
           /*
           if (roll->id().rawId() == 637637354 && strip == 8) {
             std::cout << "Going through 3 cons"
                << " " <<  DCC
                << " " << DCCin
                <<std::endl;
             std::cout << "DCC=" << getDCC(10)
                 << " DCCin=" << getDCCNumberFromTB(7, 10)
                 << std::endl;
             if (matches == 1) {
               std::cout << "got one\n";
             }
          }*/

         
           if (matches != 1) continue;
           TBLoc loc(tb,tc);
         
         // check if coresponding chamber from plane 4 hasnt have tbInput assigned, if not find new one 
           if (m_tbInputs3to4[loc].find(tbInput) == m_tbInputs3to4[loc].end()){
             bool conAdded = false;
           // get first free connection
             for (int con = 0; con < 18; ++con){
               if ( m_tbInputs[loc].find(con) == m_tbInputs[loc].end() ) {
                 m_tbInputs3to4[loc][tbInput]=con;
                 m_tbInputs[loc].insert(con); // input just taken
                 conAdded = true; 
                 break;
               } 
             }
             if (!conAdded) std::cout << "Warning - no more connections were avaliable!!\n";
           }
           
            
           int tbInput4 = m_tbInputs3to4[loc][tbInput];
           m_4thPlaneConnections[loc][TDetStrip(roll->id().rawId(), strip)] 
               = TStripConnection(tbInput4, lbInTBInput, packedStrip);
           /*
           if (roll->id().rawId() == 637637354) {
             std::cout << "Connection added for:  " <<  roll->id().rawId() 
                 << " " << strip << std::endl; 
           }*/
           
           
         } // TC iter
       } // connection vec iter
     } // strip iter
   } // roll iter

   if (chamberMatches!=1)
     std::cout << d <<  "  -> no  of matches is  "  << chamberMatches << std::endl;
      

 }


}

std::string WriteVHDL::writeConeDef(const edm::EventSetup& evtSetup, int tower, int sector,  std::string PACt )
{
    std::stringstream ret;



    edm::ESHandle<L1RPCConeBuilder> coneBuilder;
    evtSetup.get<L1RPCConeBuilderRcd>().get(coneBuilder);

    edm::ESHandle<RPCGeometry> rpcGeom;
    evtSetup.get<MuonGeometryRecord>().get(rpcGeom);



    edm::ESHandle<L1RPCConeDefinition> coneDef;
    evtSetup.get<L1RPCConeDefinitionRcd>().get(coneDef);


    static edm::ESHandle<RPCReadOutMapping>  map;
    static bool isMapValid = false;

    if (!isMapValid){ 
      edm::ESHandle<RPCEMap> nmap;
      evtSetup.get<RPCEMapRcd>().get(nmap);
      const RPCEMap* eMap=nmap.product();
      map = eMap->convert(); //*/
      isMapValid = true;
    }
    
    prepareEncdap4thPlaneConnections(rpcGeom,  map);

   /*
   static edm::ESWatcher<RPCEMapRcd> recordWatcher;
   const RPCReadOutMapping* map = 0;

   if (recordWatcher.check(evtSetup)) {  
    delete map; 
    edm::ESHandle<RPCEMap> readoutMapping;
    evtSetup.get<RPCEMapRcd>().get(readoutMapping);
    map = readoutMapping->convert();
   }*/



    bool beg = true;
    for(TrackingGeometry::DetContainer::const_iterator it = rpcGeom->dets().begin();
      it != rpcGeom->dets().end();
      ++it)
    {
      if( dynamic_cast<const RPCRoll* >( *it ) == 0 ) continue;
      RPCRoll const* roll = dynamic_cast< RPCRoll const*>( *it );
      
      int detId = roll->id().rawId();
      //iterate over strips
      
      for (int strip = 1; strip<= roll->nstrips(); ++strip){

          std::pair<L1RPCConeBuilder::TStripConVec::const_iterator, 
                    L1RPCConeBuilder::TStripConVec::const_iterator> 
                    itPair = coneBuilder->getConVec(detId, strip);

          if (itPair.first!=itPair.second){
              throw cms::Exception("") << " FIXME found uncompressed connection. " << tower << "\n";
          }

          std::pair<L1RPCConeBuilder::TCompressedConVec::const_iterator, L1RPCConeBuilder::TCompressedConVec::const_iterator>
                           compressedConnPair = coneBuilder->getCompConVec(detId);

          L1RPCConeBuilder::TCompressedConVec::const_iterator itComp = compressedConnPair.first;

          for (; itComp!=compressedConnPair.second; ++itComp){
              int logstrip = itComp->getLogStrip(strip,coneDef->getLPSizeVec());
              if (logstrip==-1) continue;


              // iterate over all PACs 
              if (itComp->m_tower != tower) continue;

              int dccInputChannel = getDCCNumber(tower, sector);
              int PACstart = sector*12;
              int PACend = PACstart+11;

              for(int PAC = PACstart; PAC <= PACend; ++PAC){
                   if (itComp->m_PAC != PAC ) continue;
  
                   LinkBoardElectronicIndex a = {0,0,0,0};
                   std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> linkStrip =
                        std::make_pair(a, LinkBoardPackedStrip(0,0));

                   std::pair<int,int> stripInDetUnit(detId, strip);
                   std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> > aVec = map->rawDataFrame( stripInDetUnit);
                   std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> >::const_iterator CI;

                   //bool connectionFound = false;
                   for(CI=aVec.begin();CI!=aVec.end();++CI){

                     if(CI->first.dccInputChannelNum==dccInputChannel) linkStrip = *CI;
                     // connectionFound = true;
                     

                   }
                   // check if it is missing plane 4 in endcap

                   int tbLink = linkStrip.first.tbLinkInputNum;
                   int lbNum = linkStrip.first.lbNumInLink;
                   int packedStrip = linkStrip.second.packedStrip();
                   bool plane4HackWorked = false; 
                   if ( std::abs(roll->id().region()) == 1 && std::abs(roll->id().station())==4 ) {
                     std::cout << "Warning, never should get here!" << std::endl;
                     TBLoc loc( getTBNumber(tower), sector);
                     TDetStrip ds(detId, strip);
                     if ( m_4thPlaneConnections.find(loc) ==m_4thPlaneConnections.end() ) {
                       std::cout << "4thplane problem - unkown TB\n";
                     } else if (m_4thPlaneConnections[loc].find(ds) ==m_4thPlaneConnections[loc].end()){
                       std::cout << "4thplane problem - unkown strip " 
                           << strip
                           << " " << detId
                           << " " <<roll->id() << std::endl;
                       
                     } else {
                       //std::cout << "4thplane fine" << std::endl;
                       tbLink = m_4thPlaneConnections[loc][ds].tbInput;
                       lbNum = m_4thPlaneConnections[loc][ds].lbInTBInput;
                       packedStrip =  m_4thPlaneConnections[loc][ds].packedStrip;
                       plane4HackWorked = true;
                     }
                   } // endcap 4th plane
            
                   
                   //if(!connectionFound) { test me...
                   if(linkStrip.second.packedStrip()==-17 && !plane4HackWorked) {
                       std::cout << "Problem" << std::endl;
                   
                   } else {

                     if(!beg) ret<<","; else beg = false;
                     ret << "(" << PAC - PACstart  << ",\t "<< PACt<<", \t"<<  (int)itComp->m_logplane - 1<<",\t"
                         << tbLink<<",\t"
                         << lbNum <<",\t"
                         << logstrip<<",\t "
                         << packedStrip <<", \t";
                      ret << 1 << ") --" << roll->id() << std::endl;	

                   }

            } // PAC iteration

      } // cone connections interation

    } // strip in roll iteration

  } // roll iteration



  ret << "\n );";

  return ret.str();


}

// ------------ method called once each job just after ending the event loop  ------------
void 
WriteVHDL::endJob() {
}

// returns DCC channel for given tower, sec
int WriteVHDL::getDCCNumber(int iTower, int iSec){
  
  int tbNumber = getTBNumber(iTower);
  return getDCCNumberFromTB(tbNumber, iSec); //Count DCC input channel from 1
  
}

int WriteVHDL::getDCCNumberFromTB(int tbNumber, int iSec){

  int phiFactor = iSec%4;
  return (tbNumber + phiFactor*9); //Count DCC input channel from 1
}


int WriteVHDL::getTBNumber(int iTower){
  
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
  return tbNumber;
  
  
}

int WriteVHDL::getDCC(int iSec){
  if (iSec < 0 || iSec > 12) throw cms::Exception("Problem!!!\n");
  int DCC = 792; // sec 0...3
  if ( iSec > 3 && iSec < 8) {
    DCC = 791;
  } else if ( iSec > 7 ) {
    DCC = 790;
  }

  return DCC;
}

//define this as a plug-in
DEFINE_FWK_MODULE(WriteVHDL);
