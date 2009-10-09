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
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      int getDCCNumber(int iTower, int iSec);
      int   m_towerBeg;
      int   m_towerEnd;
      int   m_sectorBeg;
      int   m_sectorEnd;
      std::string m_templateName;  
      std::string m_outdirName;  


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
WriteVHDL::beginJob(const edm::EventSetup& evtSetup) 
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

      if( dynamic_cast< RPCRoll* >( *it ) == 0 ) continue;

      RPCRoll* roll = dynamic_cast< RPCRoll*>( *it );

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
  
                   LinkBoardElectronicIndex a;
                   std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> linkStrip =
                        std::make_pair(a, LinkBoardPackedStrip(0,0));

                   std::pair<int,int> stripInDetUnit(detId, strip);
                   std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> > aVec = map->rawDataFrame( stripInDetUnit);
                   std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> >::const_iterator CI;

                   for(CI=aVec.begin();CI!=aVec.end();++CI){

                     if(CI->first.dccInputChannelNum==dccInputChannel) linkStrip = *CI;

                   }
            
                   if(linkStrip.second.packedStrip()==-17) {
                       std::cout << "Problem" << std::endl;
                   
                   } else {

                     if(!beg) ret<<","; else beg = false;

	             ret << "(" << PAC - PACstart  << ",\t "<< PACt<<", \t"<<  (int)itComp->m_logplane - 1<<",\t"
                         <<linkStrip.first.tbLinkInputNum<<",\t"
                         <<linkStrip.first.lbNumInLink<<",\t"
                         << logstrip<<",\t "
                         <<linkStrip.second.packedStrip()<<", \t";
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

int WriteVHDL::getDCCNumber(int iTower, int iSec){

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


//define this as a plug-in
DEFINE_FWK_MODULE(WriteVHDL);
