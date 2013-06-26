// -*- C++ -*-
//
// Package:    RPCTriggerConfig
// Class:      RPCTriggerConfig
// 
/**\class RPCTriggerConfig RPCTriggerConfig.h L1TriggerConfig/RPCTriggerConfig/interface/RPCTriggerConfig.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Tue Mar 20 12:30:19 CET 2007
// $Id: RPCTriggerConfig.cc,v 1.6 2009/03/26 12:06:38 fruboes Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/ParameterSet/interface/FileInPath.h>


#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"
#include <string>


#include "L1Trigger/RPCTrigger/interface/RPCPatternsParser.h"
//
// class decleration
//

class RPCTriggerConfig : public edm::ESProducer {
   public:
      RPCTriggerConfig(const edm::ParameterSet&);
      ~RPCTriggerConfig();

      typedef std::auto_ptr<L1RPCConfig> ReturnType;

      ReturnType produce(const L1RPCConfigRcd&);
   private:
      // ----------member data ---------------------------

     int m_ppt;
     std::string m_patternsDir;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RPCTriggerConfig::RPCTriggerConfig(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed


   m_ppt = iConfig.getUntrackedParameter<int>("PACsPerTower");
   std::string dataDir = iConfig.getUntrackedParameter<std::string>("filedir");
   
   edm::FileInPath fp(dataDir+"pacPat_t0sc0sg0.xml");
   std::string patternsDirNameUnstriped = fp.fullPath();
   m_patternsDir = patternsDirNameUnstriped.substr(0,patternsDirNameUnstriped.find_last_of("/")+1);

  
	       

}


RPCTriggerConfig::~RPCTriggerConfig()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
RPCTriggerConfig::ReturnType
RPCTriggerConfig::produce(const L1RPCConfigRcd& iRecord)
{
   using namespace edm::es;
   std::auto_ptr<L1RPCConfig> pL1RPCConfig = std::auto_ptr<L1RPCConfig>( new L1RPCConfig() );

   pL1RPCConfig->setPPT(m_ppt);
   
   // parse and isert patterns
   int scCnt = 0, sgCnt = 0;
   if(m_ppt == 1) {
       scCnt = 1;
       sgCnt = 1;
    }
    else if(m_ppt == 12) {
       scCnt = 1;
       sgCnt = 12;
    }
    else if(m_ppt == 144) {
       scCnt = 12;
       sgCnt = 12;
    }
    else {
       throw cms::Exception("BadConfig") << "Bad number of ppt requested: " << m_ppt << "\n";
    }


    for (int tower = 0; tower < RPCConst::m_TOWER_COUNT; ++tower) {
      for (int logSector = 0; logSector < scCnt; ++logSector) {
         for (int logSegment = 0; logSegment < sgCnt; ++logSegment) {
	 
            std::stringstream fname;
            fname << m_patternsDir
                  << "pacPat_t" << tower 
        	  << "sc"  << logSector 
	          << "sg" <<logSegment 
        	  << ".xml";
		  
	    // TODO: this should go to logSth
	    LogDebug("RPCTriggerConfig") << "Parsing: " << fname.str() <<std::endl;
		  
            RPCPatternsParser parser;
	    parser.parse(fname.str());

	    RPCPattern::RPCPatVec npats = parser.getPatternsVec(tower, logSector, logSegment);
            for (unsigned int ip=0; ip<npats.size(); ip++) {
              npats[ip].setCoords(tower,logSector,logSegment);
              pL1RPCConfig->m_pats.push_back(npats[ip]);
            }

            RPCPattern::TQualityVec nquals = parser.getQualityVec(); 
            for (unsigned int iq=0; iq<nquals.size(); iq++) {
              nquals[iq].m_tower=tower;
              nquals[iq].m_logsector=logSector;
              nquals[iq].m_logsegment=logSegment;
              pL1RPCConfig->m_quals.push_back(nquals[iq]);
            }
	    
	    LogDebug("RPCTriggerConfig") 
	              << "  RPCPatterns: " << npats.size() 
		      << " qualities: "<<  nquals.size()
		      << std::endl;
	    
	 
         } // segments
      } // sectors
    } // towers



   return pL1RPCConfig ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RPCTriggerConfig);
