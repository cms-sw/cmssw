#ifndef POPCON_ANALYZERBASE_H
#define POPCON_ANALYZERBASE_H
//
// Original Author:  Marcin BOGUSZ
//         Created:  Tue Jul  3 10:48:22 CEST 2007

// system include files
//#include <memory>



//
// class decleration
//
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/PopCon/interface/OutputServiceWrapper.h"


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/PopCon/interface/StateCreator.h"
//#include "CondCore/PopCon/interface/Logger.h"

#include "CondCore/PopCon/interface/IOVPair.h"

namespace popcon
{
   class PopConAnalyzerBase : public edm::EDAnalyzer {
   public:
     
     //One needs to inherit this class and implement the constructor to 
     // instantiate handler object
     PopConAnalyzerBase(const edm::ParameterSet& pset, 
			const std::string& object_name);
     
     
     ~PopConAnalyzerBase();
     
     
   private:

     virtual void beginJob(const edm::EventSetup& es);
     
     //this method handles the transformation algorithm, 
     //Subdetector responsibility ends with returning the payload vector.
     //Therefore this code is stripped of DBOutput service, state management etc.             
     virtual void analyze(const edm::Event& evt, const edm::EventSetup& est);
     
     //initialize the source handler
     virtual void initSource(const edm::Event& evt, const edm::EventSetup& est)=0;
     
     
     //This class takes ownership of the vector (and payload objects)
     virtual void takeTheData() =0 ;


   private:
     
    std::string m_payload_name;
     
    
     //If state corruption is detected, this parameter specifies the program behaviour
     bool tryToValidate;
     //corrupted data detected, just write the log and exit
     bool corrupted;
     bool greenLight;
     //Someone claims to have fixed the problem indicated in exception section
     //TODO log it as well
     bool fixed;
     bool sinceAppend;
     std::string logMsg;
     
     bool m_debug;
     
     std::string m_offline_connection;
     
     
     virtual void displayHelper() =0;
     
   };
}

#endif
