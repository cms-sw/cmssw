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
#include "CondCore/PopCon/interface/OutputServiceWrapper.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/TagInfo.h"
#include "CondCore/DBOutputService/interface/LogDBEntry.h"


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

// #include "CondCore/PopCon/interface/StateCreator.h"
//#include "CondCore/PopCon/interface/Logger.h"

#include "CondCore/PopCon/interface/IOVPair.h"

namespace popcon
{
   class PopConAnalyzerBase : public edm::EDAnalyzer {
   public:
     
     //One needs to inherit this class and implement the constructor to 
     // instantiate handler object
     PopConAnalyzerBase(const edm::ParameterSet& pset);
     
     
     ~PopConAnalyzerBase();
     

     std::string tag() const {
       return m_tag;
     }
 
     Time_t lastSince() const {
       return m_tagInfo.lastInterval.first;
     }
    
   private:

     virtual void beginJob(const edm::EventSetup& es);
     virtual void endJob();
     
     //this method handles the transformation algorithm, 
     //Subdetector responsibility ends with returning the payload vector.
     //Therefore this code is stripped of DBOutput service, state management etc.             
     virtual void analyze(const edm::Event& evt, const edm::EventSetup& est);
     

     virtual std::string sourceId() const =0;
      
     //This class takes ownership of the vector (and payload objects)
     virtual bool takeTheData() =0 ;
     
     virtual void write() =0 ;
     
   protected:
     template <typename T>
     void writeThem (std::vector<std::pair<T*,popcon::IOVPair> > &  payload_vect, Time_t lsc){
       m_output.write(payload_vect,lsc);
     }

   protected:

     std::string logMsg;
     

     edm::Service<cond::service::PoolDBOutputService> m_dbService;

     std::string  m_record;

     std::string m_payload_name;
          
     bool sinceAppend;

     bool m_loggingOn;
 
     bool m_debug;
     
     std::string m_tag;

     cond::TagInfo m_tagInfo;

     cond::LogDBEntry m_logDBEntry;
   
     OutputServiceWrapper m_output;
    
     
     //If state corruption is detected, this parameter specifies the program behaviour
     bool tryToValidate;
     //corrupted data detected, just write the log and exit
     bool corrupted;
     bool greenLight;
     //Someone claims to have fixed the problem indicated in exception section
     //TODO log it as well
     bool fixed;
      
      
     virtual void displayHelper() const=0;
     
   };
}

#endif
