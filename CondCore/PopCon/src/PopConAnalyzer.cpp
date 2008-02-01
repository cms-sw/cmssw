#include "CondCore/PopCon/interface/PopConAnalyzerBase.h"


namespace popcon {

  PopConAnalyzerBase::PopConAnalyzerBase(const edm::ParameterSet& pset):
    m_payload_name(pset.getUntrackedParameter<std::string> ("name","")) ,
    m_offline_connection(pset.getParameter<std::string> ("connect")),
    sinceAppend(pset.getParameter<bool> ("SinceAppendMode")),
    m_debug(pset.getParameter< bool > ("debug")),
    m_output(pset.getParameter<std::string> ("record"),sinceAppend),
    tryToValidate(false), corrupted(false), greenLight (true), fixed(true)
    {
    
    //TODO set the policy (cfg or global configuration?)
    //Policy if corrupted data found
  }
  
  PopConAnalyzerBase::~PopConAnalyzerBase(){}
  

  std::string  PopConAnalyzerBase::tag() const {
    return m_output.tag();
  }


  void PopConAnalyzerBase::beginJob(const edm::EventSetup& es) {	
    if(m_debug) std::cerr << "Begin Job\n"; 
    try{
      std::cout<<"offline_connection "<<m_offline_connection<<std::endl;
      std::cout<<"payload name "<<m_payload_name<<std::endl;
      
      /*     
	
      //checks the exceptions, validates new data if necessary
      if (stc->previousExceptions(fixed)){
	std::cerr << "There's been a problem with a previous run" << std::endl;
	if (!fixed){	
	  //TODO - set the flag
	  logMsg="Running with unfixed state, EXITING";
	  return;
	}else{
	  std::cerr << "Handled exception, attempting to validate" << std::endl;
	  //TODO - implement ?
	}
      }
      
      if (stc->checkAndCompareState()){
	//std::cerr << "State OK" << std::endl;
	greenLight = true;
	logMsg="OK";
	corrupted = false;
	//	safe to run;
      }else{
	//	data is corrupted;
	if (tryToValidate){
	  std::cerr << "attempting to validate" << std::endl;
	  corrupted = false;
	}else {
	  //Report an error 	
	  std::cerr << "State Corruption, EXITING!!!\n";
	  logMsg="State corruption";
	  return;
	}
      }
      */
    }catch(popcon::Exception & er){
      std::cerr << "Begin Job PopCon exception\n";
      logMsg = "Begin Job exception";
      greenLight = false;
    }catch(cond::Exception& er){
      std::cerr<< "Begin Job cond exception " << er.what()<<std::endl;
      logMsg = "Begin Job exception";
    }catch(std::exception& e){
      greenLight = false;
      std::cerr << "Begin Job std-based exception\n";
      logMsg = "Begin Job exception";
    } 
    
  }
     
    
  //this method handles the transformation algorithm, 
  //Subdetector responsibility ends with returning the payload vector.
  //Therefore this code is stripped of DBOutput service, state management etc.     
  
  void PopConAnalyzerBase::analyze(const edm::Event& evt, const edm::EventSetup& est){
    if(m_debug) std::cerr << "Analyze Begins\n"; 
    try{
      if(greenLight){
	//get New objects 	  
	if (takeTheData()) {
	  if(m_debug)
	    displayHelper();	 
	  /// write in DB
	  write();
	}
      }
      if(m_debug)  std::cerr << "Analyze Ends\n"; 
    }catch(std::exception& e){
      std::cerr << "Analyzer Exception\n";
      std::cerr << e.what() << std::endl; 
    }
  }
  
  void PopConAnalyzerBase::endJob(){
    if(m_debug) std::cerr << "endjob begins\n";	
    
    /*
      
      try{
      if (!fixed || corrupted){	
      if(m_debug)
      std::cerr << "Corrupted | unfixed state | problem with PopCon DB\n";
      lgr->finalizeExecution(logMsg);
      }else{ //ok 
      if(m_debug)
      std::cerr << "OK, finalizing the log\n";
      lgr->finalizeExecution(logMsg);
      stc->generateStatusData();
      stc->storeStatusData();
      if(m_debug)
      std::cerr << "Deleting stc\n";	
      delete stc;
      }	  
      lgr->unlock();
      
      }catch(std::exception& e){
      std::cerr << "Exception caught in destructor: "<< e.what();
      }
      
      if(m_debug)
	std::cerr << "Deleting lgr\n";	
      delete lgr;
      
      if(m_debug)
	std::cerr << "Destructor ends\n";	

      */

    }
    
}
