#ifndef CalibMuonDTCalibDBUtils_H
#define CalibMuonDTCalibDBUtils_H

/** \class DTCalibDBUtils
 *  Simple interface to PoolDBOutputService to write objects to DB.
 *
 *  $Date: 2007/01/22 11:08:04 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include <string>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class DTCalibDBUtils {
public:
  /// Constructor
  DTCalibDBUtils();

  /// Destructor
  virtual ~DTCalibDBUtils();

  // Operations

  /// Write the payload to the DB using PoolDBOutputService.
  /// New payload are created in the DB, existing payload are appended.
  template<typename T>
  static void writeToDB(std::string record, T* payload) {
    // Write the ttrig object to DB
    edm::Service<cond::service::PoolDBOutputService> dbOutputSvc;
      if(dbOutputSvc.isAvailable()){
	try{
	  if(dbOutputSvc->isNewTagRequest(record)){
	    //create mode
	    dbOutputSvc->template createNewIOV<T>(payload,dbOutputSvc->endOfTime(),record);
	  }else{
	    //append mode. Note: correct PoolDBESSource must be loaded
	    dbOutputSvc->template appendSinceTime<T>(payload,dbOutputSvc->currentTime(),record);
	  }
	}catch(const cond::Exception& er){
	  std::cout << er.what() << std::endl;
	}catch(const std::exception& er){
	  std::cout << "[DTCalibDBUtils] caught std::exception " << er.what() << std::endl;
	}catch(...){
	  std::cout << "[DTCalibDBUtils] Funny error" << std::endl;
	}
      }else{
	std::cout << "Service PoolDBOutputService is unavailable" << std::endl;
      }
    
  }


protected:

private:

};
#endif

