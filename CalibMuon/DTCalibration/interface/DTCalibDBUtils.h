#ifndef CalibMuonDTCalibDBUtils_H
#define CalibMuonDTCalibDBUtils_H

/** \class DTCalibDBUtils
 *  Simple interface to PoolDBOutputService to write objects to DB.
 *
 *  $Date: $
 *  $Revision: $
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
	  cout << er.what() << endl;
	}catch(const std::exception& er){
	  cout << "[DTCalibDBUtils] caught std::exception " << er.what() << endl;
	}catch(...){
	  cout << "[DTCalibDBUtils] Funny error" << endl;
	}
      }else{
	cout << "Service PoolDBOutputService is unavailable" << endl;
      }
    
  }


protected:

private:

};
#endif

