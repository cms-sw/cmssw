#ifndef DTDBWriterInterface_H
#define DTDBWriterInterface_H

/** \class DTDBWriterInterface
 *  Utility class to write DT objects in the DB.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara & M. Zanetti
 */
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondCore/DBCommon/interface/DBWriter.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/IOVService/interface/IOV.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include <string>
#include <iostream>


namespace edm {
  class ParameterSet;
}

class DTDBWriterInterface {
public:
  /// Constructor
  DTDBWriterInterface(const edm::ParameterSet& pset);


  /// Destructor
  virtual ~DTDBWriterInterface();


  /// Write an object to DB
  template <class DTDBType>
  void write2DB(DTDBType* dbObject) {
    cond::ServiceLoader* loader = new cond::ServiceLoader;
    // Set the coral password
    ::putenv(const_cast<char*>(theCoralUser.c_str()));
    ::putenv(const_cast<char*>(theCoralPasswd.c_str()));
    
    // Set the authentication method  
    if(theAuthMethod == 1) {
      loader->loadAuthenticationService(cond::XML);
    }else{
      loader->loadAuthenticationService(cond::Env);
    }

    // Set the message level
    switch (theMessageLvl) {
    case 0 :
      loader->loadMessageService(cond::Error);
      break;    
    case 1:
      loader->loadMessageService(cond::Warning);
      break;
    case 2:
      loader->loadMessageService(cond::Info);
      break;
    case 3:
      loader->loadMessageService(cond::Debug);
      break;  
    default:
      loader->loadMessageService();
    }
    try{
      cond::DBSession* session = new cond::DBSession(theDbName);
      session->setCatalog(theDbCatalog);
      session->connect(cond::ReadWriteCreate);
      cond::DBWriter pwriter(*session, theContainerName);
      cond::DBWriter iovwriter(*session, "IOV");

      session->startUpdateTransaction();

      cond::IOV* anIOV= new cond::IOV; 
   
      std::string mytok = pwriter.template markWrite<DTDBType>(dbObject);//FIXME
      // Set the IOV
      anIOV->iov.insert(make_pair(tillWhen, mytok));
      if(debug)
	cout << "  iov size " << anIOV->iov.size() << endl;

      if(debug)
	cout << "  markWrite IOV..." << endl;
      std::string aniovToken = iovwriter.template markWrite<cond::IOV>(anIOV);
      if(debug)
	cout << "   Commit..." << endl;
      session->commit();//commit all in one
      if(debug)
	cout << "  iov size " << anIOV->iov.size() << endl;
      session->disconnect();
      delete session;
      if(debug)
	cout << "  Add MetaData... " << endl;
      cond::MetaData metadata_svc(theDbName, *loader );
      metadata_svc.connect();
      metadata_svc.addMapping(mytok, aniovToken );//FIXME
      metadata_svc.disconnect();
      if(debug)
	cout << "   Done." << endl;
    } catch( const cond::Exception& er ) {
      std::cout << er.what() << std::endl;
    } catch( ... ) {
      std::cout << "Unknown excpetion while writeing to DB!" << std::endl;
    }


    delete loader;
  }


  /// Set the IOV for the objects you want to write
  void setIOV(unsigned long till) {
    tillWhen = till;
  }

protected:

private:
  // Debug verbosity
  bool debug;
  // Coral user and passwd
  std::string theCoralUser;
  std::string theCoralPasswd;
  // The authentication method
  int theAuthMethod;
  // Set the message level
  int theMessageLvl;
  // The DB name and catalog
  std::string theDbName;
  std::string theDbCatalog;

  std::string theContainerName;
  // IOV 
  unsigned long tillWhen;

  

};
#endif

