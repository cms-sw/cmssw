// -*- C++ -*-
//
// Package:    CondTools/SiPixel
// Class:      PixelPopConCalibSourceHandler
// 
/**\class PixelPopConCalibSourcehandler PixelPopConCalibSourceHandler.cc CondTools/SiPixel/src/PixelPopConCalibSourceHandler.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Eads
//         Created:  8 Feb 2008
// $Id: PixelPopConCalibSourceHandler.cc,v 1.4 2010/06/02 06:10:11 ursl Exp $
//
//

#include "CondTools/SiPixel/interface/PixelPopConCalibSourceHandler.h"

#include <iostream>
#include <sstream>

// DBCommon includes
// #include "CondCore/DBCommon/interface/SessionConfiguration.h"
// #include "CondCore/DBCommon/interface/Exception.h"
// #include "CondCore/DBCommon/interface/Connection.h"

// CORAL includes
// #include "RelationalAccess/IView.h"
// #include "RelationalAccess/ISessionProxy.h"
// #include "RelationalAccess/ISchema.h"
// #include "RelationalAccess/ISessionProperties.h"
// #include "RelationalAccess/IQuery.h"
// #include "RelationalAccess/ICursor.h"
// #include "CoralBase/AttributeList.h"
// #include "CoralBase/Attribute.h"

#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"

// test poolDBOutput
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


using namespace std;

// constructor
PixelPopConCalibSourceHandler::PixelPopConCalibSourceHandler(edm::ParameterSet const &pset) {

  // try to get a parameter
  _connectString = pset.getParameter<string>("connectString");
  //cout << "  connectString: " << _connectString << endl;

  // get the schema and view name to use from the config file
  _viewName = pset.getParameter<string>("viewName");
  _schemaName = pset.getParameter<string>("schemaName");  

  // get the key name and/or run number to use
  _runNumber = pset.getParameter<int>("runNumber");
  _configKeyName = pset.getParameter<string>("configKeyName");

  // get the "since" IOV parameter
  _sinceIOV = pset.getParameter<unsigned int>("sinceIOV");
  
} // constructor

// destructor
PixelPopConCalibSourceHandler::~PixelPopConCalibSourceHandler() {

} // destructor

string PixelPopConCalibSourceHandler::id() const {

  return string("PixelPopConCalibSourceHandler");

} // string PixelPopConCalibSourceHandler::id()


// getNewObjects method using coral 
void PixelPopConCalibSourceHandler::getNewObjects_coral() {
	cout << "I'm sorry, the PixelPopConCalibSourceHandler::getNewObjects_coral() is not currently working." << endl;
	cout << "I am not able to build calibration configuration objects from the database" << endl;
	return;
	
//   // create the empty SiPixelCalibConfiguration object
//   SiPixelCalibConfiguration* calibConfig = new SiPixelCalibConfiguration();
  
//   // set up the DB session
//   cond::DBSession* dbsession = new cond::DBSession();
//   dbsession->configuration().setAuthenticationMethod(cond::XML);
//   dbsession->configuration().setMessageLevel(cond::Error);
//   try{
//     dbsession->open();
//   }catch(cond::Exception& er){
//     std::cerr<< "CoralIface::initialize cond " << er.what()<<std::endl;
//     throw;
//   }catch(std::exception& er){
//     std::cerr<< "CoralIface::initialize std " << er.what()<<std::endl;
//     throw;
//   }

//   cond::Connection dbconn(_connectString);
//   dbconn.connect(dbsession);
//   cond::CoralTransaction& coraldb=dbconn.coralTransaction();
//   coraldb.start(true);

//   // build and execute the query
//   coral::IQuery* query = coraldb.coralSessionProxy().schema(_schemaName).newQuery();
//   query->addToTableList(_viewName);
//   query->addToOutputList("CONFG_KEY");
//   query->addToOutputList("VERSION");
//   query->addToOutputList("RUN_TYPE");
//   query->addToOutputList("RUN_NUMBER");
//   query->addToOutputList("CALIB_OBJ_DATA_FILE");
//   query->addToOutputList("CALIB_OBJ_DATA_CLOB");

//   // if _runNumber is -1, query by config key name
//   if (_runNumber == -1)
//     query->setCondition("CONFG_KEY = '" + _configKeyName + "'", coral::AttributeList());
//   else // query by run number
//     query->setCondition("RUN_NUMBER = " + _runNumber, coral::AttributeList());
//   coral::ICursor& cursor = query->execute();

//   // parse the response, build the Calib object
//   bool found_fNtriggers = false;
//   bool found_fRowPattern = false;
//   bool found_fColumnPattern = false;
//   bool found_fVCalValues = false;
//   bool found_fMode = false;
//   while ( cursor.next() ) {
// 	  cout << "Inside cursor.next() loop" << endl;
//     //cursor.currentRow().toOutputStream( std::cout ) << std::endl;
//     coral::AttributeList row = cursor.currentRow();
    
//     string mystring = row["CALIB_OBJ_DATA_CLOB"].data<string>();
//     cout << "mystring: " << mystring << endl;

//     // get fMode
//     if (!found_fMode) {
//       calibConfig->setCalibrationMode(row["CALIB_MODE"].data<string>());
//       found_fMode = true;
//     } // if (!found_fMode)

//     // fill fNTriggers
//     if (row["PARAMETER"].data<string>() == "Number of Triggers") {
//       if (found_fNtriggers) {
// 	cout << "Warning: found mulitple entries for fNtriggers!" << endl;
//       }
//       int fNtriggers = atoi(row["VALUE"].data<string>().c_str());
//       //cout << "Number of triggers: " << fNtriggers << endl;
//       calibConfig->setNTriggers(static_cast<short>(fNtriggers));
//       found_fNtriggers = true;
//     } // fill fNTriggers

//     /*
//     // fill fROCIds
//     if (row["PARAMETER"].data<string>() == "ROC") {
//       if (found_fROCIds) {
// 	cout << "Warning: found mulitple entries for fROCIds!" << endl;
//       }
//       string rocidlist = row["VALUE"].data<string>();
//       // split the string
//       string buff;
//       vector<string> ROCIds;
//       stringstream ss(rocidlist);
//       while (ss >> buff) {
// 	ROCIds.push_back(buff);
//       }
//       calibConfig->setROCIds(ROCIds);
//       found_fROCIds = true;
//     } // fill fROCIds
//     */

//     // fill fRowPattern
//     if (row["PARAMETER"].data<string>() == "RowNums") {
//       if (found_fRowPattern) {
// 	cout << "Warning: found mulitple entries for fRowPattern!" << endl;
//       }
//       string rowlist = row["VALUE"].data<string>();
//       // split the string
//       string buff;
//       vector<short> rows;
//       stringstream ss(rowlist);
//       while (ss >> buff) {
// 	rows.push_back(static_cast<short>(atoi(buff.c_str())));
//       }
//       calibConfig->setRowPattern(rows);
//       found_fRowPattern = true;
//     } // fill fRowPattern

//     // fill fColumnPattern
//     if (row["PARAMETER"].data<string>() == "ColNums") {
//       if (found_fColumnPattern) {
// 	cout << "Warning: found mulitple entries for fColumnPattern!" << endl;
//       }
//       string collist = row["VALUE"].data<string>();
//       // split the string
//       string buff;
//       vector<short> cols;
//       stringstream ss(collist);
//       while (ss >> buff) {
// 	cols.push_back(static_cast<short>(atoi(buff.c_str())));
//       }
//       calibConfig->setColumnPattern(cols);
//       found_fColumnPattern = true;
//     } // fill fColumnPattern

//     // fill fVCalValues
//     if (row["CALIB_OBJECT"].data<string>() == "Vcal DAC") {
//       if (found_fVCalValues) {
// 	cout << "Warning: found mulitple entries for fVCalValues!" << endl;
//       }
//       string vcallist = row["VALUE"].data<string>();
//       // split the string
//       string buff;
//       vector<short> vcals;
//       stringstream ss(vcallist);
//       while (ss >> buff) {
// 	vcals.push_back(static_cast<short>(atoi(buff.c_str())));
//       }
//       calibConfig->setVCalValues(vcals);
//       found_fVCalValues = true;
//     } // fill fRowPattern

//     /*
//     for (coral::AttributeList::iterator it = row.begin();
// 	 it != row.end(); ++it) {
//       if it->specification().name() == "PARAMETER"

//       if (it->specification().name() == "RUN_NUMBER")
// 	cout << it->specification().name() << ", " << it->data<long long>() << endl;
//       else
// 	cout << it->specification().name() << ", " << it->data<string>() << endl;
    
//     } // loop over attribute list
//     */
//   } // while (cursor.next())

//   // spit out calibConfig object
//   cout << endl << "** calibConfig: " << endl;
//   cout << "      fNtriggers: " << calibConfig->getNTriggers() << endl;
//   /*
//   cout << "      fROCIds: ";
//   vector<string> rocids = calibConfig->getROCIds();
//   for (vector<string>::const_iterator it = rocids.begin();
//        it != rocids.end(); ++it)
//     cout << *it << ", ";
//   cout << endl;
//   */
//   cout << "      fRowPattern: ";
//   vector<short> rowpattern = calibConfig->getRowPattern();
//   for (vector<short>::const_iterator it = rowpattern.begin();
//        it != rowpattern.end(); ++it)
//     cout << *it << ", ";
//   cout << endl;
//   cout << "      fColumnPattern: ";
//   vector<short> columnpattern = calibConfig->getColumnPattern();
//   for (vector<short>::const_iterator it = columnpattern.begin();
//        it != columnpattern.end(); ++it)
//     cout << *it << ", ";
//   cout << endl;
//   cout << "      fVcalValues: ";
//   vector<short> vcalvalues = calibConfig->getVCalValues();
//   for (vector<short>::const_iterator it = vcalvalues.begin();
//        it != vcalvalues.end(); ++it)
//     cout << *it << ", ";
//   cout << endl;
//   cout << "      fNmode: " << calibConfig->getCalibrationMode() << endl;

//   // see what's in the db
//   //cout << "tagInfo: " << tagInfo().name << ", " << tagInfo().lastInterval.first << endl;

//   m_to_transfer.push_back(std::make_pair(calibConfig,_sinceIOV));


//   delete dbsession;
} // void PixelPopConCalibSourceHandler::getNewObjects_coral()

// getNewObjects method using a text file
void PixelPopConCalibSourceHandler::getNewObjects_file() {
	// check to make sure the _connectString begins with "file//"
	if (_connectString.find("file://") != 0) {
		cout << "Invalid connectString: " << _connectString << endl;
		cout << "It should begin with \"file://\"" << endl;
		return;
	}
	// strip the "file://" from the connect string
	string inputFilename = _connectString.erase(0, 7);
	
	cout << "PopCon-ing the calibration configuration file located at " << inputFilename << endl;

	// use online code to parse the file
	pos::PixelCalibConfiguration fancyCalib(inputFilename);
	SiPixelCalibConfiguration *calibConfig = new SiPixelCalibConfiguration(fancyCalib);
	
	m_to_transfer.push_back(std::make_pair(calibConfig, _sinceIOV));
  
} // void PixelPopConCalibSourceHandler::getNewObjects_file()

