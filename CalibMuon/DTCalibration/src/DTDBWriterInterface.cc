/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/03/28 10:46:48 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "CalibMuon/DTCalibration/interface/DTDBWriterInterface.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"



using namespace edm;
using namespace std;



DTDBWriterInterface::DTDBWriterInterface(const ParameterSet& pset) {
  // Debug verbosity
  debug = pset.getUntrackedParameter<bool>("debug",false);
  // Coral user and passwd
  theCoralUser = pset.getUntrackedParameter<string>("coralUser","");
  theCoralPasswd = pset.getUntrackedParameter<string>("coralPasswd","");
  // The authentication method
  theAuthMethod = pset.getUntrackedParameter<int>("authMethod",0);
  // Set the message level
  theMessageLvl = pset.getUntrackedParameter<int>("messageLevel",0);
  // The DB name and catalog
  theDbName = pset.getParameter<string>("dbName");
  theDbCatalog = pset.getParameter<string>("dbCatalog");
  theTag = pset.getParameter<string>("tag");
  theContainerName = pset.getUntrackedParameter<string>("containerName","DTDBObject");
  tillWhen = edm::IOVSyncValue::endOfTime().eventID().run();
}



DTDBWriterInterface::~DTDBWriterInterface(){}

