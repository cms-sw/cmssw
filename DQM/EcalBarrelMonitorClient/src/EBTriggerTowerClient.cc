/*
 * \file EBTriggerTowerClient.cc
 *
 * $Date: 2011/09/02 13:55:01 $
 * $Revision: 1.128 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBTriggerTowerClient.h"

EBTriggerTowerClient::EBTriggerTowerClient(const edm::ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<std::vector<int> >("superModules", superModules_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    o01_[ism-1] = 0;
    meo01_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me_o01_[ism-1] = 0;
    me_o02_[ism-1] = 0;

  }

  ievt_ = 0;
  jevt_ = 0;
  dqmStore_ = 0;

}

EBTriggerTowerClient::~EBTriggerTowerClient() {

}

void EBTriggerTowerClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBTriggerTowerClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBTriggerTowerClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBTriggerTowerClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBTriggerTowerClient::endJob(void) {

  if ( debug_ ) std::cout << "EBTriggerTowerClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBTriggerTowerClient::endRun(void) {

  if ( debug_ ) std::cout << "EBTriggerTowerClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBTriggerTowerClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/TriggerPrimitives/Timing" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_o01_[ism-1] ) dqmStore_->removeElement( me_o01_[ism-1]->getName() );
    name = "TrigPrimClient Timing " + Numbers::sEB(ism);
    me_o01_[ism-1] = dqmStore_->book2D(name, name, 17, 0., 85., 4, 0., 20.);
    me_o01_[ism-1]->setAxisTitle("ieta'", 1);
    me_o01_[ism-1]->setAxisTitle("iphi'", 2);

//     if ( me_o02_[ism-1] ) dqmStore_->removeElement( me_o02_[ism-1]->getName() );
//     name = "TrigPrimClient Non Single Timing " + Numbers::sEB(ism);
//     me_o02_[ism-1] = dqmStore_->book2D(name, name, 17, 0., 85., 4, 0., 20.);
//     me_o02_[ism-1]->setAxisTitle("ieta'", 1);
//     me_o02_[ism-1]->setAxisTitle("iphi'", 2);
//     me_o02_[ism-1]->setAxisTitle("fraction", 3);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_o01_[ism-1] ) me_o01_[ism-1]->Reset();
    if ( me_o02_[ism-1] ) me_o02_[ism-1]->Reset();

  }

}

void EBTriggerTowerClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( o01_[ism-1] ) delete o01_[ism-1];
    }

    o01_[ism-1] = 0;

    meo01_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_o01_[ism-1] ) dqmStore_->removeElement( me_o01_[ism-1]->getFullname() );
    me_o01_[ism-1] = 0;
    if ( me_o02_[ism-1] ) dqmStore_->removeElement( me_o02_[ism-1]->getFullname() );
    me_o02_[ism-1] = 0;

  }

}

#ifdef WITH_ECAL_COND_DB
bool EBTriggerTowerClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  return true;

}
#endif

void EBTriggerTowerClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBTriggerTowerClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  MonitorElement* me;
  std::string name;
  std::stringstream ss;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/TriggerPrimitives/EmulMatching/TrigPrimTask matching index " + Numbers::sEB(ism) );
    o01_[ism-1] = UtilsClient::getHisto( me, cloneME_, o01_[ism-1] );
    meo01_[ism-1] = me;

    if ( me_o01_[ism-1] ) me_o01_[ism-1]->Reset();
    if ( me_o02_[ism-1] ) me_o02_[ism-1]->Reset();

    if ( o01_[ism-1] ) {

      for (int ie = 1; ie <= 17; ie++) {
	for (int ip = 1; ip <= 4; ip++) {

          // find the most frequent TP timing that matches the emulator
          float index=-1;
          double max=0;
          double total=0;

	  int itcc(ism + 36);
	  int itt((ie - 1) * 4 + ip);

          for (int j = -1; j<6; j++) {
            double sampleEntries = o01_[ism-1]->GetBinContent(itt, j+2);
            if(sampleEntries > max) {
              index = j;
              max = sampleEntries;
            }
            total += sampleEntries;
          }
          if ( max > 0 ) {
	    me_o01_[ism-1]->setBinContent(ie, ip, index );
          }
	  if ((int)total != (int)max) {
	    ss.str("");
	    ss << "TT " << itcc << " " << itt;
	    name = "TrigPrimClient non single timing " + ss.str();
	    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives/EmulationErrors/Timing");
	    me = dqmStore_->book1D(name, name, 1, 0., 1.);
	    me->setBinContent(1, 1.0 - max / total);
	    me->setAxisTitle("fraction", 2);
	  }
        }

      }
    }

  }

}

