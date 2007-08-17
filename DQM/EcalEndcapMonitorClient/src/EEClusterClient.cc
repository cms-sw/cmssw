/*
 * \file EEClusterClient.cc
 *
 * $Date: 2007/08/14 17:44:45 $
 * $Revision: 1.14 $
 * \author G. Della Ricca
 * \author E. Di Marco
 *
*/

#include <memory>
#include <iostream>
#include <fstream>

#include "TStyle.h"
#include "TGaxis.h"
#include "TGraph.h"
#include "TLine.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalsOnlineDat.h"

#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EEClusterClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEClusterClient::EEClusterClient(const ParameterSet& ps){

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // enableQT switch
  enableQT_ = ps.getUntrackedParameter<bool>("enableQT", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  allEE_[0] = 0;
  allEE_[1] = 0;
  allEE_[2] = 0;

  allEEBasic_[0] = 0;
  allEEBasic_[1] = 0;
  allEEBasic_[2] = 0;

  eneEE_[0] = 0;
  eneEE_[1] = 0;
  enePolarEE_[0] = 0;
  enePolarEE_[1] = 0;
  numEE_[0] = 0;
  numEE_[1] = 0;
  numPolarEE_[0] = 0;
  numPolarEE_[1] = 0;

  eneEEBasic_[0] = 0;
  eneEEBasic_[1] = 0;
  enePolarEEBasic_[0] = 0;
  enePolarEEBasic_[1] = 0;
  numEEBasic_[0] = 0;
  numEEBasic_[1] = 0;
  numPolarEEBasic_[0] = 0;
  numPolarEEBasic_[1] = 0;

  s_ = 0;

}

EEClusterClient::~EEClusterClient(){

}

void EEClusterClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EEClusterClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {


  }

}

void EEClusterClient::beginRun(void){

  if ( verbose_ ) cout << "EEClusterClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EEClusterClient::endJob(void) {

  if ( verbose_ ) cout << "EEClusterClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EEClusterClient::endRun(void) {

  if ( verbose_ ) cout << "EEClusterClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EEClusterClient::setup(void) {

  dbe_->setCurrentFolder( "EcalEndcap/EEClusterClient" );

}

void EEClusterClient::cleanup(void) {

  if ( cloneME_ ) {
    if ( allEEBasic_[0] ) delete allEEBasic_[0];
    if ( allEEBasic_[1] ) delete allEEBasic_[1];
    if ( allEEBasic_[2] ) delete allEEBasic_[2];

    if ( eneEEBasic_[0] ) delete eneEEBasic_[0];
    if ( eneEEBasic_[1] ) delete eneEEBasic_[1];
    if ( enePolarEEBasic_[0] ) delete enePolarEEBasic_[0];
    if ( enePolarEEBasic_[1] ) delete enePolarEEBasic_[1];
    if ( numEEBasic_[0] ) delete numEEBasic_[0];
    if ( numEEBasic_[1] ) delete numEEBasic_[1];
    if ( numPolarEEBasic_[0] ) delete numPolarEEBasic_[0];
    if ( numPolarEEBasic_[1] ) delete numPolarEEBasic_[1];

    if ( allEE_[0] ) delete allEE_[0];
    if ( allEE_[1] ) delete allEE_[1];
    if ( allEE_[2] ) delete allEE_[2];

    if ( eneEE_[0] ) delete eneEE_[0];
    if ( eneEE_[1] ) delete eneEE_[1];
    if ( enePolarEE_[0] ) delete enePolarEE_[0];
    if ( enePolarEE_[1] ) delete enePolarEE_[1];
    if ( numEE_[0] ) delete numEE_[0];
    if ( numEE_[1] ) delete numEE_[1];
    if ( numPolarEE_[0] ) delete numPolarEE_[0];
    if ( numPolarEE_[1] ) delete numPolarEE_[1];

    if ( s_ ) delete s_;

  }

  allEEBasic_[0] = 0;
  allEEBasic_[1] = 0;
  allEEBasic_[2] = 0;

  eneEEBasic_[0] = 0;
  eneEEBasic_[1] = 0;
  enePolarEEBasic_[0] = 0;
  enePolarEEBasic_[1] = 0;
  numEEBasic_[0] = 0;
  numEEBasic_[1] = 0;
  numPolarEEBasic_[0] = 0;
  numPolarEEBasic_[1] = 0;

  allEE_[0] = 0;
  allEE_[1] = 0;
  allEE_[2] = 0;

  eneEE_[0] = 0;
  eneEE_[1] = 0;
  enePolarEE_[0] = 0;
  enePolarEE_[1] = 0;
  numEE_[0] = 0;
  numEE_[1] = 0;
  numPolarEE_[0] = 0;
  numPolarEE_[1] = 0;

  s_ = 0;

}

bool EEClusterClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EEClusterClient::subscribe(void){

  if ( verbose_ ) cout << "EEClusterClient: subscribe" << endl;

  Char_t histo[200];

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC size");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy map EE +");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number map EE +");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy polar map EE +");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number polar map EE +");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy map EE -");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number map EE -");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy polar map EE -");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number polar map EE -");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC size");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy map EE +");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number map EE +");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy polar map EE +");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number polar map EE +");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy map EE -");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number map EE -");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy polar map EE -");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number polar map EE -");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT dicluster invariant mass");
  mui_->subscribe(histo);


  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EEClusterClient: collate" << endl;

    sprintf(histo, "EECLT BC energy");
    me_allEEBasic_[0] = mui_->collate1D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy");
    mui_->add(me_allEEBasic_[0], histo);

    sprintf(histo, "EECLT BC number");
    me_allEEBasic_[1] = mui_->collate1D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number");
    mui_->add(me_allEEBasic_[1], histo);

    sprintf(histo, "EECLT BC size");
    me_allEEBasic_[2] = mui_->collate1D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC size");
    mui_->add(me_allEEBasic_[2], histo);

    sprintf(histo, "EECLT BC energy map EE -");
    me_eneEEBasic_[0] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy map EE -");
    mui_->add(me_eneEEBasic_[0], histo);

    sprintf(histo, "EECLT BC number map EE -");
    me_numEEBasic_[0] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number map EE -");
    mui_->add(me_numEEBasic_[0], histo);

    sprintf(histo, "EECLT BC energy polar map EE -");
    me_enePolarEEBasic_[0] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy polar map EE -");
    mui_->add(me_enePolarEEBasic_[0], histo);

    sprintf(histo, "EECLT BC number polar map EE -");
    me_numPolarEEBasic_[0] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number map EE -");
    mui_->add(me_numPolarEEBasic_[0], histo);

    sprintf(histo, "EECLT BC energy map EE +");
    me_eneEEBasic_[1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy map EE +");
    mui_->add(me_eneEEBasic_[1], histo);

    sprintf(histo, "EECLT BC number map EE +");
    me_numEEBasic_[1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number map EE +");
    mui_->add(me_numEEBasic_[1], histo);

    sprintf(histo, "EECLT BC energy polar map EE +");
    me_enePolarEEBasic_[1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy polar map EE +");
    mui_->add(me_enePolarEEBasic_[1], histo);

    sprintf(histo, "EECLT BC number polar map EE +");
    me_numPolarEEBasic_[1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number polar map EE +");
    mui_->add(me_numPolarEEBasic_[1], histo);

    sprintf(histo, "EECLT SC energy");
    me_allEE_[0] = mui_->collate1D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy");
    mui_->add(me_allEE_[0], histo);

    sprintf(histo, "EECLT SC number");
    me_allEE_[1] = mui_->collate1D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number");
    mui_->add(me_allEE_[1], histo);

    sprintf(histo, "EECLT SC size");
    me_allEE_[2] = mui_->collate1D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC size");
    mui_->add(me_allEE_[2], histo);

    sprintf(histo, "EECLT SC energy map EE -");
    me_eneEE_[0] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy map EE -");
    mui_->add(me_eneEE_[0], histo);

    sprintf(histo, "EECLT SC number map EE -");
    me_numEE_[0] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number map EE -");
    mui_->add(me_numEE_[0], histo);

    sprintf(histo, "EECLT SC energy polar map EE -");
    me_enePolarEE_[0] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy polar map EE -");
    mui_->add(me_enePolarEE_[0], histo);

    sprintf(histo, "EECLT SC number polar map EE -");
    me_numPolarEE_[0] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number map EE -");
    mui_->add(me_numPolarEE_[0], histo);

    sprintf(histo, "EECLT SC energy map EE +");
    me_eneEE_[1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy map EE +");
    mui_->add(me_eneEE_[1], histo);

    sprintf(histo, "EECLT SC number map EE +");
    me_numEE_[1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number map EE +");
    mui_->add(me_numEE_[1], histo);

    sprintf(histo, "EECLT SC energy polar map EE +");
    me_enePolarEE_[1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy polar map EE +");
    mui_->add(me_enePolarEE_[1], histo);

    sprintf(histo, "EECLT SC number polar map EE +");
    me_numPolarEE_[1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number polar map EE +");
    mui_->add(me_numPolarEE_[1], histo);

    sprintf(histo, "EECLT dicluster invariant mass");
    me_s_ = mui_->collate1D(histo, histo, "EcalEndcap/Sums/EEClusterTask");
    sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT dicluster invariant mass");
    mui_->add(me_s_, histo);
  }

}

void EEClusterClient::subscribeNew(void){

  Char_t histo[200];

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC size");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy map EE -");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number map EE -");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy polar map EE -");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number polar map EE -");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy map EE +");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number map EE +");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy polar map EE +");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number polar map EE +");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC size");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy map EE -");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number map EE -");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy polar map EE -");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number polar map EE -");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy map EE +");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number map EE +");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy polar map EE +");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number polar map EE +");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT dicluster invariant mass");
  mui_->subscribeNew(histo);

}

void EEClusterClient::unsubscribe(void){

  if ( verbose_ ) cout << "EEClusterClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EEClusterClient: uncollate" << endl;

    if ( mui_ ) {

      mui_->removeCollate(me_allEEBasic_[0]);
      mui_->removeCollate(me_allEEBasic_[1]);
      mui_->removeCollate(me_allEEBasic_[2]);
      mui_->removeCollate(me_eneEEBasic_[0]);
      mui_->removeCollate(me_eneEEBasic_[1]);
      mui_->removeCollate(me_enePolarEEBasic_[0]);
      mui_->removeCollate(me_enePolarEEBasic_[1]);
      mui_->removeCollate(me_numEEBasic_[0]);
      mui_->removeCollate(me_numEEBasic_[1]);
      mui_->removeCollate(me_numPolarEEBasic_[0]);
      mui_->removeCollate(me_numPolarEEBasic_[1]);

      mui_->removeCollate(me_allEE_[0]);
      mui_->removeCollate(me_allEE_[1]);
      mui_->removeCollate(me_allEE_[2]);
      mui_->removeCollate(me_eneEE_[0]);
      mui_->removeCollate(me_eneEE_[1]);
      mui_->removeCollate(me_enePolarEE_[0]);
      mui_->removeCollate(me_enePolarEE_[1]);
      mui_->removeCollate(me_numEE_[0]);
      mui_->removeCollate(me_numEE_[1]);
      mui_->removeCollate(me_numPolarEE_[0]);
      mui_->removeCollate(me_numPolarEE_[1]);
      mui_->removeCollate(me_s_);

    }

  }

  Char_t histo[200];

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC size");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy map EE -");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number map EE -");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy polar map EE -");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number polar map EE -");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy map EE +");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number map EE +");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC energy polar map EE +");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT BC number polar map EE +");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC size");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy map EE -");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number map EE -");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy polar map EE -");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number polar map EE -");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy map EE +");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number map EE +");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC energy polar map EE +");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT SC number polar map EE +");
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalEndcap/EEClusterTask/EECLT dicluster invariant mass");
  mui_->unsubscribe(histo);


}

void EEClusterClient::softReset(void){

}

void EEClusterClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EEClusterClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[200];

  MonitorElement* me;

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC energy");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC energy").c_str());
  }
  me = dbe_->get(histo);
  allEEBasic_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, allEEBasic_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC number");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC number").c_str());
  }
  me = dbe_->get(histo);
  allEEBasic_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, allEEBasic_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC size");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC size").c_str());
  }
  me = dbe_->get(histo);
  allEEBasic_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, allEEBasic_[2] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC energy map EE -");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC energy map EE -").c_str());
  }
  me = dbe_->get(histo);
  eneEEBasic_[0] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, eneEEBasic_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC number map EE -");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC number map EE -").c_str());
  }
  me = dbe_->get(histo);
  numEEBasic_[0] = UtilsClient::getHisto<TH2F*>( me, cloneME_, numEEBasic_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC energy polar map EE -");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC energy polar map EE -").c_str());
  }
  me = dbe_->get(histo);
  enePolarEEBasic_[0] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, enePolarEEBasic_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC number polar map EE -");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC number polar map EE -").c_str());
  }
  me = dbe_->get(histo);
  numPolarEEBasic_[0] = UtilsClient::getHisto<TH2F*>( me, cloneME_, numPolarEEBasic_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC energy map EE +");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC energy map EE +").c_str());
  }
  me = dbe_->get(histo);
  eneEEBasic_[1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, eneEEBasic_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC number map EE +");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC number map EE +").c_str());
  }
  me = dbe_->get(histo);
  numEEBasic_[1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, numEEBasic_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC energy polar map EE +");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC energy polar map EE +").c_str());
  }
  me = dbe_->get(histo);
  enePolarEEBasic_[1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, enePolarEEBasic_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT BC number polar map EE +");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT BC number polar map EE +").c_str());
  }
  me = dbe_->get(histo);
  numPolarEEBasic_[1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, numPolarEEBasic_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC energy");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC energy").c_str());
  }
  me = dbe_->get(histo);
  allEE_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, allEE_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC number");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC number").c_str());
  }
  me = dbe_->get(histo);
  allEE_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, allEE_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC size");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC size").c_str());
  }
  me = dbe_->get(histo);
  allEE_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, allEE_[2] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC energy map EE -");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC energy map EE -").c_str());
  }
  me = dbe_->get(histo);
  eneEE_[0] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, eneEE_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC number map EE -");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC number map EE -").c_str());
  }
  me = dbe_->get(histo);
  numEE_[0] = UtilsClient::getHisto<TH2F*>( me, cloneME_, numEE_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC energy polar map EE -");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC energy polar map EE -").c_str());
  }
  me = dbe_->get(histo);
  enePolarEE_[0] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, enePolarEE_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC number polar map EE -");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC number polar map EE -").c_str());
  }
  me = dbe_->get(histo);
  numPolarEE_[0] = UtilsClient::getHisto<TH2F*>( me, cloneME_, numPolarEE_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC energy map EE +");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC energy map EE +").c_str());
  }
  me = dbe_->get(histo);
  eneEE_[1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, eneEE_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC number map EE +");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC number map EE +").c_str());
  }
  me = dbe_->get(histo);
  numEE_[1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, numEE_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC energy polar map EE +");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC energy polar map EE +").c_str());
  }
  me = dbe_->get(histo);
  enePolarEE_[1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, enePolarEE_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT SC number polar map EE +");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT SC number polar map EE +").c_str());
  }
  me = dbe_->get(histo);
  numPolarEE_[1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, numPolarEE_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEClusterTask/EECLT dicluster invariant mass");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEClusterTask/EECLT dicluster invariant mass").c_str());
  }
  me = dbe_->get(histo);
  s_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, s_ );

}

void EEClusterClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EEClusterClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:ClusterTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">CLUSTER</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  //  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  //  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  //  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  //  htmlFile << "<hr>" << endl;
  htmlFile <<  "<a href=\"#bc_plots\"> Basic Clusters plots </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#sc_plots\"> Super Clusters plots </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#hl_plots\"> Higher Level Quantities plots </a>" << endl;
  htmlFile << "<p>" << endl;

  htmlFile << "<hr>" << endl;
  htmlFile << "<p>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize1D = 250;
  const int csize2D = 500;

  const double histMax = 1.e15;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C labelGrid1("labelGrid1","label grid for EE -", 10, -150.0, 150.0, 10, -150.0, 150.0);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      labelGrid1.SetBinContent(i, j, -10);
    }
  }

  labelGrid1.SetBinContent(2, 5, -3);
  labelGrid1.SetBinContent(2, 7, -2);
  labelGrid1.SetBinContent(4, 9, -1);
  labelGrid1.SetBinContent(7, 9, -9);
  labelGrid1.SetBinContent(9, 7, -8);
  labelGrid1.SetBinContent(9, 5, -7);
  labelGrid1.SetBinContent(8, 3, -6);
  labelGrid1.SetBinContent(5, 2, -5);
  labelGrid1.SetBinContent(3, 3, -4);

  labelGrid1.SetMarkerSize(2);
  labelGrid1.SetMinimum(-9.01);
  labelGrid1.SetMaximum(-0.01);

  TH2C labelGrid2("labelGrid2","label grid for EE +", 10, -150.0, 150.0, 10, -150.0, 150.0);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      labelGrid2.SetBinContent(i, j, -10);
    }
  }

  labelGrid2.SetBinContent(2, 5, +7);
  labelGrid2.SetBinContent(2, 7, +8);
  labelGrid2.SetBinContent(4, 9, +9);
  labelGrid2.SetBinContent(7, 9, +1);
  labelGrid2.SetBinContent(9, 7, +2);
  labelGrid2.SetBinContent(9, 5, +3);
  labelGrid2.SetBinContent(8, 3, +4);
  labelGrid2.SetBinContent(6, 2, +5);
  labelGrid2.SetBinContent(3, 3, +6);

  labelGrid2.SetMarkerSize(2);
  labelGrid2.SetMinimum(+0.01);
  labelGrid2.SetMaximum(+9.01);

  TGaxis Xaxis(-150.0, -150.0,  150.0, -150.0, -150.0, 150.0, 50210, "N");
  TGaxis Yaxis(-150.0, -150.0, -150.0,  150.0, -150.0, 150.0, 50210, "N");
  Xaxis.SetLabelSize(0.02);
  Yaxis.SetLabelSize(0.02);

  string imgNameAll[3], imgNameEneMap[2], imgNameEnePolarMap[2], imgNameNumMap[2], imgNameNumPolarMap[2];
  string imgNameEneXproj[2], imgNameNumXproj[2], imgNameEneYproj[2], imgNameNumYproj[2];
  string imgNameHL, imgName, meName;

  TCanvas* cEne = new TCanvas("cEne", "Temp", csize1D, csize1D);
  TCanvas* cMap = new TCanvas("cMap", "Temp", csize2D, csize2D);

  TH1F* obj1f;
  TProfile2D* objp;
  TProfile2D* objpPolar;
  TH2F* objf;
  TH2F* objfPolar;
  TH1D* obj1dX;
  TH1D* obj1dY;

  gStyle->SetPaintTextFormat("+g");

  // ====> B A S I C     C L U S T E R S <===
  // ==========================================================================
  // all Ecal Endcap 1D plots
  // ==========================================================================

  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    imgNameAll[iCanvas-1] = "";

    obj1f = allEEBasic_[iCanvas-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
      imgNameAll[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameAll[iCanvas-1];

      cEne->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
      gPad->SetLogy(0);
    }
  }

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"bc_plots\"> <B> Basic Clusters plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    if ( imgNameAll[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameAll[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // Energy profiles
  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {

    imgNameEneMap[iCanvas-1] = "";

    objp = eneEEBasic_[iCanvas-1];
    objpPolar = enePolarEEBasic_[iCanvas-1];

    if ( objp ) {

      meName = objp->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameEneMap[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameEneMap[iCanvas-1];

      cMap->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      objp->GetXaxis()->SetNdivisions(10, kFALSE);
      objp->GetYaxis()->SetNdivisions(10, kFALSE);
      objp->GetZaxis()->SetLabelSize(0.02);
      cMap->SetGridx();
      cMap->SetGridy();
      objp->Draw("colz");
      if ( iCanvas == 1 ) labelGrid1.Draw("text,same");
      if ( iCanvas == 2 ) labelGrid2.Draw("text,same");
      Xaxis.Draw();
      Yaxis.Draw();
      cMap->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(3.0*(Numbers::ixSectorsEE[i]-50), 3.0*(Numbers::iySectorsEE[i]-50), 3.0*(Numbers::ixSectorsEE[i+1]-50), 3.0*(Numbers::iySectorsEE[i+1]-50));
        }
      }
      cMap->Update();
      objp->GetXaxis()->SetLabelColor(0);
      objp->GetYaxis()->SetLabelColor(0);
      cMap->SaveAs(imgName.c_str());
      objp->GetXaxis()->SetLabelColor(1);
      objp->GetYaxis()->SetLabelColor(1);
    }

    if ( objpPolar ) {
      meName = objpPolar->GetName();
      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameEnePolarMap[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameEnePolarMap[iCanvas-1];

      char projXName[100];
      char projYName[100];
      sprintf(projXName,"%s_px",meName.c_str());
      imgNameEneXproj[iCanvas-1] = string(projXName) + ".png";
      sprintf(projYName,"%s_py",meName.c_str());
      imgNameEneYproj[iCanvas-1] = string(projYName) + ".png";

      obj1dX = objpPolar->ProjectionX(projXName,1,objpPolar->GetNbinsY(),"e");
      obj1dY = objpPolar->ProjectionY(projYName,1,objpPolar->GetNbinsX(),"e");

      cEne->cd();
      gStyle->SetOptStat("emr");
      obj1dX->GetXaxis()->SetNdivisions(50205, kFALSE);
      obj1dY->GetXaxis()->SetNdivisions(50206, kFALSE);

      imgName = htmlDir + imgNameEneXproj[iCanvas-1];
      obj1dX->SetStats(kTRUE);
      obj1dX->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());

      imgName = htmlDir + imgNameEneYproj[iCanvas-1];
      obj1dY->SetStats(kTRUE);
      obj1dY->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
    }
  }

  // Cluster occupancy profiles
  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {

    imgNameNumMap[iCanvas-1] = "";

    objf = numEEBasic_[iCanvas-1];
    objfPolar = numPolarEEBasic_[iCanvas-1];

    if ( objf ) {

      meName = objf->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameNumMap[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameNumMap[iCanvas-1];

      cMap->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      objf->GetXaxis()->SetNdivisions(10, kFALSE);
      objf->GetYaxis()->SetNdivisions(10, kFALSE);
      objf->GetZaxis()->SetLabelSize(0.02);
      cMap->SetGridx();
      cMap->SetGridy();
      objf->Draw("colz");
      if ( iCanvas == 1 ) labelGrid1.Draw("text,same");
      if ( iCanvas == 2 ) labelGrid2.Draw("text,same");
      Xaxis.Draw();
      Yaxis.Draw();
      cMap->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(3.0*(Numbers::ixSectorsEE[i]-50), 3.0*(Numbers::iySectorsEE[i]-50), 3.0*(Numbers::ixSectorsEE[i+1]-50), 3.0*(Numbers::iySectorsEE[i+1]-50));
        }
      }
      cMap->Update();
      objf->GetXaxis()->SetLabelColor(0);
      objf->GetYaxis()->SetLabelColor(0);
      cMap->SaveAs(imgName.c_str());
      objf->GetXaxis()->SetLabelColor(1);
      objf->GetYaxis()->SetLabelColor(1);
    }
    
    if ( objfPolar ) {

      meName = objfPolar->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameNumPolarMap[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameNumPolarMap[iCanvas-1];

      char projXName[100];
      char projYName[100];
      sprintf(projXName,"%s_px",meName.c_str());
      imgNameNumXproj[iCanvas-1] = string(projXName) + ".png";
      sprintf(projYName,"%s_py",meName.c_str());
      imgNameNumYproj[iCanvas-1] = string(projYName) + ".png";

      obj1dX = objfPolar->ProjectionX(projXName,1,objfPolar->GetNbinsY(),"e");
      obj1dY = objfPolar->ProjectionY(projYName,1,objfPolar->GetNbinsX(),"e");

      cEne->cd();
      gStyle->SetOptStat("emr");
      obj1dX->GetXaxis()->SetNdivisions(50205, kFALSE);
      obj1dY->GetXaxis()->SetNdivisions(50206, kFALSE);

      imgName = htmlDir + imgNameNumXproj[iCanvas-1];
      obj1dX->SetStats(kTRUE);
      obj1dX->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());

      imgName = htmlDir + imgNameNumYproj[iCanvas-1];
      obj1dY->SetStats(kTRUE);
      obj1dY->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
    }
  }


  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {
    if ( imgNameEneMap[iCanvas-1].size() != 0)
      htmlFile << "<td><img src=\"" << imgNameEneMap[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }
  htmlFile << "</tr>" << endl;
  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {
    if ( imgNameNumMap[iCanvas-1].size() != 0)
      htmlFile << "<td><img src=\"" << imgNameNumMap[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;


  // projections...
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {
    if ( imgNameEneXproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameEneXproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    if ( imgNameEneYproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameEneYproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {
    if ( imgNameNumXproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameNumXproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    if ( imgNameNumYproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameNumYproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;




  // ====>  S U P E R   C L U S T E R S   <====

  // ==========================================================================
  // all Ecal Endcap 1D plots
  // ==========================================================================

  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    imgNameAll[iCanvas-1] = "";

    obj1f = allEE_[iCanvas-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
      imgNameAll[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameAll[iCanvas-1];

      cEne->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
      gPad->SetLogy(0);
    }
  }

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"sc_plots\"> <B> Super Clusters plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    if ( imgNameAll[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameAll[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // Energy profiles
  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {

    imgNameEneMap[iCanvas-1] = "";

    objp = eneEE_[iCanvas-1];
    objpPolar = enePolarEE_[iCanvas-1];

    if ( objp ) {

      meName = objp->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameEneMap[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameEneMap[iCanvas-1];

      cMap->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      objp->GetXaxis()->SetNdivisions(10, kFALSE);
      objp->GetYaxis()->SetNdivisions(10, kFALSE);
      objp->GetZaxis()->SetLabelSize(0.02);
      cMap->SetGridx();
      cMap->SetGridy();
      objp->Draw("colz");
      if ( iCanvas == 1 ) labelGrid1.Draw("text,same");
      if ( iCanvas == 2 ) labelGrid2.Draw("text,same");
      Xaxis.Draw();
      Yaxis.Draw();
      cMap->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(3.0*(Numbers::ixSectorsEE[i]-50), 3.0*(Numbers::iySectorsEE[i]-50), 3.0*(Numbers::ixSectorsEE[i+1]-50), 3.0*(Numbers::iySectorsEE[i+1]-50));
        }
      }
      cMap->Update();
      objp->GetXaxis()->SetLabelColor(0);
      objp->GetYaxis()->SetLabelColor(0);
      cMap->SaveAs(imgName.c_str());
      objp->GetXaxis()->SetLabelColor(1);
      objp->GetYaxis()->SetLabelColor(1);
    }

    if ( objpPolar ) {
      meName = objpPolar->GetName();
      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameEnePolarMap[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameEnePolarMap[iCanvas-1];

      char projXName[100];
      char projYName[100];
      sprintf(projXName,"%s_px",meName.c_str());
      imgNameEneXproj[iCanvas-1] = string(projXName) + ".png";
      sprintf(projYName,"%s_py",meName.c_str());
      imgNameEneYproj[iCanvas-1] = string(projYName) + ".png";

      obj1dX = objpPolar->ProjectionX(projXName,1,objpPolar->GetNbinsY(),"e");
      obj1dY = objpPolar->ProjectionY(projYName,1,objpPolar->GetNbinsX(),"e");

      cEne->cd();
      gStyle->SetOptStat("emr");
      obj1dX->GetXaxis()->SetNdivisions(50205, kFALSE);
      obj1dY->GetXaxis()->SetNdivisions(50206, kFALSE);

      imgName = htmlDir + imgNameEneXproj[iCanvas-1];
      obj1dX->SetStats(kTRUE);
      obj1dX->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());

      imgName = htmlDir + imgNameEneYproj[iCanvas-1];
      obj1dY->SetStats(kTRUE);
      obj1dY->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
    }
  }

  // Cluster occupancy profiles
  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {

    imgNameNumMap[iCanvas-1] = "";

    objf = numEE_[iCanvas-1];
    objfPolar = numPolarEE_[iCanvas-1];

    if ( objf ) {

      meName = objf->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameNumMap[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameNumMap[iCanvas-1];

      cMap->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      objf->GetXaxis()->SetNdivisions(10, kFALSE);
      objf->GetYaxis()->SetNdivisions(10, kFALSE);
      objf->GetZaxis()->SetLabelSize(0.02);
      cMap->SetGridx();
      cMap->SetGridy();
      objf->Draw("colz");
      if ( iCanvas == 1 ) labelGrid1.Draw("text,same");
      if ( iCanvas == 2 ) labelGrid2.Draw("text,same");
      Xaxis.Draw();
      Yaxis.Draw();
      cMap->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(3.0*(Numbers::ixSectorsEE[i]-50), 3.0*(Numbers::iySectorsEE[i]-50), 3.0*(Numbers::ixSectorsEE[i+1]-50), 3.0*(Numbers::iySectorsEE[i+1]-50));
        }
      }
      cMap->Update();
      objf->GetXaxis()->SetLabelColor(0);
      objf->GetYaxis()->SetLabelColor(0);
      cMap->SaveAs(imgName.c_str());
      objf->GetXaxis()->SetLabelColor(1);
      objf->GetYaxis()->SetLabelColor(1);
    }
    
    if ( objfPolar ) {

      meName = objfPolar->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameNumPolarMap[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameNumPolarMap[iCanvas-1];

      char projXName[100];
      char projYName[100];
      sprintf(projXName,"%s_px",meName.c_str());
      imgNameNumXproj[iCanvas-1] = string(projXName) + ".png";
      sprintf(projYName,"%s_py",meName.c_str());
      imgNameNumYproj[iCanvas-1] = string(projYName) + ".png";

      obj1dX = objfPolar->ProjectionX(projXName,1,objfPolar->GetNbinsY(),"e");
      obj1dY = objfPolar->ProjectionY(projYName,1,objfPolar->GetNbinsX(),"e");

      cEne->cd();
      gStyle->SetOptStat("emr");
      obj1dX->GetXaxis()->SetNdivisions(50205, kFALSE);
      obj1dY->GetXaxis()->SetNdivisions(50206, kFALSE);

      imgName = htmlDir + imgNameNumXproj[iCanvas-1];
      obj1dX->SetStats(kTRUE);
      obj1dX->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());

      imgName = htmlDir + imgNameNumYproj[iCanvas-1];
      obj1dY->SetStats(kTRUE);
      obj1dY->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
    }
  }


  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {
    if ( imgNameEneMap[iCanvas-1].size() != 0)
      htmlFile << "<td><img src=\"" << imgNameEneMap[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }
  htmlFile << "</tr>" << endl;
  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {
    if ( imgNameNumMap[iCanvas-1].size() != 0)
      htmlFile << "<td><img src=\"" << imgNameNumMap[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;


  // projections...
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {
    if ( imgNameEneXproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameEneXproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    if ( imgNameEneYproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameEneYproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {
    if ( imgNameNumXproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameNumXproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    if ( imgNameNumYproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameNumYproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;


  // ===========================================================================
  // Higher Level variables
  // ===========================================================================
  
  imgNameHL = "";

  obj1f = s_;

  if ( obj1f ) {

    meName = obj1f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }

    imgNameHL = meName + ".png";
    imgName = htmlDir + imgNameHL;

    cEne->cd();
    gStyle->SetOptStat("euomr");
    obj1f->SetStats(kTRUE);
    obj1f->Draw();
    cEne->Update();
    cEne->SaveAs(imgName.c_str());

  }

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"hl_plots\"> <B> Higher Level Quantities plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameHL.size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameHL << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  delete cEne;
  delete cMap;

  gStyle->SetPaintTextFormat();

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

