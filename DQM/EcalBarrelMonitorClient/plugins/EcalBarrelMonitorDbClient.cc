/*
 * \file EcalBarrelMonitorDbClient.cc
 *
 * $Date: 2010/03/27 20:30:36 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"
#endif

#include "DQM/EcalBarrelMonitorClient/interface/EcalBarrelMonitorClient.h"

class EcalBarrelMonitorDbClient: public EcalBarrelMonitorClient{

public:

/// Constructor
EcalBarrelMonitorDbClient(const edm::ParameterSet & ps) : EcalBarrelMonitorClient(ps) {};

/// Destructor
virtual ~EcalBarrelMonitorDbClient() {};

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalBarrelMonitorDbClient);

