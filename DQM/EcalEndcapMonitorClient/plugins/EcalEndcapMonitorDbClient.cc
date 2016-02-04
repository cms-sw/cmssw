/*
 * \file EcalEndcapMonitorDbClient.cc
 *
 * $Date: 2010/08/22 18:37:47 $
 * $Revision: 1.1 $
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

#include "DQM/EcalEndcapMonitorClient/interface/EcalEndcapMonitorClient.h"

class EcalEndcapMonitorDbClient: public EcalEndcapMonitorClient{

public:

/// Constructor
EcalEndcapMonitorDbClient(const edm::ParameterSet & ps) : EcalEndcapMonitorClient(ps) {};

/// Destructor
virtual ~EcalEndcapMonitorDbClient() {};

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalEndcapMonitorDbClient);

