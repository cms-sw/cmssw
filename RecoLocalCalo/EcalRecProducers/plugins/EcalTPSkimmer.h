#ifndef RecoLocalCalo_EcalRecProducers_EcalTPSkimmer_hh
#define RecoLocalCalo_EcalRecProducers_EcalTPSkimmer_hh

/** \class EcalTPSkimmer
 *   produce a subset of TP information
 *
 *  \author Federico Ferri, CEA/Saclay Irfu/SPP
 *
 **/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class EcalTPSkimmer : public edm::stream::EDProducer<> {
public:
  explicit EcalTPSkimmer(const edm::ParameterSet& ps);
  ~EcalTPSkimmer() override;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  bool alreadyInserted(EcalTrigTowerDetId ttId);
  void insertTP(EcalTrigTowerDetId ttId, edm::Handle<EcalTrigPrimDigiCollection>& in, EcalTrigPrimDigiCollection& out);

  std::string tpCollection_;

  bool skipModule_;
  bool doBarrel_;
  bool doEndcap_;

  std::vector<uint32_t> chStatusToSelectTP_;
  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap_;
  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> ttMapToken_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> chStatusToken_;

  std::set<EcalTrigTowerDetId> insertedTP_;

  edm::EDGetTokenT<EcalTrigPrimDigiCollection> tpInputToken_;

  std::string tpOutputCollection_;
};

#endif
