/*
Description: "Reflags" HB/HE hits based on their ECAL, HCAL, and tracking isolation.

Original Author: John Paul Chou (Brown University)
                 Thursday, September 2, 2010
*/

#include "HBHEIsolatedNoiseReflagger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/JetReco/interface/TrackExtrapolation.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "RecoMET/METAlgorithms/interface/HcalHPDRBXMap.h"

HBHEIsolatedNoiseReflagger::HBHEIsolatedNoiseReflagger(const edm::ParameterSet& iConfig) :
  hbheLabel_(iConfig.getParameter<edm::InputTag>("hbheInput")),
  ebLabel_(iConfig.getParameter<edm::InputTag>("ebInput")),
  eeLabel_(iConfig.getParameter<edm::InputTag>("eeInput")),
  trackExtrapolationLabel_(iConfig.getParameter<edm::InputTag>("trackExtrapolationInput")),
  
  LooseHcalIsol_(iConfig.getParameter<double>("LooseHcalIsol")),
  LooseEcalIsol_(iConfig.getParameter<double>("LooseEcalIsol")),
  LooseTrackIsol_(iConfig.getParameter<double>("LooseTrackIsol")),
  TightHcalIsol_(iConfig.getParameter<double>("TightHcalIsol")),
  TightEcalIsol_(iConfig.getParameter<double>("TightEcalIsol")),
  TightTrackIsol_(iConfig.getParameter<double>("TightTrackIsol")),

  LooseRBXEne1_(iConfig.getParameter<double>("LooseRBXEne1")),
  LooseRBXEne2_(iConfig.getParameter<double>("LooseRBXEne2")),
  LooseRBXHits1_(iConfig.getParameter<int>("LooseRBXHits1")),
  LooseRBXHits2_(iConfig.getParameter<int>("LooseRBXHits2")),
  TightRBXEne1_(iConfig.getParameter<double>("TightRBXEne1")),
  TightRBXEne2_(iConfig.getParameter<double>("TightRBXEne2")),
  TightRBXHits1_(iConfig.getParameter<int>("TightRBXHits1")),
  TightRBXHits2_(iConfig.getParameter<int>("TightRBXHits2")),

  LooseHPDEne1_(iConfig.getParameter<double>("LooseHPDEne1")),
  LooseHPDEne2_(iConfig.getParameter<double>("LooseHPDEne2")),
  LooseHPDHits1_(iConfig.getParameter<int>("LooseHPDHits1")),
  LooseHPDHits2_(iConfig.getParameter<int>("LooseHPDHits2")),
  TightHPDEne1_(iConfig.getParameter<double>("TightHPDEne1")),
  TightHPDEne2_(iConfig.getParameter<double>("TightHPDEne2")),
  TightHPDHits1_(iConfig.getParameter<int>("TightHPDHits1")),
  TightHPDHits2_(iConfig.getParameter<int>("TightHPDHits2")),

  LooseDiHitEne_(iConfig.getParameter<double>("LooseDiHitEne")),
  TightDiHitEne_(iConfig.getParameter<double>("TightDiHitEne")),
  LooseMonoHitEne_(iConfig.getParameter<double>("LooseMonoHitEne")),
  TightMonoHitEne_(iConfig.getParameter<double>("TightMonoHitEne")),
  
  debug_(iConfig.getUntrackedParameter<bool>("debug",true)),
  objvalidator_(iConfig)
{
  produces<HBHERecHitCollection>();
}

HBHEIsolatedNoiseReflagger::~HBHEIsolatedNoiseReflagger()
{
}


void
HBHEIsolatedNoiseReflagger::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{
  // get the ECAL channel status map
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  evSetup.get<EcalChannelStatusRcd>().get( ecalChStatus );
  const EcalChannelStatus* dbEcalChStatus = ecalChStatus.product();

  // get the HCAL channel status map
  edm::ESHandle<HcalChannelQuality> hcalChStatus;    
  evSetup.get<HcalChannelQualityRcd>().get( hcalChStatus );
  const HcalChannelQuality* dbHcalChStatus = hcalChStatus.product();

  // get the severity level computers
  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl;
  evSetup.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComputerHndl);
  const HcalSeverityLevelComputer* hcalSevLvlComputer = hcalSevLvlComputerHndl.product();

  edm::ESHandle<EcalSeverityLevelAlgo> ecalSevLvlAlgoHndl;
  evSetup.get<EcalSeverityLevelAlgoRcd>().get(ecalSevLvlAlgoHndl);
  const EcalSeverityLevelAlgo* ecalSevLvlAlgo = ecalSevLvlAlgoHndl.product();

  // get the calotower mappings
  edm::ESHandle<CaloTowerConstituentsMap> ctcm;
  evSetup.get<CaloGeometryRecord>().get(ctcm);
  
  // get the HB/HE hits
  edm::Handle<HBHERecHitCollection> hbhehits_h;
  iEvent.getByLabel(hbheLabel_, hbhehits_h);

  // get the ECAL hits
  edm::Handle<EcalRecHitCollection> ebhits_h;
  iEvent.getByLabel(ebLabel_, ebhits_h);
  edm::Handle<EcalRecHitCollection> eehits_h;
  iEvent.getByLabel(eeLabel_, eehits_h);

  // get the tracks
  edm::Handle<std::vector<reco::TrackExtrapolation> > trackextraps_h;
  iEvent.getByLabel(trackExtrapolationLabel_, trackextraps_h);

  // set the status maps and severity level computers for the hit validator
  objvalidator_.setHcalChannelQuality(dbHcalChStatus);
  objvalidator_.setEcalChannelStatus(dbEcalChStatus);
  objvalidator_.setHcalSeverityLevelComputer(hcalSevLvlComputer);
  objvalidator_.setEcalSeverityLevelAlgo(ecalSevLvlAlgo);
  objvalidator_.setEBRecHitCollection(&(*ebhits_h));
  objvalidator_.setEERecHitCollection(&(*eehits_h));

  // organizer the hits
  PhysicsTowerOrganizer pto(iEvent, evSetup, hbhehits_h, ebhits_h, eehits_h, trackextraps_h, objvalidator_, *(ctcm.product()));
  HBHEHitMapOrganizer organizer(hbhehits_h, objvalidator_, pto);

  // get the rbxs, hpds, dihits, and monohits
  std::vector<HBHEHitMap> rbxs;
  std::vector<HBHEHitMap> hpds;
  std::vector<HBHEHitMap> dihits;
  std::vector<HBHEHitMap> monohits;
  organizer.getRBXs(rbxs, LooseRBXEne1_<TightRBXEne1_ ? LooseRBXEne1_ : TightRBXEne1_);
  organizer.getHPDs(hpds, LooseHPDEne1_<TightHPDEne1_ ? LooseHPDEne1_ : TightHPDEne1_);
  organizer.getDiHits(dihits, LooseDiHitEne_<TightDiHitEne_ ? LooseDiHitEne_ : TightDiHitEne_);
  organizer.getMonoHits(monohits, LooseMonoHitEne_<TightMonoHitEne_ ? LooseMonoHitEne_ : TightMonoHitEne_);

  if(debug_ && (rbxs.size()>0 || hpds.size()>0 || dihits.size()>0 || monohits.size()>0)) {
    edm::LogInfo("HBHEIsolatedNoiseReflagger") << "RBXs:" << std::endl;
    DumpHBHEHitMap(rbxs);
    edm::LogInfo("HBHEIsolatedNoiseReflagger") << "\nHPDs:" << std::endl;
    DumpHBHEHitMap(hpds);
    edm::LogInfo("HBHEIsolatedNoiseReflagger") << "\nDiHits:" << std::endl;
    DumpHBHEHitMap(dihits);
    edm::LogInfo("HBHEIsolatedNoiseReflagger") << "\nMonoHits:" << std::endl;
    DumpHBHEHitMap(monohits);
  }

  //  bool result=true;

  // determine which hits are noisy
  std::set<const HBHERecHit*> noisehits;
  for(int i=0; i<static_cast<int>(rbxs.size()); i++) {
    int nhits=rbxs[i].nHits();
    double ene=rbxs[i].hitEnergy();
    double trkfide=rbxs[i].hitEnergyTrackFiducial();
    double isolhcale=rbxs[i].hcalEnergySameTowers()+rbxs[i].hcalEnergyNeighborTowers();
    double isolecale=rbxs[i].ecalEnergySameTowers();
    double isoltrke=rbxs[i].trackEnergySameTowers()+rbxs[i].trackEnergyNeighborTowers();
    if((isolhcale/ene<LooseHcalIsol_ && isolecale/ene<LooseEcalIsol_ && isoltrke/ene<LooseTrackIsol_ && ((trkfide>LooseRBXEne1_ && nhits>=LooseRBXHits1_) || (trkfide>LooseRBXEne2_ && nhits>=LooseRBXHits2_))) ||
       (isolhcale/ene<TightHcalIsol_ && isolecale/ene<TightEcalIsol_ && isoltrke/ene<TightTrackIsol_ && ((trkfide>TightRBXEne1_ && nhits>=TightRBXHits1_) || (trkfide>TightRBXEne2_ && nhits>=TightRBXHits2_)))) {
      for(HBHEHitMap::hitmap_const_iterator it=rbxs[i].beginHits(); it!=rbxs[i].endHits(); ++it)
	noisehits.insert(it->first);
      //      result=false;
    }
  }

  for(int i=0; i<static_cast<int>(hpds.size()); i++) {
    int nhits=hpds[i].nHits();
    double ene=hpds[i].hitEnergy();
    double trkfide=hpds[i].hitEnergyTrackFiducial();
    double isolhcale=hpds[i].hcalEnergySameTowers()+hpds[i].hcalEnergyNeighborTowers();
    double isolecale=hpds[i].ecalEnergySameTowers();
    double isoltrke=hpds[i].trackEnergySameTowers()+hpds[i].trackEnergyNeighborTowers();
    if((isolhcale/ene<LooseHcalIsol_ && isolecale/ene<LooseEcalIsol_ && isoltrke/ene<LooseTrackIsol_ && ((trkfide>LooseHPDEne1_ && nhits>=LooseHPDHits1_) || (trkfide>LooseHPDEne2_ && nhits>=LooseHPDHits2_))) ||
       (isolhcale/ene<TightHcalIsol_ && isolecale/ene<TightEcalIsol_ && isoltrke/ene<TightTrackIsol_ && ((trkfide>TightHPDEne1_ && nhits>=TightHPDHits1_) || (trkfide>TightHPDEne2_ && nhits>=TightHPDHits2_)))) {
      for(HBHEHitMap::hitmap_const_iterator it=hpds[i].beginHits(); it!=hpds[i].endHits(); ++it)
	noisehits.insert(it->first);
      //      result=false;
    }
  }

  for(int i=0; i<static_cast<int>(dihits.size()); i++) {
    double ene=dihits[i].hitEnergy();
    double trkfide=dihits[i].hitEnergyTrackFiducial();
    double isolhcale=dihits[i].hcalEnergySameTowers()+dihits[i].hcalEnergyNeighborTowers();
    double isolecale=dihits[i].ecalEnergySameTowers();
    double isoltrke=dihits[i].trackEnergySameTowers()+dihits[i].trackEnergyNeighborTowers();
    if((isolhcale/ene<LooseHcalIsol_ && isolecale/ene<LooseEcalIsol_ && isoltrke/ene<LooseTrackIsol_ && trkfide>0.99*ene && trkfide>LooseDiHitEne_) ||
       (isolhcale/ene<TightHcalIsol_ && isolecale/ene<TightEcalIsol_ && isoltrke/ene<TightTrackIsol_ && ene>TightDiHitEne_)) {
      for(HBHEHitMap::hitmap_const_iterator it=dihits[i].beginHits(); it!=dihits[i].endHits(); ++it)
	noisehits.insert(it->first);
      //      result=false;
    }
  }
  
  for(int i=0; i<static_cast<int>(monohits.size()); i++) {
    double ene=monohits[i].hitEnergy();
    double trkfide=monohits[i].hitEnergyTrackFiducial();
    double isolhcale=monohits[i].hcalEnergySameTowers()+monohits[i].hcalEnergyNeighborTowers();
    double isolecale=monohits[i].ecalEnergySameTowers();
    double isoltrke=monohits[i].trackEnergySameTowers()+monohits[i].trackEnergyNeighborTowers();
    if((isolhcale/ene<LooseHcalIsol_ && isolecale/ene<LooseEcalIsol_ && isoltrke/ene<LooseTrackIsol_ && trkfide>0.99*ene && trkfide>LooseMonoHitEne_) ||
       (isolhcale/ene<TightHcalIsol_ && isolecale/ene<TightEcalIsol_ && isoltrke/ene<TightTrackIsol_ && ene>TightMonoHitEne_)) {
      for(HBHEHitMap::hitmap_const_iterator it=monohits[i].beginHits(); it!=monohits[i].endHits(); ++it)
	noisehits.insert(it->first);
      //      result=false;
    }
  }

  // prepare the output HBHE RecHit collection
  std::auto_ptr<HBHERecHitCollection> pOut(new HBHERecHitCollection());
  // loop over rechits, and set the new bit you wish to use
  for(HBHERecHitCollection::const_iterator it=hbhehits_h->begin(); it!=hbhehits_h->end(); ++it) {
    const HBHERecHit* hit=&(*it);
    HBHERecHit newhit(*hit);
    if(noisehits.end()!=noisehits.find(hit)) {
      newhit.setFlagField(1, HcalCaloFlagLabels::HBHEIsolatedNoise);
    }
    pOut->push_back(newhit);
  }

  iEvent.put(pOut);

  return;  
}


void HBHEIsolatedNoiseReflagger::DumpHBHEHitMap(std::vector<HBHEHitMap>& i) const
{
  for(std::vector<HBHEHitMap>::const_iterator it=i.begin(); it!=i.end(); ++it) {
    edm::LogInfo("HBHEIsolatedNoiseReflagger") << "hit energy=" << it->hitEnergy()
	      << "; # hits=" << it->nHits()
	      << "; hcal energy same=" << it->hcalEnergySameTowers()
                    << "; ecal energy same=" << it->ecalEnergySameTowers()
                  << "; track energy same=" << it->trackEnergySameTowers()
                  << "; neighbor hcal energy=" << it->hcalEnergyNeighborTowers() << std::endl;
        edm::LogInfo("HBHEIsolatedNoiseReflagger") << "hits:" << std::endl;
        for(HBHEHitMap::hitmap_const_iterator it2=it->beginHits(); it2!=it->endHits(); ++it2) {
          const HBHERecHit *hit=it2->first;
            edm::LogInfo("HBHEIsolatedNoiseReflagger") << "RBX #=" << HcalHPDRBXMap::indexRBX(hit->id())
                      << "; HPD #=" << HcalHPDRBXMap::indexHPD(hit->id())
                      << "; " << (*hit) << std::endl;
        }
  }
  return;
}


//define this as a plug-in
DEFINE_FWK_MODULE(HBHEIsolatedNoiseReflagger);
