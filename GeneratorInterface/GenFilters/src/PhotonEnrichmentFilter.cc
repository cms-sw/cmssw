// -*- C++ -*-
//
// Package:    PhotonEnrichmentFilter
// Class:      PhotonEnrichmentFilter
// 
/**\class PhotonEnrichmentFilter PhotonEnrichmentFilter.cc GeneratorInterface/GenFilters/src/PhotonEnrichmentFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Douglas Ryan Berry,512 1-008,+41227670494,
//         Created:  Mon Jul 26 10:02:34 CEST 2010
//
//


// system include files
#include <memory>
#include <vector>
#include <iostream>
#include <cmath>
#include <sstream>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace std;
//
// class declaration
//

class PhotonEnrichmentFilter : public edm::EDFilter {
public:
  explicit PhotonEnrichmentFilter(const edm::ParameterSet&);
  ~PhotonEnrichmentFilter();
  
private:
  virtual void beginJob() override ;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;
  // ----------member data ---------------------------
  bool Debug_;
  //bool Report_;
  double ClusterConeSize_;
  double EMSeedThreshold_;
  double PionSeedThreshold_;
  double GenParticleThreshold_;
  double SecondarySeedThreshold_;
  double IsoConeSize_;
  double IsolationCutOff_;
  
  double ClusterEtThreshold_;
  double ClusterEtRatio_;
  double CaloIsoEtRatio_;
  double TrackIsoEtRatio_;
  double ClusterTrackEtRatio_;

  int MaxClusterCharge_;
  int ChargedParticleThreshold_;
  int ClusterNonSeedThreshold_;
  int ClusterSeedThreshold_;
  int NumPhotons_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PhotonEnrichmentFilter::PhotonEnrichmentFilter(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  Debug_ = (bool) iConfig.getParameter<bool>("Debug");
  //Report_ = (bool) iConfig.getParameter<bool>("Report");
  ClusterConeSize_ = (double) iConfig.getParameter<double>("ClusterConeSize");
  EMSeedThreshold_ = (double) iConfig.getParameter<double>("EMSeedThreshold");
  PionSeedThreshold_ = (double) iConfig.getParameter<double>("PionSeedThreshold");
  GenParticleThreshold_ = (double) iConfig.getParameter<double>("GenParticleThreshold");
  SecondarySeedThreshold_ = (double) iConfig.getParameter<double>("SecondarySeedThreshold");
  IsoConeSize_ = (double) iConfig.getParameter<double>("IsoConeSize");
  IsolationCutOff_ = (double) iConfig.getParameter<double>("IsolationCutOff");
  
  ClusterEtThreshold_ = (double) iConfig.getParameter<double>("ClusterEtThreshold");
  ClusterEtRatio_ = (double) iConfig.getParameter<double>("ClusterEtRatio");
  CaloIsoEtRatio_ = (double) iConfig.getParameter<double>("CaloIsoEtRatio");
  TrackIsoEtRatio_ = (double) iConfig.getParameter<double>("TrackIsoEtRatio");
  ClusterTrackEtRatio_ = (double) iConfig.getParameter<double>("ClusterTrackEtRatio");

  MaxClusterCharge_ = (int) iConfig.getParameter<int>("MaxClusterCharge");
  ChargedParticleThreshold_ = (int) iConfig.getParameter<int>("ChargedParticleThreshold");
  ClusterNonSeedThreshold_ = (int) iConfig.getParameter<int>("ClusterNonSeedThreshold");
  ClusterSeedThreshold_ = (int) iConfig.getParameter<int>("ClusterSeedThreshold");
  NumPhotons_ = (int) iConfig.getParameter<int>("NumPhotons");
}


PhotonEnrichmentFilter::~PhotonEnrichmentFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
PhotonEnrichmentFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace reco;
  
  Handle<GenParticleCollection> GenParticles; 
  iEvent.getByLabel("genParticles", GenParticles); 

  bool FilterResult = false;
  double etalimit = 2.4;
  //EventNumber_t eventNumber = iEvent.id().event();
  //RunNumber_t runNumber = iEvent.id().run();
  vector <pair <GenParticle, GenParticle> > ClusterSeeds;
  int NumPassClusters = 0;
  
  for (GenParticleCollection::const_iterator itGenParticles = GenParticles->begin(); itGenParticles != GenParticles->end(); ++itGenParticles) {
    double CandidateEt = itGenParticles->et();
    double CandidateEta = itGenParticles->eta();
    double CandidatePDGID = abs(itGenParticles->pdgId());
    double CandidatePhi = itGenParticles->phi();

    GenParticle SecondarySeed = *itGenParticles;
    
    if ((CandidatePDGID==22 || CandidatePDGID==11
         || CandidatePDGID==211 || CandidatePDGID==310 || CandidatePDGID==130
         || CandidatePDGID==321 || CandidatePDGID==2112 || CandidatePDGID==2212 || CandidatePDGID==3122)
         && abs(CandidateEta)<etalimit && CandidateEt>EMSeedThreshold_) {
      bool newseed=true;

      if ((CandidatePDGID==211 || CandidatePDGID==321
           || CandidatePDGID==2112 || CandidatePDGID==2212 || CandidatePDGID==3122)
           && CandidateEt<PionSeedThreshold_) newseed=false;

      if (newseed) {
        for (GenParticleCollection::const_iterator checkGenParticles = GenParticles->begin(); checkGenParticles != GenParticles->end(); ++checkGenParticles) {
          double GenEt = checkGenParticles->et();
          double GenEta = checkGenParticles->eta();
          double GenPDGID = abs(checkGenParticles->pdgId());
          double GenPhi = checkGenParticles->phi();
          double dR = deltaR2(CandidateEta,CandidatePhi,GenEta,GenPhi);
          
          if ((((GenPDGID==22 || GenPDGID==11 || GenPDGID==310 || GenPDGID==130) && GenEt>CandidateEt)
               || ((GenPDGID==211 || GenPDGID==321 || GenPDGID==2112 || GenPDGID==2212 || GenPDGID==3122) && GenEt>CandidateEt && GenEt>PionSeedThreshold_))
               && dR<ClusterConeSize_) newseed=false;

          if ((GenPDGID==211 || GenPDGID==321 || GenPDGID==2112 || GenPDGID==2212 || GenPDGID==3122)
              && (GenEt>SecondarySeed.et() || SecondarySeed.et()==CandidateEt)
              && GenEt>SecondarySeedThreshold_ && GenEt<CandidateEt && dR<ClusterConeSize_) SecondarySeed=*checkGenParticles;
        }
      }
      
      if (newseed) ClusterSeeds.push_back(make_pair(*itGenParticles,SecondarySeed));
    }
  }
   
  for (vector<pair <GenParticle, GenParticle> >::const_iterator itClusterSeeds = ClusterSeeds.begin(); itClusterSeeds != ClusterSeeds.end(); ++itClusterSeeds) {
    double CaloIsoEnergy = 0;
    double ClusterEnergy = 0;
    double ClusterTrackEnergy = 0;
    double ClusterEta = itClusterSeeds->first.eta();
    double ClusterPhi = itClusterSeeds->first.phi();
    double TrackIsoEnergy = 0;
    double ClusterTotalCharge = 0;
    double ClusterTotalEnergy = 0;

    double SecondaryEta = itClusterSeeds->second.eta();
    double SecondaryPhi = itClusterSeeds->second.phi();
    
    int NumChargesInConeCounter = 0;
    int NumSeedsInConeCounter = 0;
    int NumNonSeedsInConeCounter = 0;

    for(GenParticleCollection::const_iterator itGenParticles = GenParticles->begin(); itGenParticles != GenParticles->end(); ++itGenParticles) { 
      bool TheSecondarySeed = false;
      bool TheSeedParticle = false;
      double GenCharge = itGenParticles->charge();
      double GenEt = itGenParticles->et();
      double GenEta = itGenParticles->eta();
      double GenPDGID = abs(itGenParticles->pdgId());
      double GenPhi = itGenParticles->phi();
      double GenStatus = itGenParticles->status();
      double dR = deltaR2(GenEta,GenPhi,ClusterEta,ClusterPhi);

      if (ClusterEta==GenEta && ClusterPhi==GenPhi) TheSeedParticle=true;
      if (SecondaryEta==GenEta && SecondaryPhi==GenPhi && !(SecondaryEta==ClusterEta && SecondaryPhi==ClusterPhi)) TheSecondarySeed=true;
      if (GenStatus==1 && GenEt>GenParticleThreshold_) {

        if (dR<ClusterConeSize_) {
          if (GenCharge!=0 && !TheSeedParticle && !TheSecondarySeed) {
            NumChargesInConeCounter++;
            ClusterTotalCharge += GenCharge;
            ClusterTrackEnergy += GenEt;
          }
          if (GenPDGID!=12 && GenPDGID!=14 && GenPDGID!=16) ClusterTotalEnergy+=GenEt;
          if (GenPDGID!=12 && GenPDGID!=14 && GenPDGID!=16 && GenPDGID!=22 && GenPDGID!=11 && GenPDGID!=310 && GenPDGID!=130 && !TheSeedParticle && !TheSecondarySeed) NumNonSeedsInConeCounter++;
          if (GenPDGID==22 || GenPDGID==11 || GenPDGID==310 || GenPDGID==130 || TheSeedParticle || TheSecondarySeed) {
            NumSeedsInConeCounter++;
            ClusterEnergy += GenEt;
          }
        }
        if (dR<IsoConeSize_ && dR>ClusterConeSize_) {
          if (GenCharge!=0) TrackIsoEnergy += GenEt;
          if (GenPDGID>100 || GenPDGID==22 || GenPDGID==11) CaloIsoEnergy += GenEt;
        }

      }

    }

    if (Debug_) cout << "ClusterEnergy: " << ClusterEnergy << " | CaloIsoEtRatio: " << CaloIsoEnergy/ClusterEnergy << " | TrackIsoEtRatio: " << TrackIsoEnergy/ClusterEnergy << " | ClusterTrackEtRatio: " << ClusterTrackEnergy/ClusterEnergy << " | ClusterEtRatio: " << ClusterEnergy/ClusterTotalEnergy << " | ChargedParticles: " << NumChargesInConeCounter << " | ClusterNonSeeds: " << NumNonSeedsInConeCounter << " | ClusterSeeds: " << NumSeedsInConeCounter << endl;
    if ((ClusterEnergy<IsolationCutOff_
         && ClusterEnergy>ClusterEtThreshold_
         && ClusterEnergy/ClusterTotalEnergy>ClusterEtRatio_
         && CaloIsoEnergy/ClusterEnergy<CaloIsoEtRatio_
         && TrackIsoEnergy/ClusterEnergy<TrackIsoEtRatio_
         && ClusterTrackEnergy/ClusterEnergy<ClusterTrackEtRatio_
         && abs(ClusterTotalCharge)<MaxClusterCharge_
         && NumChargesInConeCounter<ChargedParticleThreshold_
         && NumNonSeedsInConeCounter<ClusterNonSeedThreshold_
         && NumSeedsInConeCounter<ClusterSeedThreshold_) ||
        (ClusterEnergy>=IsolationCutOff_
         && ClusterEnergy>ClusterEtThreshold_
         && ClusterEnergy/ClusterTotalEnergy>ClusterEtRatio_
         //&& CaloIsoEnergy/ClusterEnergy<CaloIsoEtRatio_
         //&& TrackIsoEnergy/ClusterEnergy<TrackIsoEtRatio_
         && ClusterTrackEnergy/ClusterEnergy<ClusterTrackEtRatio_
         && abs(ClusterTotalCharge)<MaxClusterCharge_
         && NumChargesInConeCounter<ChargedParticleThreshold_
         && NumNonSeedsInConeCounter<ClusterNonSeedThreshold_
         //&& NumSeedsInConeCounter<ClusterSeedThreshold_
         )
        ) NumPassClusters++;

    /* if (Report_) {
      Handle<PhotonCollection> Photons;
      iEvent.getByLabel("photons", Photons);

      for (PhotonCollection::const_iterator itPhotons = Photons->begin(); itPhotons != Photons->end(); ++itPhotons) {

        if (itPhotons->pt()>15. && abs(itPhotons->eta())<2.5 && itPhotons->ecalRecHitSumEtConeDR04()<4.2+0.003*itPhotons->pt() &&
            itPhotons->hcalTowerSumEtConeDR04()<2.2+0.001*itPhotons->pt() && itPhotons->trkSumPtHollowConeDR04()<2.0+0.001*itPhotons->pt() &&
            itPhotons->hadronicOverEm()<0.05 && itPhotons->isEB() && itPhotons->sigmaIetaIeta()<0.010) {

          double PhotonEta = itPhotons->eta();
          double PhotonPhi = itPhotons->phi();
          double PhotonClusterDr = deltaR2(PhotonEta,PhotonPhi,ClusterEta,ClusterPhi);
        
          if (PhotonClusterDr<0.1 && (itPhotons->ecalRecHitSumEtConeDR04()>3 && itPhotons->ecalRecHitSumEtConeDR04()<5)) {

            if (!((ClusterEnergy<IsolationCutOff_
                 && ClusterEnergy>ClusterEtThreshold_
                 && ClusterEnergy/ClusterTotalEnergy>ClusterEtRatio_
                 && CaloIsoEnergy/ClusterEnergy<CaloIsoEtRatio_
                 && TrackIsoEnergy/ClusterEnergy<TrackIsoEtRatio_
                 && ClusterTrackEnergy/ClusterEnergy<ClusterTrackEtRatio_
                 && abs(ClusterTotalCharge)<MaxClusterCharge_
                 && NumChargesInConeCounter<ChargedParticleThreshold_
                 && NumNonSeedsInConeCounter<ClusterNonSeedThreshold_
                 && NumSeedsInConeCounter<ClusterSeedThreshold_) ||
                (ClusterEnergy>=IsolationCutOff_
                 && ClusterEnergy>ClusterEtThreshold_
                 && ClusterEnergy/ClusterTotalEnergy>ClusterEtRatio_
                 //&& CaloIsoEnergy/ClusterEnergy<CaloIsoEtRatio_
                 //&& TrackIsoEnergy/ClusterEnergy<TrackIsoEtRatio_
                 && ClusterTrackEnergy/ClusterEnergy<ClusterTrackEtRatio_
                 && abs(ClusterTotalCharge)<MaxClusterCharge_
                 //&& NumChargesInConeCounter<ChargedParticleThreshold_
                 //&& NumNonSeedsInConeCounter<ClusterNonSeedThreshold_
                 //&& NumSeedsInConeCounter<ClusterSeedThreshold_
                 ))
                ) {
              cout << "Event " << eventNumber << " in run " << runNumber << " has a photon pt of " << itPhotons->pt() << " and failed because:" << endl;
              //cout << "ClusterEnergy: " << ClusterEnergy << " | CaloIsoEtRatio: " << CaloIsoEnergy/ClusterEnergy << " | TrackIsoEtRatio: " << TrackIsoEnergy/ClusterEnergy << " | ClusterTrackEtRatio: " << ClusterTrackEnergy/ClusterEnergy << " | ClusterEtRatio: " << ClusterEnergy/ClusterTotalEnergy << " | ChargedParticles: " << NumChargesInConeCounter << " | ClusterNonSeeds: " << NumNonSeedsInConeCounter << " | ClusterSeeds: " << NumSeedsInConeCounter << endl;
            }
            if (ClusterEnergy<=ClusterEtThreshold_) cout << "ClusterEnergy: " << ClusterEnergy << endl;
            if (ClusterEnergy/ClusterTotalEnergy<=ClusterEtRatio_) cout << "ClusterEtRatio: " << ClusterEnergy/ClusterTotalEnergy << endl;
            if (ClusterEnergy<IsolationCutOff_ && CaloIsoEnergy/ClusterEnergy>=CaloIsoEtRatio_) cout << "CaloIsoEtRatio: " << CaloIsoEnergy/ClusterEnergy << endl;
            if (ClusterEnergy<IsolationCutOff_ && TrackIsoEnergy/ClusterEnergy>=TrackIsoEtRatio_) cout << "TrackIsoEtRatio: " << TrackIsoEnergy/ClusterEnergy << endl;
            if (ClusterTrackEnergy/ClusterEnergy>=ClusterTrackEtRatio_) cout << "ClusterTrackEtRatio: " << ClusterTrackEnergy/ClusterEnergy << endl;
            if (abs(ClusterTotalCharge)>=MaxClusterCharge_) cout << "ClusterTotalCharge: " << ClusterTotalCharge << endl;
            if (ClusterEnergy<IsolationCutOff_ && NumChargesInConeCounter>=ChargedParticleThreshold_) cout << "ChargedParticles: " << NumChargesInConeCounter << endl;
            if (ClusterEnergy<IsolationCutOff_ && NumNonSeedsInConeCounter>=ClusterNonSeedThreshold_) cout << "ClusterNonSeeds: " << NumNonSeedsInConeCounter << endl;
            if (ClusterEnergy<IsolationCutOff_ && NumSeedsInConeCounter>=ClusterSeedThreshold_) cout << "ClusterSeeds: " << NumSeedsInConeCounter << endl;
          
          }

        }
        
      }
      
    } */
    
  }

  if (NumPassClusters>=NumPhotons_) FilterResult=true;
  return FilterResult;
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
PhotonEnrichmentFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PhotonEnrichmentFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PhotonEnrichmentFilter);
