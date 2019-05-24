// -*- C++ -*-
//
// Package:    L1Trigger/L1TTrackMatch
// Class:      L1TkBsCandidateProducer
// 
/**\class L1TkBsCandidateProducer

 Description: Producer of L1 Bs Candidates, Input: L1 Tracks

 Implementation:
     [Notes on implementation]
*/
//
// Authors:  R. Bhattacharya, S. Dutta and S. Sarkar
// Created:  Fri May 10 16:25:00 CET 2019
// $Id$
//
//
// -*- C++ -*-
//
//
// system include files
#include <memory>
#include <string>
#include <array>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkPhiCandidate.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPhiCandidateFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkBsCandidate.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkBsCandidateFwd.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TMath.h"

using namespace l1t;
//
// class declaration
//
class L1TkBsCandidateProducer: public edm::EDProducer {
public:
  using L1TTTrackType = TTTrack<Ref_Phase2TrackerDigi_>;
  using L1TTTrackCollectionType = std::vector<L1TTTrackType>;
  using L1TTStubCollection 
    = std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>;

  explicit L1TkBsCandidateProducer(const edm::ParameterSet&);
  ~L1TkBsCandidateProducer();
   
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  int findPhiCandidates(const edm::Handle<L1TTTrackCollectionType>& trkHandle, 
			const TrackerTopology* tTopo,
			L1TkPhiCandidateCollection& list) const;
  bool selectTrack(L1TTTrackCollectionType::const_iterator itrk, 
		   const TrackerTopology* tTopo) const;
  void stubInfo(L1TTTrackCollectionType::const_iterator itrk, 
		const TrackerTopology* tTopo, 
		int& nStub, 
		int& nStub_SS, 
		int& nStub_PS) const;
  static void deltaPos(L1TTTrackCollectionType::const_iterator itrk, 
		       L1TTTrackCollectionType::const_iterator jtrk, 
		       double& dxy, double& dz);
  static void deltaPos(const L1TkPhiCandidate& phia, 
		       const L1TkPhiCandidate& phib, 
		       double& dxy, double& dz);

  // ----------member data ---------------------------
  bool verbose_ {true};
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> trackToken_;

  double trkEtaMax_,
         trkPtMin_;
    bool applyTrkQuality_;
  double trkChi2Max_;
     int trkLayersMin_,
         trkPSLayersMin_;
  double trkPairDzMax_,
         trkPairDxyMax_,
         phiMassMin_,
         phiMassMax_,
         phiPairDzMax_,
         phiPairDxyMax_,
         phiPairDrMin_,
         phiPairDrMax_,
         phiTrkPairDrMax_,
         bsMassMin_,
         bsMassMax_;

  std::string label_;    
  int evcounters[8];
};
//
// constructors and destructor
//
L1TkBsCandidateProducer::L1TkBsCandidateProducer(const edm::ParameterSet& iConfig) :
  verbose_(iConfig.getParameter<bool>("verbose")),
  trackToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
  trkEtaMax_(iConfig.getParameter<double>("TrackEtaMax")),
  trkPtMin_(iConfig.getParameter<double>("TrackPtMin")),
  applyTrkQuality_(iConfig.getParameter<bool>("ApplyTrackQuality")),
  trkChi2Max_(iConfig.getParameter<double>("TrackChi2Max")),
  trkLayersMin_(iConfig.getParameter<int>("TrackLayersMin")),
  trkPSLayersMin_(iConfig.getParameter<int>("TrackPSLayersMin")),
  trkPairDzMax_(iConfig.getParameter<double>("TrackPairDzMax")),
  trkPairDxyMax_(iConfig.getParameter<double>("TrackPairDxyMax")),
  phiMassMin_(iConfig.getParameter<double>("PhiMassMin")),
  phiMassMax_(iConfig.getParameter<double>("PhiMassMax")),
  phiPairDzMax_(iConfig.getParameter<double>("PhiPairDzMax")),
  phiPairDxyMax_(iConfig.getParameter<double>("PhiPairDxyMax")),
  phiPairDrMin_(iConfig.getParameter<double>("PhiPairDrMin")),
  phiPairDrMax_(iConfig.getParameter<double>("PhiPairDrMax")),
  phiTrkPairDrMax_(iConfig.getParameter<double>("PhiTrkPairDrMax")),
  bsMassMin_(iConfig.getParameter<double>("BsMassMin")),
  bsMassMax_(iConfig.getParameter<double>("BsMassMax")),
  label_(iConfig.getParameter<std::string>("label"))  // label of the collection produced  
{
  produces<L1TkBsCandidateCollection>(label_);
}
L1TkBsCandidateProducer::~L1TkBsCandidateProducer() {
}
// ------------ method called to produce the data  ------------
void L1TkBsCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::unique_ptr<L1TkBsCandidateCollection> collection(new L1TkBsCandidateCollection);

  ++evcounters[0];
  // the L1Tracks
  edm::Handle<L1TTTrackCollectionType> trkHandle;
  bool found_coll = iEvent.getByToken(trackToken_, trkHandle);
  if (found_coll && trkHandle.isValid()) {
    edm::ESHandle<TrackerTopology> tTopoHandle;
    iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
    const TrackerTopology* tTopo = tTopoHandle.product();

    int ntrk = 0;
    for (auto it = trkHandle->begin(); it != trkHandle->end(); ++it)
      if (selectTrack(it, tTopo)) ++ntrk;

    if (ntrk > 3) ++evcounters[1];

    // First pass: build Phi candidates from L1 Tracks (assigning kaon mass to each track)
    L1TkPhiCandidateCollection phiColl;
    int nCand = findPhiCandidates(trkHandle, tTopo, phiColl); 
    if (nCand > 1) {
      ++evcounters[2];
      // Second pass: build Bs candidates from recontructed Phi candidates
      int icounters[] { 0, 0, 0, 0, 0 };
      for (size_t i = 0; i < phiColl.size(); ++i) {
	const auto& phia = phiColl[i]; 
	const auto& trka_1 = phia.getTrkPtr(0);
	const auto& trka_2 = phia.getTrkPtr(1);
	double drTrkPhia = phia.dRTrkPair();
	
	for (size_t j = i+1; j < phiColl.size(); ++j) {
	  const auto& phib = phiColl[j]; 
	  const auto& trkb_1 = phib.getTrkPtr(0);
	  const auto& trkb_2 = phib.getTrkPtr(1);
	  
	  // The same track should not end up in both the Phi's under consideration
	  if (trka_1 == trkb_1 || trka_1 == trkb_2 || trka_2 == trkb_1 || trka_2 == trkb_2) continue;
	  ++icounters[0];
	  
	  // The pair of Phi's must come from the same vertex
	  double dxy, dz;
	  L1TkBsCandidateProducer::deltaPos(phia, phib, dxy, dz);      
	  if (std::fabs(dz) > phiPairDzMax_ || std::fabs(dxy) > phiPairDxyMax_) continue;
	  ++icounters[1];
	  
	  double dr = reco::deltaR(phia.p4(), phib.p4());
	  if (dr < phiPairDrMin_ || dr > phiPairDrMax_) continue;
	  ++icounters[2];
	  
	  double drTrkPhib = phib.dRTrkPair();
	  if (drTrkPhia > phiTrkPairDrMax_ || drTrkPhib > phiTrkPairDrMax_) continue;

	  ++icounters[3];
	  
	  math::XYZTLorentzVector bsv = phia.p4() + phib.p4();
	  double mass = bsv.M();    
	  if (mass < bsMassMin_ || mass > bsMassMax_) continue;
	  ++icounters[4];
	  L1TkBsCandidate bsCandidate(bsv, phia, phib);
	  collection->push_back(bsCandidate);
	}
      }
      for (size_t i = 0; i < 5; ++i)
	if (icounters[i] > 0) ++evcounters[3+i];
    }
  }
  else {
    edm::LogError("L1TkBsCandidateProducer")
      << "\nWarning: L1TTTrackCollectionType not found in the event. Exit."
      << std::endl;
  }
  iEvent.put(std::move(collection), label_);
}
int L1TkBsCandidateProducer::findPhiCandidates(const edm::Handle<L1TTTrackCollectionType>& trkHandle, 
					       const TrackerTopology* tTopo,                               
					       L1TkPhiCandidateCollection& list) const
{
  size_t itrk = 0;
  for (auto it = trkHandle->begin(); it != trkHandle->end(); ++it, itrk++) {
    edm::Ptr<L1TTTrackType> trkPtr1(trkHandle, itrk);

    if (!selectTrack(it, tTopo)) continue;
    math::PtEtaPhiMLorentzVector trkv1(it->getMomentum().perp(), 
				       it->getMomentum().eta(), 
				       it->getMomentum().phi(), 
				       L1TkPhiCandidate::kmass);

    size_t jtrk = 0;
    for (auto jt = trkHandle->begin(); jt != trkHandle->end(); ++jt, jtrk++) {
      edm::Ptr<L1TTTrackType> trkPtr2(trkHandle, jtrk);

      //if (trkPtr2 == trkPtr1) continue;
      if (jtrk <= itrk) continue;
      if (!selectTrack(jt, tTopo)) continue;

      // Oppositely charged tracks
      if (std::signbit(it->getRInv(5)) == std::signbit(jt->getRInv(5))) continue;

      // Track pair must come from the same vertex
      double dxy, dz;
      L1TkBsCandidateProducer::deltaPos(it, jt, dxy, dz);

      // Apply |dz| and dxy cuts in the given order
      if (std::fabs(dz) > trkPairDzMax_ || std::fabs(dxy) > trkPairDxyMax_) continue;
      math::PtEtaPhiMLorentzVector trkv2(jt->getMomentum().perp(), 
					 jt->getMomentum().eta(), 
					 jt->getMomentum().phi(), 
					 L1TkPhiCandidate::kmass);

      // Select mass window
      math::XYZTLorentzVector p4(trkv1.Px() + trkv2.Px(), 
				 trkv1.Py() + trkv2.Py(), 
				 trkv1.Pz() + trkv2.Pz(), 
				 trkv1.T()  + trkv2.T());
      double mass = p4.M();
      if (mass < phiMassMin_ || mass > phiMassMax_) continue;

      L1TkPhiCandidate cand(p4, trkPtr1, trkPtr2);
      list.push_back(cand);
    }
  }
  return list.size();
}
bool L1TkBsCandidateProducer::selectTrack(L1TTTrackCollectionType::const_iterator itrk, 
					  const TrackerTopology* tTopo) const 
{
  if (std::fabs(itrk->getMomentum(5).eta()) > trkEtaMax_ ||
      itrk->getMomentum(5).perp() < trkPtMin_) return false;
  
  if (applyTrkQuality_) {
    int nStub, nStub_SS, nStub_PS;
    stubInfo(itrk, tTopo, nStub, nStub_SS, nStub_PS);
    if (itrk->getChi2Red(5) > trkChi2Max_ || 
	nStub < trkLayersMin_ ||
	nStub_PS < trkPSLayersMin_) return false;
  }
  return true;
}
void L1TkBsCandidateProducer::stubInfo(L1TTTrackCollectionType::const_iterator itrk, 
				       const TrackerTopology* tTopo, 
				       int& nStub, 
				       int& nStub_SS, 
				       int& nStub_PS) const 
{
  L1TTStubCollection stubs = itrk->getStubRefs();
  nStub = stubs.size();
  nStub_PS = 0; 
  nStub_SS = 0;
  for (auto it = stubs.begin(); it != stubs.end(); ++it) {
    DetId detid = (*it)->getDetId();
    if (detid.det() != DetId::Detector::Tracker) continue;
    if (detid.subdetId() == StripSubdetector::TOB) {
      (tTopo->tobLayer(detid) <= 3) ? nStub_PS++ : nStub_SS++;
    }	
    else if (detid.subdetId() == StripSubdetector::TID) {
      (tTopo->tidRing(detid) <= 9) ? nStub_PS++ : nStub_SS++;
    }
  }
}
void L1TkBsCandidateProducer::deltaPos(L1TTTrackCollectionType::const_iterator itrk, 
				       L1TTTrackCollectionType::const_iterator jtrk, 
				       double& dxy, double& dz) 
{
  dxy = std::sqrt(std::pow(itrk->getPOCA(5).x() - jtrk->getPOCA(5).x(), 2) 
		+ std::pow(itrk->getPOCA(5).y() - jtrk->getPOCA(5).y(), 2));
  dz  = itrk->getPOCA(5).z() - jtrk->getPOCA(5).z();
}
void L1TkBsCandidateProducer::deltaPos(const L1TkPhiCandidate& phia, 
				       const L1TkPhiCandidate& phib, 
				       double& dxy, double& dz) 
{
  dxy = std::sqrt(std::pow(phia.vx() - phib.vx(), 2) + std::pow(phia.vy() - phib.vy(), 2));
  dz  = phia.vz() - phib.vz();
}
// ------------ method called once each job just before starting event loop  ------------
void L1TkBsCandidateProducer::beginJob() {
  for (size_t i = 0; i < 8; ++i) evcounters[i] = 0;
}

// ------------ method called once each job just after ending the event loop  ------------
void L1TkBsCandidateProducer::endJob() {
  if (verbose_) {
    for (size_t i = 0; i < 8; ++i) 
      std::cout << setw(10) << evcounters[i] << std::endl;
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TkBsCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(L1TkBsCandidateProducer);
