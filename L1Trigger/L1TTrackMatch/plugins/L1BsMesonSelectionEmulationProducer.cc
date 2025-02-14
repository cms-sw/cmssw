// -*- C++ -*-
//
// Package:    L1Trigger/L1TTrackMatch
// Class:      L1BsMesonSelectionEmulationProducer
//
/**\class L1BsMesonSelectionProducer L1BsMesonSelectionProducer.cc L1Trigger/L1TTrackMatch/plugins/L1BsMesonSelectionProducer.cc                                  
 Description: Build Bs meson candidates from Phi collection
 Implementation:                                                                                                                                       
     Input:                                                                                                                                             
         l1t::TkLightMesonWordCollection - A collection of reconstructed Phi candidates                                                                    
     Output:                                                                                                                                 
         l1t::TkLightMesonWordCollection - A collection of reconstructed Bs candidates                                                                        
*/
// ----------------------------------------------------------------------------
// Authors: Alexx Perloff, Pritam Palit (original version, 2021),
//          Sweta Baradia, Suchandra Dutta, Subir Sarkar (February 2025)
//-----------------------------------------------------------------------------

// system include files
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <TMath.h>
#include <cmath>
#include <bitset>
#include <array>
// Xilinx HLS includes
#include <ap_fixed.h>
#include <ap_int.h>
#include <stdio.h>
#include <cassert>
#include <cstdlib>

// user include files
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidateFwd.h"
#include "DataFormats/L1Trigger/interface/TkLightMesonWord.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
//#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "L1TrackWordUnpacker.h"

// class declaration
using namespace std;
using namespace edm;
using namespace l1t;

//class L1BsMesonSelectionEmulationProducer : public edm::one::EDProducer<edm::one::SharedResources> {
class L1BsMesonSelectionEmulationProducer : public edm::global::EDProducer<> {
public:
  using L1TTTrackType            = TTTrack<Ref_Phase2TrackerDigi_>;
  using TTTrackCollection        = std::vector<L1TTTrackType>;
  //using TTTrackRef               = edm::Ref<TTTrackCollection>; 
  using TTTrackRefCollection     = edm::RefVector<TTTrackCollection>;
  using TTTrackCollectionHandle  = edm::Handle<TTTrackRefCollection>;
  //using TTTrackRefCollectionUPtr = std::unique_ptr<TTTrackRefCollection>; 
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  explicit L1BsMesonSelectionEmulationProducer(const edm::ParameterSet&);
  ~L1BsMesonSelectionEmulationProducer();
  static constexpr double KaonMass = 0.493677 ;
  size_t bsSize = 10;

  double ETAPHI_LSB = M_PI / (1 << 12);
  double Z0_LSB = 0.05;

private:
  // ----------constants, enums and typedefs --------- 
  using TkLightMesonWordCollectionHandle = edm::Handle<TkLightMesonWordCollection>;
  
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  //void produce(edm::Event&, const edm::EventSetup&) override;
  
  // ----------selectors -----------------------------
  // Based on recommendations from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenericSelectors
  
  // ----------member data ---------------------------
  const edm::EDGetTokenT<TkLightMesonWordCollection> l1PhiMesonWordToken_;
  const edm::EDGetTokenT<TTTrackRefCollection> posTrackToken_;
  const edm::EDGetTokenT<TTTrackRefCollection> negTrackToken_;
  const std::string outputCollectionName_;
  int debug_;
  const edm::ParameterSet cutSet_;
  const double phiPairdzMax_, phiPairdRMin_, phiPairdRMax_, phiPairMMin_, phiPairMMax_, bsPtMin_;
};

//
// constructors and destructor
//
L1BsMesonSelectionEmulationProducer::L1BsMesonSelectionEmulationProducer(const edm::ParameterSet& iConfig)
  : l1PhiMesonWordToken_(consumes<TkLightMesonWordCollection>(iConfig.getParameter<edm::InputTag>("l1PhiMesonWordInputTag"))),
    posTrackToken_(consumes<TTTrackRefCollection>(iConfig.getParameter<edm::InputTag>("posTrackInputTag"))),
    negTrackToken_(consumes<TTTrackRefCollection>(iConfig.getParameter<edm::InputTag>("negTrackInputTag"))),
    outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName")),
    debug_(iConfig.getParameter<int>("debug")),
    cutSet_(iConfig.getParameter<edm::ParameterSet>("cutSet")),
    phiPairdzMax_(cutSet_.getParameter<double>("phiPairdzMax")),
    phiPairdRMin_(cutSet_.getParameter<double>("phiPairdRMin")),
    phiPairdRMax_(cutSet_.getParameter<double>("phiPairdRMax")),
    phiPairMMin_(cutSet_.getParameter<double>("phiPairMMin")),
    phiPairMMax_(cutSet_.getParameter<double>("phiPairMMax")),
    bsPtMin_(cutSet_.getParameter<double>("bsPtMin"))
{
  produces<l1t::TkLightMesonWordCollection>(outputCollectionName_);
}

L1BsMesonSelectionEmulationProducer::~L1BsMesonSelectionEmulationProducer() {}
//
// member functions
//
// ------------ method called to produce the data  ------------
void L1BsMesonSelectionEmulationProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
//void L1BsMesonSelectionEmulationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)  {

  auto L1BsMesonEmulationOutput = std::make_unique<l1t::TkLightMesonWordCollection>();

  TkLightMesonWordCollectionHandle l1PhiMesonWordHandle;
  iEvent.getByToken(l1PhiMesonWordToken_, l1PhiMesonWordHandle);
  size_t nPhiMesonOutputApproximate = l1PhiMesonWordHandle->size();
  if(nPhiMesonOutputApproximate < 2) return;
  
  TTTrackCollectionHandle posTrackHandle;
  iEvent.getByToken(posTrackToken_, posTrackHandle);
  
  TTTrackCollectionHandle negTrackHandle;
  iEvent.getByToken(negTrackToken_, negTrackHandle);

  L1BsMesonEmulationOutput->reserve(bsSize);

  for (size_t i = 0; i < nPhiMesonOutputApproximate; i++) {
    const auto& phiMesonWord1 = l1PhiMesonWordHandle->at(i);        
    double ptPhi1   = phiMesonWord1.pt();
    double etaPhi1  = phiMesonWord1.glbeta();
    double phiPhi1  = phiMesonWord1.glbphi();
    double z0Phi1   = phiMesonWord1.z0();
    double massPhi1 = phiMesonWord1.mass();

    unsigned int pIndexPhi1 = phiMesonWord1.firstIndex();
    unsigned int nIndexPhi1 = phiMesonWord1.secondIndex();

    double pxPhi1 = ptPhi1 * cos(phiPhi1);
    double pyPhi1 = ptPhi1 * sin(phiPhi1);
    double pzPhi1 = ptPhi1 * sinh(etaPhi1);
    double ePhi1  = sqrt(pow(ptPhi1, 2) + pow(pzPhi1, 2) + pow(massPhi1, 2));    

    for (size_t j = i+1; j < nPhiMesonOutputApproximate; j++) {
      const auto& phiMesonWord2 = l1PhiMesonWordHandle->at(j);
      
      double ptPhi2   = phiMesonWord2.pt();
      double etaPhi2  = phiMesonWord2.glbeta();
      double phiPhi2  = phiMesonWord2.glbphi();
      double z0Phi2   = phiMesonWord2.z0();
      double massPhi2 = phiMesonWord2.mass();

      // Ensure that the 2 Phi mesons are made of 4 distinct tracks
      unsigned int pIndexPhi2 = phiMesonWord2.firstIndex();
      unsigned int nIndexPhi2 = phiMesonWord2.secondIndex();
      if (pIndexPhi1 == pIndexPhi2 || nIndexPhi1 == nIndexPhi2) continue;

      double pxPhi2 = ptPhi2 * cos(phiPhi2);
      double pyPhi2 = ptPhi2 * sin(phiPhi2);
      double pzPhi2 = ptPhi2 * sinh(etaPhi2);
      double ePhi2  = sqrt(pow(ptPhi2, 2) + pow(pzPhi2, 2) + pow(massPhi2, 2));
      
      double pxBs = pxPhi1 + pxPhi2;
      double pyBs = pyPhi1 + pyPhi2;
      double pzBs = pzPhi1 + pzPhi2;
      double eBs  = ePhi1  + ePhi2;

      if (std::fabs(z0Phi1 - z0Phi2) > phiPairdzMax_) continue;
  
      double drPhiPair = reco::deltaR(etaPhi1, phiPhi1, etaPhi2, phiPhi2);
      if (drPhiPair < phiPairdRMin_ || drPhiPair > phiPairdRMax_) continue;
      
      double massPhiPair = sqrt(pow(eBs, 2) - pow(pxBs, 2) - pow(pyBs, 2) - pow(pzBs, 2));
      if (massPhiPair < phiPairMMin_ || massPhiPair > phiPairMMax_) continue;
      
      double pt = sqrt(pow(pxBs, 2) + pow(pyBs, 2));
      if (pt < bsPtMin_) continue;
      
      l1t::TkLightMesonWord::valid_t validBs = phiMesonWord1.valid() && phiMesonWord2.valid();
      l1t::TkLightMesonWord::pt_t ptBs = sqrt(pow(pxBs, 2) + pow(pyBs, 2));
      l1t::TkLightMesonWord::glbphi_t phiBs = atan2(pyBs, pxBs) / ETAPHI_LSB;
      l1t::TkLightMesonWord::glbeta_t etaBs = asinh(pzBs/sqrt(pow(pxBs, 2) + pow(pyBs, 2))) / ETAPHI_LSB;
      l1t::TkLightMesonWord::z0_t z0Bs = ((z0Phi1 + z0Phi2) / Z0_LSB) * 0.5;
      l1t::TkLightMesonWord::mass_t massBs  = sqrt(pow(eBs, 2) - pow(pxBs, 2) - pow(pyBs, 2) - pow(pzBs, 2));
      l1t::TkLightMesonWord::type_t typeBs = l1t::TkLightMesonWord::TkLightMesonTypes::kBsType;
      l1t::TkLightMesonWord::ntracks_t ntracksBs = 3;
      l1t::TkLightMesonWord::index_t firstPhiIndex     = i;
      l1t::TkLightMesonWord::index_t secondPhiIndex    = j;
      l1t::TkLightMesonWord::unassigned_t unassignedBs = 0;

      l1t::TkLightMesonWord bsWord(validBs, ptBs, phiBs, etaBs, z0Bs, massBs, typeBs, ntracksBs, firstPhiIndex, secondPhiIndex, unassignedBs); 
      L1BsMesonEmulationOutput->push_back(bsWord);
    }
  }
  // Put the outputs into the event
  iEvent.put(std::move(L1BsMesonEmulationOutput), outputCollectionName_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1BsMesonSelectionEmulationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1PhiMesonWordInputTag", edm::InputTag("l1PhiMesonSelectionEmulationProducer","Level1PhiMesonEmulationColl"));
  desc.add<edm::InputTag>("posTrackInputTag", edm::InputTag("l1KaonTrackSelectionProducer", "Level1TTKaonTracksSelectedEmulationPositivecharge"));
  desc.add<edm::InputTag>("negTrackInputTag", edm::InputTag("l1KaonTrackSelectionProducer", "Level1TTKaonTracksSelectedEmulationNegativecharge"));
  desc.add<std::string>("outputCollectionName", "Level1BsHadronicEmulationColl");
  desc.add<int>("debug", 0)->setComment("Verbosity levels: 0, 1, 2, 3");
  {
    edm::ParameterSetDescription descCutSet;
    descCutSet.add<double>("phiPairdzMax", 0.5)->setComment("dz between phi pair must be less than this value, [cm]");
    descCutSet.add<double>("phiPairdRMin", 0.2)->setComment("dR between phi pair must be greater than this value");
    descCutSet.add<double>("phiPairdRMax", 1.0)->setComment("dR between phi pair must be less than this value");
    descCutSet.add<double>("phiPairMMin", 5.0)->setComment("Bs mass must be greater than this value, [GeV]");
    descCutSet.add<double>("phiPairMMax", 5.8)->setComment("Bs mass must be less than this value, [GeV]");
    descCutSet.add<double>("bsPtMin", 13.0)->setComment("Bs pt must be greater than this value, [GeV]");
    desc.add<edm::ParameterSetDescription>("cutSet", descCutSet);
  }
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1BsMesonSelectionEmulationProducer);
