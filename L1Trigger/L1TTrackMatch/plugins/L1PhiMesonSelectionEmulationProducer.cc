// -*- C++ -*-
//
// Package:    L1Trigger/L1TTrackMatch
// Class:      L1PhiMesonSelectionEmulationProducer
//
/**\class L1PhiMesonSelectionEmulationProducer L1PhiMesonSelectionEmulationProducer.cc L1Trigger/L1TTrackMatch/plugins/L1PhiMesonSelectionEmulationProducer.cc

 Description: Build Phi meson candidates from positively and negatively charged selected L1Tracks (assuming kaons)

 Implementation:
     Inputs:
         std::vector<TTTrack> - Positively and negatively charged collection of selected L1Tracks which inherits from a bit-accurate TTTrack_TrackWord, used for emulation purposes
     Outputs:
         l1t::TkLightMesonWordCollection - A collection of reconstructed Phi meson candidates
*/
// ----------------------------------------------------------------------------                                                    
// Authors: Alexx Perloff, Pritam Palit (original version, 2021),
//          Sweta Baradia, Suchandra Dutta, Subir Sarkar (January 2025)
//----------------------------------------------------------------------------- 

// system include files
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <TMath.h>
#include <cmath>
#include <bitset>
#include <TSystem.h>
#include <array>
// Xilinx HLS includes
#include <ap_fixed.h>
#include <ap_int.h>
#include <stdio.h>
#include <cassert>
#include <cstdlib>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuAlgo.h"
#include "DataFormats/L1Trigger/interface/TkLightMesonWord.h"
#include "L1TrackWordUnpacker.h"


using namespace std;
using namespace edm;
using namespace l1t;

using L1TTTrackType            = TTTrack<Ref_Phase2TrackerDigi_>;                                 
using TTTrackCollection        = std::vector<L1TTTrackType>;                                    
using TTTrackRef               = edm::Ref<TTTrackCollection>;                                   
using TTTrackRefCollection     = edm::RefVector<TTTrackCollection>;                             
using TTTrackCollectionHandle  = edm::Handle<TTTrackRefCollection>;
using TTTrackRefCollectionUPtr = std::unique_ptr<TTTrackRefCollection>; 

// Class declaration 
class L1PhiMesonSelectionEmulationProducer : public edm::global::EDProducer<> {
public:
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  explicit L1PhiMesonSelectionEmulationProducer(const edm::ParameterSet&);
  ~L1PhiMesonSelectionEmulationProducer() override;

  static constexpr double KaonMass = 0.493677 ;
  static constexpr double ETAPHI_LSB = M_PI / (1 << 12);
  static constexpr double Z0_LSB = 0.05;
  size_t phiSize = 20;
  
private:
  // ----------member functions ----------------------
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
 
  // ----------selectors -----------------------------
  // Based on recommendations from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenericSelectors
  // ----------member data ---------------------------
  const edm::EDGetTokenT<TTTrackRefCollection> posTrackToken_;
  const edm::EDGetTokenT<TTTrackRefCollection> negTrackToken_;
  const std::string outputCollectionName_;
  const edm::ParameterSet cutSet_;
  const double tkPairdzMax_, tkPairdRMax_, tkPairMMin_, tkPairMMax_;
  
  int debug_;
};

//
// constructors and destructor
//
L1PhiMesonSelectionEmulationProducer::L1PhiMesonSelectionEmulationProducer(const edm::ParameterSet& iConfig)
  : posTrackToken_(consumes<TTTrackRefCollection>(iConfig.getParameter<edm::InputTag>("l1PosKaonTracksInputTag"))),
    negTrackToken_(consumes<TTTrackRefCollection>(iConfig.getParameter<edm::InputTag>("l1NegKaonTracksInputTag"))),
    outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName")),
    cutSet_(iConfig.getParameter<edm::ParameterSet>("cutSet")),
    tkPairdzMax_(cutSet_.getParameter<double>("tkPairdzMax")),
    tkPairdRMax_(cutSet_.getParameter<double>("tkPairdRMax")),
    tkPairMMin_(cutSet_.getParameter<double>("tkPairMMin")),
    tkPairMMax_(cutSet_.getParameter<double>("tkPairMMax")),
    debug_(iConfig.getParameter<int>("debug")) {
  produces<l1t::TkLightMesonWordCollection>(outputCollectionName_);
}

L1PhiMesonSelectionEmulationProducer::~L1PhiMesonSelectionEmulationProducer() {}

// ------------ method called to produce the data  ------------                                                                                                    
void L1PhiMesonSelectionEmulationProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto L1PhiMesonEmulationOutput = std::make_unique<l1t::TkLightMesonWordCollection>();

  TTTrackCollectionHandle posTrackHandle;
  iEvent.getByToken(posTrackToken_, posTrackHandle);

  TTTrackCollectionHandle negTrackHandle;
  iEvent.getByToken(negTrackToken_, negTrackHandle);

  auto nPosKaon = posTrackHandle->size();
  auto nNegKaon = negTrackHandle->size();

  L1PhiMesonEmulationOutput->reserve(phiSize);
  
  ap_ufixed<64, 32> etaphi_conv = 1.0 / ETAPHI_LSB;
  ap_ufixed<64, 32> z0_conv = 1.0 * Z0_LSB;
  
  for (size_t i = 0; i < nPosKaon; i++) {
    const auto& trackPosKaonRef = posTrackHandle->at(i);
    const auto& trackPosKaon = *trackPosKaonRef;

    // bitwise emulated values for positive tracks
    double trkptPos  = l1trackunpacker::FloatPtFromBits(trackPosKaon);
    double trketaPos = l1trackunpacker::FloatEtaFromBits(trackPosKaon);
    double trkphiPos = l1trackunpacker::FloatPhiFromBits(trackPosKaon);
    double trkz0Pos  = l1trackunpacker::FloatZ0FromBits(trackPosKaon);

    double trkpxPos = trkptPos * cos(trkphiPos);
    double trkpyPos = trkptPos * sin(trkphiPos);
    double trkpzPos = trkptPos * sinh(trketaPos);
    double trkePos  = sqrt(pow(trkptPos, 2) + pow(trkpzPos, 2) + pow(KaonMass, 2));

    for (size_t j = 0; j < nNegKaon; j++) {
      const auto& trackNegKaonRef = negTrackHandle->at(j);
      const auto& trackNegKaon = *trackNegKaonRef;

      // bitwise emulated values for negative tracks
      double trkptNeg  = l1trackunpacker::FloatPtFromBits(trackNegKaon);
      double trketaNeg = l1trackunpacker::FloatEtaFromBits(trackNegKaon);
      double trkphiNeg = l1trackunpacker::FloatPhiFromBits(trackNegKaon);
      double trkz0Neg  = l1trackunpacker::FloatZ0FromBits(trackNegKaon);
      
      double trkpxNeg = trkptNeg * cos(trkphiNeg);
      double trkpyNeg = trkptNeg * sin(trkphiNeg);
      double trkpzNeg = trkptNeg * sinh(trketaNeg);
      double trkeNeg  = sqrt(pow(trkptNeg, 2) + pow(trkpzNeg, 2) + pow(KaonMass, 2));
    
      double pxPhi = trkpxNeg + trkpxPos;
      double pyPhi = trkpyNeg + trkpyPos;
      double pzPhi = trkpzNeg + trkpzPos;
      double ePhi  = trkeNeg  + trkePos;

      if (std::fabs(trkz0Pos - trkz0Neg) > tkPairdzMax_) continue;
      
      double convdPhi = std::abs(trkphiPos - trkphiNeg);
      if (convdPhi > M_PI) convdPhi-= 2 * M_PI;
      double trkdrpairPhi = sqrt(pow(convdPhi,2) + pow(trketaPos - trketaNeg, 2));
      if (trkdrpairPhi > tkPairdRMax_) continue;

      double trkmasspairPhi = sqrt(pow(ePhi,2) - pow(pxPhi, 2) - pow(pyPhi, 2) - pow(pzPhi, 2)); 
      if (trkmasspairPhi < tkPairMMin_ || trkmasspairPhi > tkPairMMax_) continue;
      
      l1t::TkLightMesonWord::valid_t validPhi           = trackPosKaon.getValid() && trackNegKaon.getValid();
      l1t::TkLightMesonWord::pt_t ptPhi                 = sqrt(pow(pxPhi, 2) + pow(pyPhi, 2)); 
      l1t::TkLightMesonWord::glbphi_t phiPhi            = atan2(pyPhi, pxPhi) / ETAPHI_LSB;
      l1t::TkLightMesonWord::glbeta_t etaPhi            = asinh(pzPhi/sqrt(pow(pxPhi, 2) + pow(pyPhi, 2))) / ETAPHI_LSB;
      l1t::TkLightMesonWord::z0_t z0Phi                 = ((trkz0Pos + trkz0Neg) / Z0_LSB)* 0.5;
      l1t::TkLightMesonWord::mass_t mPhi                = sqrt(pow(ePhi, 2) - pow(pxPhi, 2) - pow(pyPhi, 2) - pow(pzPhi, 2));
      l1t::TkLightMesonWord::type_t typePhi             = l1t::TkLightMesonWord::TkLightMesonTypes::kPhiType;
      l1t::TkLightMesonWord::ntracks_t ntracksPhi       = 2;
      l1t::TkLightMesonWord::index_t firstTrkIndex      = i;
      l1t::TkLightMesonWord::index_t secondTrkIndex     = j;
      l1t::TkLightMesonWord::unassigned_t unassignedPhi = 0;

      l1t::TkLightMesonWord PhiWord(validPhi, ptPhi, phiPhi, etaPhi, z0Phi, mPhi, typePhi, ntracksPhi, firstTrkIndex, secondTrkIndex, unassignedPhi);

      // Store unique phi candidates
      bool dupl = false;
      for (const auto& el: *L1PhiMesonEmulationOutput) {
        double ptDiff  = el.pt()  - PhiWord.pt();
        double etaDiff = el.glbeta() - PhiWord.glbeta();
        double phiDiff = el.glbphi() - PhiWord.glbphi();
        if ( fabs(etaDiff) < 1.0e-03 &&
             fabs(phiDiff) < 1.0e-03 &&
             fabs(ptDiff)  < 1.0e-02 )
          {
            dupl = true;
            break;
          }
      }
      if (!dupl) {
	L1PhiMesonEmulationOutput->push_back(PhiWord);
      }
    }
  }
  iEvent.put(std::move(L1PhiMesonEmulationOutput), outputCollectionName_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1PhiMesonSelectionEmulationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1PosKaonTracksInputTag",
                          edm::InputTag("l1KaonTrackSelectionProducer","Level1TTKaonTracksSelectedEmulationPositivecharge"));
  desc.add<edm::InputTag>("l1NegKaonTracksInputTag",
                          edm::InputTag("l1KaonTrackSelectionProducer","Level1TTKaonTracksSelectedEmulationNegativecharge"));
  desc.add<std::string>("outputCollectionName", "Level1PhiMesonEmulationColl");

  {
    edm::ParameterSetDescription descCutSet;
    descCutSet.add<double>("tkPairdzMax", 0.5)->setComment("dz between phi track pair must be less than this value, [cm]");
    descCutSet.add<double>("tkPairdRMax", 0.2)->setComment("dR between phi track pair must be less than this value, []");
    descCutSet.add<double>("tkPairMMin", 1.0)->setComment("phi mass must be greater than this value, [GeV]");
    descCutSet.add<double>("tkPairMMax", 1.03)->setComment("phi mass must be less than this value, [GeV]");
    desc.add<edm::ParameterSetDescription>("cutSet", descCutSet);
  }
  desc.add<int>("debug", 0)->setComment("Verbosity levels: 0, 1, 2, 3");
  descriptions.addWithDefaultLabel(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(L1PhiMesonSelectionEmulationProducer);
