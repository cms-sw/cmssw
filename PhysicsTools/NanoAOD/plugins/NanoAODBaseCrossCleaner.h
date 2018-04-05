#ifndef PhysicsTools_NanoAOD_NanoAODBaseCrossCleaner_h
#define PhysicsTools_NanoAOD_NanoAODBaseCrossCleaner_h

// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      NanoAODBaseCrossCleaner
// 
/**\class NanoAODBaseCrossCleaner NanoAODBaseCrossCleaner.cc PhysicsTools/NanoAODBaseCrossCleaner/plugins/NanoAODBaseCrossCleaner.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Mon, 28 Aug 2017 09:26:39 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

#include "DataFormats/Common/interface/View.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

//
// class declaration
//

class NanoAODBaseCrossCleaner : public edm::stream::EDProducer<> {
   public:
      explicit NanoAODBaseCrossCleaner(const edm::ParameterSet&);
      ~NanoAODBaseCrossCleaner() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      void beginStream(edm::StreamID) override;
      void produce(edm::Event&, const edm::EventSetup&) override;
      void endStream() override;
      virtual void objectSelection( const edm::View<pat::Jet> & jets, const edm::View<pat::Muon>  & muons, const edm::View<pat::Electron> & eles, 
				    const edm::View<pat::Tau> & taus, const edm::View<pat::Photon>  & photons,
                                    std::vector<uint8_t> & jetBits, std::vector<uint8_t> & muonBits, std::vector<uint8_t> & eleBits,
  				    std::vector<uint8_t> & tauBits, std::vector<uint8_t> & photonBits) {};

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
      const std::string name_;
      const std::string doc_;

      const edm::EDGetTokenT<edm::View<pat::Jet>> jets_;
      const edm::EDGetTokenT<edm::View<pat::Muon>> muons_;
      const edm::EDGetTokenT<edm::View<pat::Electron>> electrons_;
      const edm::EDGetTokenT<edm::View<pat::Tau>> taus_;
      const edm::EDGetTokenT<edm::View<pat::Photon>> photons_;
      const StringCutObjectSelector<pat::Jet>  jetSel_;
      const StringCutObjectSelector<pat::Muon>  muonSel_;
      const StringCutObjectSelector<pat::Electron>  electronSel_;
      const StringCutObjectSelector<pat::Tau>  tauSel_;
      const StringCutObjectSelector<pat::Photon>  photonSel_;
      const std::string  jetName_;
      const std::string  muonName_;
      const std::string  electronName_;
      const std::string  tauName_;
      const std::string  photonName_;


};

#endif
