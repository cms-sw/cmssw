// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      EGMEnergyVarProducer
// 
/**\class EGMEnergyVarProducer EGMEnergyVarProducer.cc PhysicsTools/NanoAOD/plugins/EGMEnergyVarProducer.cc
 Description: [one line class summary]
 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Emanuele Di Marco
//         Created:  Wed, 06 Sep 2017 12:34:38 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

#include "TLorentzVector.h"
#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/NanoAOD/interface/MatchingUtils.h"

//
// class declaration
//

template <typename T>
class EGMEnergyVarProducer : public edm::global::EDProducer<> {
public:
  explicit EGMEnergyVarProducer(const edm::ParameterSet &iConfig):
    srcRaw_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("srcRaw"))),
    srcCorr_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("srcCorr")))
  {
    produces<edm::ValueMap<float>>("eCorr");
  }
    ~EGMEnergyVarProducer() override {};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<T>> srcRaw_;
  edm::EDGetTokenT<edm::View<T>> srcCorr_;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename T>
void
EGMEnergyVarProducer<T>::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  edm::Handle<edm::View<T>> srcRaw;
  iEvent.getByToken(srcRaw_, srcRaw);
  edm::Handle<edm::View<T>> srcCorr;
  iEvent.getByToken(srcCorr_, srcCorr);

  unsigned nSrcRaw = srcRaw->size();
  unsigned nSrcCorr = srcCorr->size();

  std::vector<float> eCorr(nSrcCorr,-1);

  for (unsigned int ir = 0; ir<nSrcRaw; ir++){
    auto egm_raw = srcRaw->ptrAt(ir);
    for (unsigned int ic = 0; ic<nSrcCorr; ic++){
      auto egm_corr = srcCorr->ptrAt(ic);
      if(matchByCommonParentSuperClusterRef(*egm_raw,*egm_corr)){
          eCorr[ir] = egm_corr->energy()/egm_raw->energy();
          break;
      }
    }
  }

  std::unique_ptr<edm::ValueMap<float>> eCorrV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerCorr(*eCorrV);
  fillerCorr.insert(srcRaw,eCorr.begin(),eCorr.end());
  fillerCorr.fill();
  iEvent.put(std::move(eCorrV),"eCorr");

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void
EGMEnergyVarProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcRaw")->setComment("input raw physics object collection");
  desc.add<edm::InputTag>("srcCorr")->setComment("input corrected physics object collection");
  std::string modname;
  if (typeid(T) == typeid(pat::Electron)) modname+="Electron";
  else if (typeid(T) == typeid(pat::Photon)) modname+="Photon";
  modname+="EnergyVarProducer";
  descriptions.add(modname,desc);
}

typedef EGMEnergyVarProducer<pat::Electron> ElectronEnergyVarProducer;
typedef EGMEnergyVarProducer<pat::Photon> PhotonEnergyVarProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronEnergyVarProducer);
DEFINE_FWK_MODULE(PhotonEnergyVarProducer);
