// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      VIDNestedWPBitmapProducer
//
/**\class VIDNestedWPBitmapProducer VIDNestedWPBitmapProducer.cc PhysicsTools/NanoAOD/plugins/VIDNestedWPBitmapProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Peruzzi
//         Created:  Mon, 04 Sep 2017 22:43:53 GMT
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

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"
#include "DataFormats/Common/interface/Ptr.h"

//
// class declaration
//

namespace {

  //will fail for non-electrons/photons (which so far are not used for this class)
  //when we want to use non-electrons/photons, template specalisation will be necessary
  template <typename T>
  bool equal(const T& lhs, const T& rhs) {
    return lhs.superCluster()->seed()->seed().rawId() == rhs.superCluster()->seed()->seed().rawId();
  }

  //returns a edm::Ptr to the object matching the passed in object in object coll
  //if objColl is invalid, passes the original pointer back
  //if valid but not found, returns null
  template <typename T>
  edm::Ptr<T> getObjInColl(edm::Ptr<T> obj, edm::Handle<edm::View<T>> objColl) {
    if (objColl.isValid()) {
      for (auto& objToMatch : objColl->ptrs()) {
        if (equal(*obj, *objToMatch)) {
          return objToMatch;
        }
      }
      return edm::Ptr<T>(objColl.id());
    }
    return obj;
  }

}  // namespace

template <typename T>
class VIDNestedWPBitmapProducer : public edm::stream::EDProducer<> {
public:
  explicit VIDNestedWPBitmapProducer(const edm::ParameterSet& iConfig)
      : src_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))),
        srcForIDToken_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("srcForID"))),
        isInit_(false) {
    auto const& vwp = iConfig.getParameter<std::vector<std::string>>("WorkingPoints");
    for (auto const& wp : vwp) {
      src_bitmaps_.push_back(consumes<edm::ValueMap<unsigned int>>(edm::InputTag(wp + std::string("Bitmap"))));
      src_cutflows_.push_back(consumes<edm::ValueMap<vid::CutFlowResult>>(edm::InputTag(wp)));
    }
    nWP = src_bitmaps_.size();
    produces<edm::ValueMap<int>>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<T>> src_;
  edm::EDGetTokenT<edm::View<T>> srcForIDToken_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<unsigned int>>> src_bitmaps_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<vid::CutFlowResult>>> src_cutflows_;

  unsigned int nWP;
  unsigned int nBits;
  unsigned int nCuts = 0;
  std::vector<unsigned int> res_;
  bool isInit_;

  void initNCuts(unsigned int);
};

template <typename T>
void VIDNestedWPBitmapProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<T>> src;
  iEvent.getByToken(src_, src);

  auto srcForIDHandle = iEvent.getHandle(srcForIDToken_);

  std::vector<edm::Handle<edm::ValueMap<unsigned int>>> src_bitmaps(nWP);
  for (unsigned int i = 0; i < nWP; i++)
    iEvent.getByToken(src_bitmaps_[i], src_bitmaps[i]);
  std::vector<edm::Handle<edm::ValueMap<vid::CutFlowResult>>> src_cutflows(nWP);
  for (unsigned int i = 0; i < nWP; i++)
    iEvent.getByToken(src_cutflows_[i], src_cutflows[i]);

  std::vector<unsigned int> res;

  for (auto const& obj : src->ptrs()) {
    auto objForID = getObjInColl(obj, srcForIDHandle);
    for (unsigned int j = 0; j < nWP; j++) {
      auto cutflow = (*(src_cutflows[j]))[objForID];
      if (!isInit_)
        initNCuts(cutflow.cutFlowSize());
      if (cutflow.cutFlowSize() != nCuts)
        throw cms::Exception("Configuration", "Trying to compress VID bitmaps for cutflows of different size");
      auto bitmap = (*(src_bitmaps[j]))[objForID];
      for (unsigned int k = 0; k < nCuts; k++) {
        if (j == 0)
          res_[k] = 0;
        if (bitmap >> k & 1) {
          if (res_[k] != j)
            throw cms::Exception(
                "Configuration",
                "Trying to compress VID bitmaps which are not nested in the correct order for all cuts");
          res_[k]++;
        }
      }
    }

    int out = 0;
    for (unsigned int k = 0; k < nCuts; k++)
      out |= (res_[k] << (nBits * k));
    res.push_back(out);
  }

  auto resV = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler filler(*resV);
  filler.insert(src, res.begin(), res.end());
  filler.fill();

  iEvent.put(std::move(resV));
}

template <typename T>
void VIDNestedWPBitmapProducer<T>::initNCuts(unsigned int n) {
  nCuts = n;
  nBits = ceil(log2(nWP + 1));
  if (nBits * nCuts > sizeof(int) * 8)
    throw cms::Exception("Configuration", "Integer cannot contain the compressed VID bitmap information");
  res_.resize(nCuts, 0);
  isInit_ = true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void VIDNestedWPBitmapProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  desc.add<edm::InputTag>("srcForID", edm::InputTag())->setComment("physics object collection the ID value maps are ");
  desc.add<std::vector<std::string>>("WorkingPoints")->setComment("working points to be saved in the bitmask");
  std::string modname;
  if (typeid(T) == typeid(reco::GsfElectron))
    modname += "Ele";
  else if (typeid(T) == typeid(reco::Photon))
    modname += "Pho";
  modname += "VIDNestedWPBitmapProducer";
  descriptions.add(modname, desc);
}

typedef VIDNestedWPBitmapProducer<reco::GsfElectron> EleVIDNestedWPBitmapProducer;
typedef VIDNestedWPBitmapProducer<reco::Photon> PhoVIDNestedWPBitmapProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(EleVIDNestedWPBitmapProducer);
DEFINE_FWK_MODULE(PhoVIDNestedWPBitmapProducer);
