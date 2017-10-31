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

//
// class declaration
//

template <typename T>
class VIDNestedWPBitmapProducer : public edm::stream::EDProducer<> {
   public:
  explicit VIDNestedWPBitmapProducer(const edm::ParameterSet &iConfig):
    src_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))),
    isInit_(false)
  {
    auto vwp = iConfig.getParameter<std::vector<std::string>>("WorkingPoints");
    for (auto wp : vwp) {
      src_bitmaps_.push_back(consumes<edm::ValueMap<unsigned int> >(edm::InputTag(wp+std::string("Bitmap"))));
      src_cutflows_.push_back(consumes<edm::ValueMap<vid::CutFlowResult> >(edm::InputTag(wp)));
    }
    nWP = src_bitmaps_.size();
    produces<edm::ValueMap<int>>();
  }
  ~VIDNestedWPBitmapProducer() override {}
  
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
  void beginStream(edm::StreamID) override {};
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override {};

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<T>> src_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<unsigned int> > > src_bitmaps_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<vid::CutFlowResult> > > src_cutflows_;

  unsigned int nWP;
  unsigned int nBits;
  unsigned int nCuts;
  std::vector<unsigned int> res_;
  bool isInit_;

  void initNCuts(unsigned int);

};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

template <typename T>
void
VIDNestedWPBitmapProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<edm::View<T>> src;
  iEvent.getByToken(src_, src);
  std::vector<edm::Handle<edm::ValueMap<unsigned int>>> src_bitmaps(nWP);
  for (unsigned int i=0; i<nWP; i++) iEvent.getByToken(src_bitmaps_[i], src_bitmaps[i]);
  std::vector<edm::Handle<edm::ValueMap<vid::CutFlowResult>>> src_cutflows(nWP);
  for (unsigned int i=0; i<nWP; i++) iEvent.getByToken(src_cutflows_[i], src_cutflows[i]);

  std::vector<unsigned int> res;

  auto npho = src->size();
  for (unsigned int i=0; i<npho; i++){
    auto obj = src->ptrAt(i);
    for (unsigned int j=0; j<nWP; j++){
      auto cutflow = (*(src_cutflows[j]))[obj];
      if (!isInit_) initNCuts(cutflow.cutFlowSize());
      if (cutflow.cutFlowSize()!=nCuts) throw cms::Exception("Configuration","Trying to compress VID bitmaps for cutflows of different size");
      auto bitmap = (*(src_bitmaps[j]))[obj];
      for (unsigned int k=0; k<nCuts; k++){
	if (j==0) res_[k] = 0;
	if (bitmap>>k & 1) {
	  if (res_[k]!=j) throw cms::Exception("Configuration","Trying to compress VID bitmaps which are not nested in the correct order for all cuts");
	  res_[k]++;
	}
      }
    }

    int out = 0;
    for (unsigned int k=0; k<nCuts; k++) out |= (res_[k] << (nBits*k));
    res.push_back(out);
  }


  std::unique_ptr<edm::ValueMap<int>> resV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filler(*resV);
  filler.insert(src,res.begin(),res.end());
  filler.fill();

  iEvent.put(std::move(resV));

}

template <typename T>
void
VIDNestedWPBitmapProducer<T>::initNCuts(unsigned int n){
  nCuts = n;
  nBits = ceil(log2(nWP+1));
  if (nBits*nCuts>sizeof(int)*8) throw cms::Exception("Configuration","Integer cannot contain the compressed VID bitmap information");
  res_.resize(nCuts,0);
  isInit_ = true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void
VIDNestedWPBitmapProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  desc.add<std::vector<std::string>>("WorkingPoints")->setComment("working points to be saved in the bitmask");
  std::string modname;
  if (typeid(T) == typeid(pat::Electron)) modname+="Ele";
  else if (typeid(T) == typeid(pat::Photon)) modname+="Pho";
  modname+="VIDNestedWPBitmapProducer";
  descriptions.add(modname,desc);
}


typedef VIDNestedWPBitmapProducer<pat::Electron> EleVIDNestedWPBitmapProducer;
typedef VIDNestedWPBitmapProducer<pat::Photon> PhoVIDNestedWPBitmapProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(EleVIDNestedWPBitmapProducer);
DEFINE_FWK_MODULE(PhoVIDNestedWPBitmapProducer);

