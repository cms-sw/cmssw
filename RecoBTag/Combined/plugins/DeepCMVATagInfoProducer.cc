// -*- C++ -*-
//
// Package:    ​RecoBTag/​SecondaryVertex
// Class:      DeepNNTagInfoProducer
//
/**\class DeepNNTagInfoProducer DeepNNTagInfoProducer.cc ​RecoBTag/DeepFlavour/plugins/DeepNNTagInfoProducer.cc
 *
 * Description: EDProducer that produces collection of DeepNNTagInfos
 *
 * Implementation:
 *    A collection of CandIPTagInfo and CandSecondaryVertexTagInfo and a CombinedSVComputer ESHandle is taken as input and a collection of DeepNNTagInfos
 *    is produced as output.
 */
//
// Original Author:  Mauro Verzetti (U. Rochester)
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include <cmath>
#include <map>

using namespace reco;

//
// class declaration
//

inline bool equals(const edm::RefToBase<Jet>& j1, const edm::RefToBase<Jet>& j2) {
  return j1.id() == j2.id() && j1.key() == j2.key();
}

class DeepCMVATagInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit DeepCMVATagInfoProducer(const edm::ParameterSet&);
  ~DeepCMVATagInfoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override {}
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override {}

  // ----------member data ---------------------------
  const edm::EDGetTokenT<std::vector<reco::ShallowTagInfo> > deepNNSrc_;
  const edm::EDGetTokenT<edm::View<reco::BaseTagInfo> > ipInfoSrc_;
  const edm::EDGetTokenT<edm::View<reco::BaseTagInfo> > muInfoSrc_;
  const edm::EDGetTokenT<edm::View<reco::BaseTagInfo> > elInfoSrc_;
  std::string jpComputer_, jpbComputer_, softmuComputer_, softelComputer_;
  double cMVAPtThreshold_;
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
DeepCMVATagInfoProducer::DeepCMVATagInfoProducer(const edm::ParameterSet& iConfig)
    : deepNNSrc_(consumes<std::vector<reco::ShallowTagInfo> >(iConfig.getParameter<edm::InputTag>("deepNNTagInfos"))),
      ipInfoSrc_(consumes<edm::View<reco::BaseTagInfo> >(iConfig.getParameter<edm::InputTag>("ipInfoSrc"))),
      muInfoSrc_(consumes<edm::View<reco::BaseTagInfo> >(iConfig.getParameter<edm::InputTag>("muInfoSrc"))),
      elInfoSrc_(consumes<edm::View<reco::BaseTagInfo> >(iConfig.getParameter<edm::InputTag>("elInfoSrc"))),
      jpComputer_(iConfig.getParameter<std::string>("jpComputerSrc")),
      jpbComputer_(iConfig.getParameter<std::string>("jpbComputerSrc")),
      softmuComputer_(iConfig.getParameter<std::string>("softmuComputerSrc")),
      softelComputer_(iConfig.getParameter<std::string>("softelComputerSrc")),
      cMVAPtThreshold_(iConfig.getParameter<double>("cMVAPtThreshold")) {
  produces<std::vector<reco::ShallowTagInfo> >();
}

DeepCMVATagInfoProducer::~DeepCMVATagInfoProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void DeepCMVATagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get input TagInfos from DeepCSV
  edm::Handle<std::vector<reco::ShallowTagInfo> > nnInfos;
  iEvent.getByToken(deepNNSrc_, nnInfos);

  // get other Taginfos
  edm::Handle<edm::View<BaseTagInfo> > ipInfos;
  iEvent.getByToken(ipInfoSrc_, ipInfos);
  edm::Handle<edm::View<BaseTagInfo> > muInfos;
  iEvent.getByToken(muInfoSrc_, muInfos);
  edm::Handle<edm::View<BaseTagInfo> > elInfos;
  iEvent.getByToken(elInfoSrc_, elInfos);

  //get computers
  edm::ESHandle<JetTagComputer> jp;
  iSetup.get<JetTagComputerRecord>().get(jpComputer_, jp);
  const JetTagComputer* compjp = jp.product();
  edm::ESHandle<JetTagComputer> jpb;
  iSetup.get<JetTagComputerRecord>().get(jpbComputer_, jpb);
  const JetTagComputer* compjpb = jpb.product();
  edm::ESHandle<JetTagComputer> softmu;
  iSetup.get<JetTagComputerRecord>().get(softmuComputer_, softmu);
  const JetTagComputer* compsoftmu = softmu.product();
  edm::ESHandle<JetTagComputer> softel;
  iSetup.get<JetTagComputerRecord>().get(softelComputer_, softel);
  const JetTagComputer* compsoftel = softel.product();

  // create the output collection
  auto tagInfos = std::make_unique<std::vector<reco::ShallowTagInfo> >();

  // loop over TagInfos, assume they are ordered in the same way, check later and throw exception if not
  for (size_t idx = 0; idx < nnInfos->size(); ++idx) {
    auto& nnInfo = nnInfos->at(idx);
    auto& ipInfo = ipInfos->at(idx);
    auto& muInfo = muInfos->at(idx);
    auto& elInfo = elInfos->at(idx);

    if (!equals(nnInfo.jet(), ipInfo.jet()) || !equals(nnInfo.jet(), muInfo.jet()) ||
        !equals(nnInfo.jet(), elInfo.jet())) {
      throw cms::Exception("ValueError")
          << "DeepNNTagInfoProducer::produce: The tagInfos taken belong to different jets!" << std::endl
          << "This could be due to: " << std::endl
          << "  - You passed tagInfos computed on different jet collection" << std::endl
          << "  - The assumption that the tagInfos are filled in the same order is actually wrong" << std::endl;
    }

    // Make vector of BaseTagInfo, needed for TagInfoHelper
    std::vector<const BaseTagInfo*> ipBaseInfo;
    ipBaseInfo.push_back(&ipInfo);
    std::vector<const BaseTagInfo*> muBaseInfo;
    muBaseInfo.push_back(&muInfo);
    std::vector<const BaseTagInfo*> elBaseInfo;
    elBaseInfo.push_back(&elInfo);

    // Copy the DeepNN TaggingVariables + add the other discriminators
    TaggingVariableList vars = nnInfo.taggingVariables();
    float softmu_discr = (*compsoftmu)(JetTagComputer::TagInfoHelper(muBaseInfo));
    float softel_discr = (*compsoftel)(JetTagComputer::TagInfoHelper(elBaseInfo));
    float jp_discr = (*compjp)(JetTagComputer::TagInfoHelper(ipBaseInfo));
    float jpb_discr = (*compjpb)(JetTagComputer::TagInfoHelper(ipBaseInfo));

    //if jetPt larger than cMVAPtThreshold_ --> default these taggers for easier SF measurements
    if ((nnInfo.jet().get())->pt() < cMVAPtThreshold_) {
      vars.insert(reco::btau::Jet_SoftMu, !(std::isinf(softmu_discr)) ? softmu_discr : -0.2, true);
      vars.insert(reco::btau::Jet_SoftEl, !(std::isinf(softel_discr)) ? softel_discr : -0.2, true);
      vars.insert(reco::btau::Jet_JBP, !(std::isinf(jpb_discr)) ? jpb_discr : -0.2, true);
      vars.insert(reco::btau::Jet_JP, !(std::isinf(jp_discr)) ? jp_discr : -0.2, true);
    }

    vars.finalize();
    tagInfos->emplace_back(vars, nnInfo.jet());

    // just to be sure, clear all containers
    ipBaseInfo.clear();
    muBaseInfo.clear();
    elBaseInfo.clear();
  }

  // put the output in the event
  iEvent.put(std::move(tagInfos));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DeepCMVATagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepCMVATagInfoProducer);
