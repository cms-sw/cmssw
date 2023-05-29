// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      VertexTableProducer
//
/**\class VertexTableProducer VertexTableProducer.cc PhysicsTools/VertexTableProducer/plugins/VertexTableProducer.cc

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

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "DataFormats/Common/interface/ValueMap.h"

//
// class declaration
//

class VertexTableProducer : public edm::stream::EDProducer<> {
public:
  explicit VertexTableProducer(const edm::ParameterSet&);
  ~VertexTableProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<std::vector<reco::Vertex>> pvs_;
  const edm::EDGetTokenT<edm::ValueMap<float>> pvsScore_;
  const edm::EDGetTokenT<edm::View<reco::VertexCompositePtrCandidate>> svs_;
  const StringCutObjectSelector<reco::Candidate> svCut_;
  const StringCutObjectSelector<reco::Vertex> goodPvCut_;
  const std::string goodPvCutString_;
  const std::string pvName_;
  const std::string svName_;
  const std::string svDoc_;
  const double dlenMin_, dlenSigMin_;
};

//
// constructors and destructor
//
VertexTableProducer::VertexTableProducer(const edm::ParameterSet& params)
    : pvs_(consumes<std::vector<reco::Vertex>>(params.getParameter<edm::InputTag>("pvSrc"))),
      pvsScore_(consumes<edm::ValueMap<float>>(params.getParameter<edm::InputTag>("pvSrc"))),
      svs_(consumes<edm::View<reco::VertexCompositePtrCandidate>>(params.getParameter<edm::InputTag>("svSrc"))),
      svCut_(params.getParameter<std::string>("svCut"), true),
      goodPvCut_(params.getParameter<std::string>("goodPvCut"), true),
      goodPvCutString_(params.getParameter<std::string>("goodPvCut")),
      pvName_(params.getParameter<std::string>("pvName")),
      svName_(params.getParameter<std::string>("svName")),
      svDoc_(params.getParameter<std::string>("svDoc")),
      dlenMin_(params.getParameter<double>("dlenMin")),
      dlenSigMin_(params.getParameter<double>("dlenSigMin"))

{
  produces<nanoaod::FlatTable>("pv");
  produces<nanoaod::FlatTable>("otherPVs");
  produces<nanoaod::FlatTable>("svs");
  produces<edm::PtrVector<reco::Candidate>>();
}

VertexTableProducer::~VertexTableProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------

void VertexTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  const auto& pvsScoreProd = iEvent.get(pvsScore_);
  auto pvsIn = iEvent.getHandle(pvs_);

  auto pvTable = std::make_unique<nanoaod::FlatTable>(1, pvName_, true);
  pvTable->addColumnValue<float>("ndof", (*pvsIn)[0].ndof(), "main primary vertex number of degree of freedom", 8);
  pvTable->addColumnValue<float>("x", (*pvsIn)[0].position().x(), "main primary vertex position x coordinate", 10);
  pvTable->addColumnValue<float>("y", (*pvsIn)[0].position().y(), "main primary vertex position y coordinate", 10);
  pvTable->addColumnValue<float>("z", (*pvsIn)[0].position().z(), "main primary vertex position z coordinate", 16);
  pvTable->addColumnValue<float>("chi2", (*pvsIn)[0].normalizedChi2(), "main primary vertex reduced chi2", 8);
  int goodPVs = 0;
  for (const auto& pv : *pvsIn)
    if (goodPvCut_(pv))
      goodPVs++;
  pvTable->addColumnValue<uint8_t>("npvs", pvsIn->size(), "total number of reconstructed primary vertices");
  pvTable->addColumnValue<uint8_t>(
      "npvsGood", goodPVs, "number of good reconstructed primary vertices. selection:" + goodPvCutString_);
  pvTable->addColumnValue<float>(
      "score", pvsScoreProd.get(pvsIn.id(), 0), "main primary vertex score, i.e. sum pt2 of clustered objects", 8);

  auto otherPVsTable =
      std::make_unique<nanoaod::FlatTable>((*pvsIn).size() > 4 ? 3 : (*pvsIn).size() - 1, "Other" + pvName_, false);
  std::vector<float> pvsz;
  std::vector<float> pvscores;
  for (size_t i = 1; i < (*pvsIn).size() && i < 4; i++) {
    pvsz.push_back((*pvsIn)[i].position().z());
    pvscores.push_back(pvsScoreProd.get(pvsIn.id(), i));
  }
  otherPVsTable->addColumn<float>("z", pvsz, "Z position of other primary vertices, excluding the main PV", 8);
  otherPVsTable->addColumn<float>("score", pvscores, "scores of other primary vertices, excluding the main PV", 8);

  const auto& svsProd = iEvent.get(svs_);
  auto selCandSv = std::make_unique<PtrVector<reco::Candidate>>();
  std::vector<float> dlen, dlenSig, pAngle, dxy, dxySig;
  std::vector<int16_t> charge;
  VertexDistance3D vdist;
  VertexDistanceXY vdistXY;

  size_t i = 0;
  const auto& PV0 = pvsIn->front();
  for (const auto& sv : svsProd) {
    if (svCut_(sv)) {
      Measurement1D dl =
          vdist.distance(PV0, VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));
      if (dl.value() > dlenMin_ and dl.significance() > dlenSigMin_) {
        dlen.push_back(dl.value());
        dlenSig.push_back(dl.significance());
        edm::Ptr<reco::Candidate> c = svsProd.ptrAt(i);
        selCandSv->push_back(c);
        double dx = (PV0.x() - sv.vx()), dy = (PV0.y() - sv.vy()), dz = (PV0.z() - sv.vz());
        double pdotv = (dx * sv.px() + dy * sv.py() + dz * sv.pz()) / sv.p() / sqrt(dx * dx + dy * dy + dz * dz);
        pAngle.push_back(std::acos(pdotv));
        Measurement1D d2d = vdistXY.distance(
            PV0, VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));
        dxy.push_back(d2d.value());
        dxySig.push_back(d2d.significance());

        int sum_charge = 0;
        for (unsigned int id = 0; id < sv.numberOfDaughters(); ++id) {
          const reco::Candidate* daughter = sv.daughter(id);
          sum_charge += daughter->charge();
        }
        charge.push_back(sum_charge);
      }
    }
    i++;
  }

  auto svsTable = std::make_unique<nanoaod::FlatTable>(selCandSv->size(), svName_, false);
  svsTable->setDoc(svDoc_);
  // For SV we fill from here only stuff that cannot be created with the SimpleFlatTableProducer
  svsTable->addColumn<float>("dlen", dlen, "decay length in cm", 10);
  svsTable->addColumn<float>("dlenSig", dlenSig, "decay length significance", 10);
  svsTable->addColumn<float>("dxy", dxy, "2D decay length in cm", 10);
  svsTable->addColumn<float>("dxySig", dxySig, "2D decay length significance", 10);
  svsTable->addColumn<float>("pAngle", pAngle, "pointing angle, i.e. acos(p_SV * (SV - PV)) ", 10);
  svsTable->addColumn<int16_t>("charge", charge, "sum of the charge of the SV tracks", 10);

  iEvent.put(std::move(pvTable), "pv");
  iEvent.put(std::move(otherPVsTable), "otherPVs");
  iEvent.put(std::move(svsTable), "svs");
  iEvent.put(std::move(selCandSv));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void VertexTableProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void VertexTableProducer::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void VertexTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("pvSrc")->setComment(
      "std::vector<reco::Vertex> and ValueMap<float> primary vertex input collections");
  desc.add<std::string>("goodPvCut")->setComment("selection on the primary vertex");
  desc.add<edm::InputTag>("svSrc")->setComment(
      "reco::VertexCompositePtrCandidate compatible secondary vertex input collection");
  desc.add<std::string>("svCut")->setComment("selection on the secondary vertex");

  desc.add<double>("dlenMin")->setComment("minimum value of dl to select secondary vertex");
  desc.add<double>("dlenSigMin")->setComment("minimum value of dl significance to select secondary vertex");

  desc.add<std::string>("pvName")->setComment("name of the flat table ouput");
  desc.add<std::string>("svName")->setComment("name of the flat table ouput");
  desc.add<std::string>("svDoc")->setComment("a few words of documentation");

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(VertexTableProducer);
