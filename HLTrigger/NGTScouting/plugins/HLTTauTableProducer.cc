#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HLTTauTableProducer : public edm::global::EDProducer<> {
public:
  using TauCollection = edm::View<reco::BaseTau>;
  using TauIPVector = edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>;
  using TauDiscrMap = reco::TauDiscriminatorContainer;
  // TauCollection = deeptau.TauCollection;
  // using TauDeepTauVector = edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::TauDiscriminatorContainer>>;
  HLTTauTableProducer(const edm::ParameterSet& cfg)
      : tableName_(cfg.getParameter<std::string>("tableName")),
        skipNonExistingSrc_(cfg.getParameter<bool>("skipNonExistingSrc")),
        tauToken_(mayConsume<TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        tauIPToken_(mayConsume<TauIPVector>(cfg.getParameter<edm::InputTag>("tauTransverseImpactParameters"))),
        deepTauVSeToken_(mayConsume<TauDiscrMap>(cfg.getParameter<edm::InputTag>("deepTauVSe"))),
        deepTauVSmuToken_(mayConsume<TauDiscrMap>(cfg.getParameter<edm::InputTag>("deepTauVSmu"))),
        deepTauVSjetToken_(mayConsume<TauDiscrMap>(cfg.getParameter<edm::InputTag>("deepTauVSjet"))),
        precision_(cfg.getParameter<int>("precision")) {
    produces<nanoaod::FlatTable>(tableName_);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("tableName", "hltHpsPFTau")
        ->setComment("Table name, needs to be the same as the main Tau table");
    desc.add<bool>("skipNonExistingSrc", false)
        ->setComment("whether or not to skip producing the table on absent input product");
    desc.add<edm::InputTag>("taus", edm::InputTag(""));
    desc.add<edm::InputTag>("tauTransverseImpactParameters", edm::InputTag(""));
    desc.add<edm::InputTag>("deepTauVSe", edm::InputTag(""));
    desc.add<edm::InputTag>("deepTauVSmu", edm::InputTag(""));
    desc.add<edm::InputTag>("deepTauVSjet", edm::InputTag(""));
    desc.add<int>("precision", 7);
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& setup) const override {
    const auto tausHandle = event.getHandle(tauToken_);
    const size_t nTaus = tausHandle.isValid() ? (*tausHandle).size() : 0;

    // resize all output vectors
    static constexpr float default_value = std::numeric_limits<float>::quiet_NaN();

    std::vector<float> deepTauVSe(nTaus, default_value);
    std::vector<float> deepTauVSmu(nTaus, default_value);
    std::vector<float> deepTauVSjet(nTaus, default_value);

    // source: RecoTauTag/RecoTau/plugins/PFTauTransverseImpactParameters.cc
    std::vector<float> dxy(nTaus, default_value);
    std::vector<float> dxy_error(nTaus, default_value);
    std::vector<float> ip3d(nTaus, default_value);
    std::vector<float> ip3d_error(nTaus, default_value);
    std::vector<float> hasSecondaryVertex(nTaus, default_value);
    std::vector<float> flightLength_x(nTaus, default_value);
    std::vector<float> flightLength_y(nTaus, default_value);
    std::vector<float> flightLength_z(nTaus, default_value);
    std::vector<float> flightLengthSig(nTaus, default_value);
    std::vector<float> secondaryVertex_x(nTaus, default_value);
    std::vector<float> secondaryVertex_y(nTaus, default_value);
    std::vector<float> secondaryVertex_z(nTaus, default_value);

    if (tausHandle.isValid() || !(this->skipNonExistingSrc_)) {
      const auto& tausProductId = tausHandle.id();
      const auto& tausIPHandle = event.getHandle(tauIPToken_);
      const auto& deepTauVSeMapHandle = event.getHandle(deepTauVSeToken_);
      const auto& deepTauVSmuMapHandle = event.getHandle(deepTauVSmuToken_);
      const auto& deepTauVSjetMapHandle = event.getHandle(deepTauVSjetToken_);

      for (size_t tau_index = 0; tau_index < nTaus; ++tau_index) {
        if (deepTauVSeMapHandle.isValid() || !(this->skipNonExistingSrc_)) {
          deepTauVSe[tau_index] = deepTauVSeMapHandle->get(tausProductId, tau_index).rawValues.at(0);
        } else {
          edm::LogWarning("HLTTauTableProducer") << " Invalid handle for DeeTauVse score input collection";
        }

        if (deepTauVSmuMapHandle.isValid() || !(this->skipNonExistingSrc_)) {
          deepTauVSmu[tau_index] = deepTauVSmuMapHandle->get(tausProductId, tau_index).rawValues.at(0);
        } else {
          edm::LogWarning("HLTTauTableProducer") << " Invalid handle for DeeTauVsMu score input collection";
        }

        if (deepTauVSjetMapHandle.isValid() || !(this->skipNonExistingSrc_)) {
          deepTauVSjet[tau_index] = deepTauVSjetMapHandle->get(tausProductId, tau_index).rawValues.at(0);
        } else {
          edm::LogWarning("HLTTauTableProducer") << " Invalid handle for DeeTauVsJet score input collection";
        }

        if (tausIPHandle.isValid() || !(this->skipNonExistingSrc_)) {
          dxy[tau_index] = tausIPHandle->value(tau_index)->dxy();
          dxy_error[tau_index] = tausIPHandle->value(tau_index)->dxy_error();
          ip3d[tau_index] = tausIPHandle->value(tau_index)->ip3d();
          ip3d_error[tau_index] = tausIPHandle->value(tau_index)->ip3d_error();
          hasSecondaryVertex[tau_index] = tausIPHandle->value(tau_index)->hasSecondaryVertex();
          flightLength_x[tau_index] = tausIPHandle->value(tau_index)->flightLength().x();
          flightLength_y[tau_index] = tausIPHandle->value(tau_index)->flightLength().y();
          flightLength_z[tau_index] = tausIPHandle->value(tau_index)->flightLength().z();
          flightLengthSig[tau_index] = tausIPHandle->value(tau_index)->flightLengthSig();

          if (hasSecondaryVertex[tau_index] > 0) {
            secondaryVertex_x[tau_index] = tausIPHandle->value(tau_index)->secondaryVertex()->x();
            secondaryVertex_y[tau_index] = tausIPHandle->value(tau_index)->secondaryVertex()->y();
            secondaryVertex_z[tau_index] = tausIPHandle->value(tau_index)->secondaryVertex()->z();
          }
        } else {
          edm::LogWarning("HLTTauTableProducer") << " Invalid handle for Tau IP input collection";
        }
      }
    } else {
      edm::LogWarning("HLTTauTableProducer") << " Invalid handle for PFTau candidate input collection";
    }

    auto tauTable = std::make_unique<nanoaod::FlatTable>(nTaus, tableName_, /*singleton*/ false, /*extension*/ true);
    tauTable->addColumn<float>("dxy", dxy, "tau transverse impact parameter", precision_);
    tauTable->addColumn<float>("dxy_error", dxy_error, " dxy_error ", precision_);
    tauTable->addColumn<float>("ip3d", ip3d, " ip3d ", precision_);
    tauTable->addColumn<float>("ip3d_error", ip3d_error, " ip3d_error ", precision_);
    tauTable->addColumn<float>("hasSecondaryVertex", hasSecondaryVertex, " hasSecondaryVertex ", precision_);
    tauTable->addColumn<float>("flightLength_x", flightLength_x, "flightLength_x", precision_);
    tauTable->addColumn<float>("flightLength_y", flightLength_y, "flightLength_y", precision_);
    tauTable->addColumn<float>("flightLength_z", flightLength_z, "flightLength_z", precision_);
    tauTable->addColumn<float>("flightLengthSig", flightLengthSig, "flightLengthSig", precision_);
    tauTable->addColumn<float>("secondaryVertex_x", secondaryVertex_x, "secondaryVertex_x", precision_);
    tauTable->addColumn<float>("secondaryVertex_y", secondaryVertex_y, "secondaryVertex_y", precision_);
    tauTable->addColumn<float>("secondaryVertex_z", secondaryVertex_z, "secondaryVertex_z", precision_);
    tauTable->addColumn<float>("deepTauVSe", deepTauVSe, "tau vs electron discriminator", precision_);
    tauTable->addColumn<float>("deepTauVSmu", deepTauVSmu, "tau vs muon discriminator", precision_);
    tauTable->addColumn<float>("deepTauVSjet", deepTauVSjet, "tau vs jet discriminator", precision_);

    event.put(std::move(tauTable), tableName_);
  }

private:
  const std::string tableName_;
  const bool skipNonExistingSrc_;
  const edm::EDGetTokenT<TauCollection> tauToken_;
  const edm::EDGetTokenT<TauIPVector> tauIPToken_;
  const edm::EDGetTokenT<TauDiscrMap> deepTauVSeToken_, deepTauVSmuToken_, deepTauVSjetToken_;
  const unsigned int precision_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTTauTableProducer);
