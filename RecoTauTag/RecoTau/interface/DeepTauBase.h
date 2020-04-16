#ifndef RecoTauTag_RecoTau_DeepTauBase_h
#define RecoTauTag_RecoTau_DeepTauBase_h

/*
 * \class DeepTauBase
 *
 * Definition of the base class for tau identification using Deep NN.
 *
 * \author Konstantin Androsov, INFN Pisa
 * \author Maria Rosaria Di Domenico, University of Siena & INFN Pisa
 */

#include <Math/VectorUtil.h>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "tensorflow/core/util/memmapped_file_system.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <TF1.h>

namespace deep_tau {

  class TauWPThreshold {
  public:
    explicit TauWPThreshold(const std::string& cut_str);
    double operator()(const pat::Tau& tau) const;

  private:
    std::unique_ptr<TF1> fn_;
    double value_;
  };

  class DeepTauCache {
  public:
    using GraphPtr = std::shared_ptr<tensorflow::GraphDef>;

    DeepTauCache(const std::map<std::string, std::string>& graph_names, bool mem_mapped);
    ~DeepTauCache();

    // A Session allows concurrent calls to Run(), though a Session must
    // be created / extended by a single thread.
    tensorflow::Session& getSession(const std::string& name = "") const { return *sessions_.at(name); }
    const tensorflow::GraphDef& getGraph(const std::string& name = "") const { return *graphs_.at(name); }

  private:
    std::map<std::string, GraphPtr> graphs_;
    std::map<std::string, tensorflow::Session*> sessions_;
    std::map<std::string, std::unique_ptr<tensorflow::MemmappedEnv>> memmappedEnv_;
  };

  class DeepTauBase : public edm::stream::EDProducer<edm::GlobalCache<DeepTauCache>> {
  public:
    using TauType = pat::Tau;
    using TauDiscriminator = reco::TauDiscriminatorContainer;
    using TauCollection = std::vector<TauType>;
    using TauRef = edm::Ref<TauCollection>;
    using TauRefProd = edm::RefProd<TauCollection>;
    using ElectronCollection = pat::ElectronCollection;
    using MuonCollection = pat::MuonCollection;
    using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
    using Cutter = TauWPThreshold;
    using CutterPtr = std::unique_ptr<Cutter>;
    using WPList = std::vector<CutterPtr>;

    struct Output {
      std::vector<size_t> num_, den_;

      Output(const std::vector<size_t>& num, const std::vector<size_t>& den) : num_(num), den_(den) {}

      std::unique_ptr<TauDiscriminator> get_value(const edm::Handle<TauCollection>& taus,
                                                  const tensorflow::Tensor& pred,
                                                  const WPList& working_points) const;
    };

    using OutputCollection = std::map<std::string, Output>;

    DeepTauBase(const edm::ParameterSet& cfg, const OutputCollection& outputs, const DeepTauCache* cache);
    ~DeepTauBase() override {}

    void produce(edm::Event& event, const edm::EventSetup& es) override;

    static std::unique_ptr<DeepTauCache> initializeGlobalCache(const edm::ParameterSet& cfg);
    static void globalEndJob(const DeepTauCache* cache) {}

  private:
    virtual tensorflow::Tensor getPredictions(edm::Event& event,
                                              const edm::EventSetup& es,
                                              edm::Handle<TauCollection> taus) = 0;
    virtual void createOutputs(edm::Event& event, const tensorflow::Tensor& pred, edm::Handle<TauCollection> taus);

  protected:
    edm::EDGetTokenT<TauCollection> tausToken_;
    edm::EDGetTokenT<pat::PackedCandidateCollection> pfcandToken_;
    edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
    std::map<std::string, WPList> workingPoints_;
    OutputCollection outputs_;
    const DeepTauCache* cache_;
  };

}  // namespace deep_tau

#endif
