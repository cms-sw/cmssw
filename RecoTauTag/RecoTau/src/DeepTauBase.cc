/*
 * \class DeepTauBase
 *
 * Implementation of the base class for tau identification using Deep NN.
 *
 * \author Konstantin Androsov, INFN Pisa
 * \author Maria Rosaria Di Domenico, University of Siena & INFN Pisa
 */

#include "RecoTauTag/RecoTau/interface/DeepTauBase.h"

namespace deep_tau {

  TauWPThreshold::TauWPThreshold(const std::string& cut_str) {
    bool simple_value = false;
    try {
      size_t pos = 0;
      value_ = std::stod(cut_str, &pos);
      simple_value = (pos == cut_str.size());
    } catch (std::invalid_argument&) {
    } catch (std::out_of_range&) {
    }
    if (!simple_value) {
      static const std::string prefix =
          "[&](double *x, double *p) { const int decayMode = p[0];"
          "const double pt = p[1]; const double eta = p[2];";
      static const int n_params = 3;
      static const auto handler = [](int, Bool_t, const char*, const char*) -> void {};

      const std::string fn_str = prefix + cut_str + "}";
      auto old_handler = SetErrorHandler(handler);
      fn_ = std::make_unique<TF1>("fn_", fn_str.c_str(), 0, 1, n_params);
      SetErrorHandler(old_handler);
      if (!fn_->IsValid())
        throw cms::Exception("TauWPThreshold: invalid formula") << "Invalid WP cut formula = '" << cut_str << "'.";
    }
  }

  double TauWPThreshold::operator()(const pat::Tau& tau) const {
    if (!fn_)
      return value_;
    fn_->SetParameter(0, tau.decayMode());
    fn_->SetParameter(1, tau.pt());
    fn_->SetParameter(2, tau.eta());
    return fn_->Eval(0);
  }

  std::unique_ptr<DeepTauBase::TauDiscriminator> DeepTauBase::Output::get_value(const edm::Handle<TauCollection>& taus,
                                                                                const tensorflow::Tensor& pred,
                                                                                const WPList& working_points) const {
    std::vector<reco::SingleTauDiscriminatorContainer> outputbuffer(taus->size());

    for (size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
      float x = 0;
      for (size_t num_elem : num_)
        x += pred.matrix<float>()(tau_index, num_elem);
      if (x != 0 && !den_.empty()) {
        float den_val = 0;
        for (size_t den_elem : den_)
          den_val += pred.matrix<float>()(tau_index, den_elem);
        x = den_val != 0 ? x / den_val : std::numeric_limits<float>::max();
      }
      outputbuffer[tau_index].rawValues.push_back(x);
      for (const auto& wp : working_points) {
        const bool pass = x > (*wp)(taus->at(tau_index));
        outputbuffer[tau_index].workingPoints.push_back(pass);
      }
    }
    std::unique_ptr<TauDiscriminator> output = std::make_unique<TauDiscriminator>();
    reco::TauDiscriminatorContainer::Filler filler(*output);
    filler.insert(taus, outputbuffer.begin(), outputbuffer.end());
    filler.fill();
    return output;
  }

  DeepTauBase::DeepTauBase(const edm::ParameterSet& cfg,
                           const OutputCollection& outputCollection,
                           const DeepTauCache* cache)
      : tausToken_(consumes<TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        pfcandToken_(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("pfcands"))),
        vtxToken_(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
        outputs_(outputCollection),
        cache_(cache) {
    for (const auto& output_desc : outputs_) {
      produces<TauDiscriminator>(output_desc.first);
      const auto& cut_list = cfg.getParameter<std::vector<std::string>>(output_desc.first + "WP");
      for (const std::string& cut_str : cut_list) {
        workingPoints_[output_desc.first].push_back(std::make_unique<Cutter>(cut_str));
      }
    }
  }

  void DeepTauBase::produce(edm::Event& event, const edm::EventSetup& es) {
    edm::Handle<TauCollection> taus;
    event.getByToken(tausToken_, taus);

    const tensorflow::Tensor& pred = getPredictions(event, taus);
    createOutputs(event, pred, taus);
  }

  void DeepTauBase::createOutputs(edm::Event& event, const tensorflow::Tensor& pred, edm::Handle<TauCollection> taus) {
    for (const auto& output_desc : outputs_) {
      auto result = output_desc.second.get_value(taus, pred, workingPoints_.at(output_desc.first));
      event.put(std::move(result), output_desc.first);
    }
  }

  std::unique_ptr<DeepTauCache> DeepTauBase::initializeGlobalCache(const edm::ParameterSet& cfg) {
    const auto graph_name_vector = cfg.getParameter<std::vector<std::string>>("graph_file");
    std::map<std::string, std::string> graph_names;
    for (const auto& entry : graph_name_vector) {
      const size_t sep_pos = entry.find(':');
      std::string entry_name, graph_file;
      if (sep_pos != std::string::npos) {
        entry_name = entry.substr(0, sep_pos);
        graph_file = entry.substr(sep_pos + 1);
      } else {
        entry_name = "";
        graph_file = entry;
      }
      graph_file = edm::FileInPath(graph_file).fullPath();
      if (graph_names.count(entry_name))
        throw cms::Exception("DeepTauCache") << "Duplicated graph entries";
      graph_names[entry_name] = graph_file;
    }
    bool mem_mapped = cfg.getParameter<bool>("mem_mapped");
    return std::make_unique<DeepTauCache>(graph_names, mem_mapped);
  }

  DeepTauCache::DeepTauCache(const std::map<std::string, std::string>& graph_names, bool mem_mapped) {
    for (const auto& graph_entry : graph_names) {
      tensorflow::SessionOptions options;
      tensorflow::setThreading(options, 1);

      const std::string& entry_name = graph_entry.first;
      const std::string& graph_file = graph_entry.second;
      if (mem_mapped) {
        memmappedEnv_[entry_name] = std::make_unique<tensorflow::MemmappedEnv>(tensorflow::Env::Default());
        const tensorflow::Status mmap_status = memmappedEnv_.at(entry_name)->InitializeFromFile(graph_file);
        if (!mmap_status.ok()) {
          throw cms::Exception("DeepTauCache: unable to initalize memmapped environment for ")
              << graph_file << ". \n"
              << mmap_status.ToString();
        }

        graphs_[entry_name] = std::make_unique<tensorflow::GraphDef>();
        const tensorflow::Status load_graph_status =
            ReadBinaryProto(memmappedEnv_.at(entry_name).get(),
                            tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                            graphs_.at(entry_name).get());
        if (!load_graph_status.ok())
          throw cms::Exception("DeepTauCache: unable to load graph from ") << graph_file << ". \n"
                                                                           << load_graph_status.ToString();

        options.config.mutable_graph_options()->mutable_optimizer_options()->set_opt_level(
            ::tensorflow::OptimizerOptions::L0);
        options.env = memmappedEnv_.at(entry_name).get();

        sessions_[entry_name] = tensorflow::createSession(graphs_.at(entry_name).get(), options);

      } else {
        graphs_[entry_name].reset(tensorflow::loadGraphDef(graph_file));
        sessions_[entry_name] = tensorflow::createSession(graphs_.at(entry_name).get(), options);
      }
    }
  }

  DeepTauCache::~DeepTauCache() {
    for (auto& session_entry : sessions_)
      tensorflow::closeSession(session_entry.second);
  }

}  // namespace deep_tau
