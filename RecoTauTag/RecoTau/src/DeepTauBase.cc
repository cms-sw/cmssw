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

TauWPThreshold::TauWPThreshold(const std::string& cut_str)
{
    bool simple_value = false;
    try {
        size_t pos = 0;
        value_ = std::stod(cut_str, &pos);
        simple_value = (pos == cut_str.size());
    } catch(std::invalid_argument&) {
    } catch(std::out_of_range&) {
    }
    if(!simple_value) {
        static const std::string prefix = "[&](double *x, double *p) { const int decayMode = p[0];"
                                          "const double pt = p[1]; const double eta = p[2];";
        static const int n_params = 3;
        static const auto handler = [](int, Bool_t, const char*, const char*) -> void {};

        const std::string fn_str = prefix + cut_str + "}";
        auto old_handler = SetErrorHandler(handler);
        fn_ = std::make_unique<TF1>("fn_", fn_str.c_str(), 0, 1, n_params);
        SetErrorHandler(old_handler);
        if(!fn_->IsValid())
            throw cms::Exception("TauWPThreshold: invalid formula") << "Invalid WP cut formula = '" << cut_str << "'.";
    }
}

double TauWPThreshold::operator()(const pat::Tau& tau) const
{
    if(!fn_)
        return value_;
    fn_->SetParameter(0, tau.decayMode());
    fn_->SetParameter(1, tau.pt());
    fn_->SetParameter(2, tau.eta());
    return fn_->Eval(0);
}

DeepTauBase::Output::ResultMap DeepTauBase::Output::get_value(const edm::Handle<TauCollection>& taus,
                                                              const tensorflow::Tensor& pred,
                                                              const WPMap& working_points) const
{
    ResultMap output;
    output[""] = std::make_unique<TauDiscriminator>(TauRefProd(taus));
    for(const auto& wp : working_points)
        output[wp.first] = std::make_unique<TauDiscriminator>(TauRefProd(taus));

    for(size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
        float x = 0;
        for(size_t num_elem : num_)
            x += pred.matrix<float>()(tau_index, num_elem);
        if(x != 0 && den_.size() > 0) {
            float den_val = 0;
            for(size_t den_elem : den_)
                den_val += pred.matrix<float>()(tau_index, den_elem);
            x = den_val != 0 ? x / den_val : std::numeric_limits<float>::max();
        }
        output[""]->setValue(tau_index, x);
        for(const auto& wp : working_points) {
            const auto& tau = taus->at(tau_index);
            const bool pass = x > (*wp.second)(tau);
            output[wp.first]->setValue(tau_index, pass);
        }
    }
    return output;
}

DeepTauBase::DeepTauBase(const edm::ParameterSet& cfg, const OutputCollection& outputCollection, const DeepTauCache* cache) :
    tausToken_(consumes<TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
    outputs_(outputCollection),
    cache_(cache)
{
    for(const auto& output_desc : outputs_) {
        produces<TauDiscriminator>(output_desc.first);
        const auto& cut_pset = cfg.getParameter<edm::ParameterSet>(output_desc.first + "WP");
        for(const std::string& wp_name : cut_pset.getParameterNames()) {
            const auto& cut_str = cut_pset.getParameter<std::string>(wp_name);
            workingPoints_[output_desc.first][wp_name] = std::make_unique<Cutter>(cut_str);
            produces<TauDiscriminator>(output_desc.first + wp_name);
        }
    }
}

void DeepTauBase::produce(edm::Event& event, const edm::EventSetup& es)
{
    edm::Handle<TauCollection> taus;
    event.getByToken(tausToken_, taus);

    const tensorflow::Tensor& pred = getPredictions(event, es, taus);
    createOutputs(event, pred, taus);
}

void DeepTauBase::createOutputs(edm::Event& event, const tensorflow::Tensor& pred, edm::Handle<TauCollection> taus)
{
    for(const auto& output_desc : outputs_) {
        auto result_map = output_desc.second.get_value(taus, pred, workingPoints_.at(output_desc.first));
        for(auto& result : result_map)
            event.put(std::move(result.second), output_desc.first + result.first);
    }
}

std::unique_ptr<DeepTauCache> DeepTauBase::initializeGlobalCache(const edm::ParameterSet& cfg )
{
    std::string graph_name = edm::FileInPath(cfg.getParameter<std::string>("graph_file")).fullPath();
    bool mem_mapped = cfg.getParameter<bool>("mem_mapped");
    return std::make_unique<DeepTauCache>(graph_name, mem_mapped);
}

DeepTauCache::DeepTauCache(const std::string& graph_name, bool mem_mapped)
{
    tensorflow::SessionOptions options;
    tensorflow::setThreading(options, 1, "no_threads");

    if(mem_mapped) {
        memmappedEnv_ = std::make_unique<tensorflow::MemmappedEnv>(tensorflow::Env::Default());
        const tensorflow::Status mmap_status = memmappedEnv_.get()->InitializeFromFile(graph_name);
        if(!mmap_status.ok())
            throw cms::Exception("DeepTauCache: unable to initalize memmapped environment for ") << graph_name << ". \n"
                                                                                                 << mmap_status.ToString();

        graph_ = std::make_unique<tensorflow::GraphDef>();
        const tensorflow::Status load_graph_status = ReadBinaryProto(memmappedEnv_.get(),
                                                                     tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                                                                     graph_.get());
        if(!load_graph_status.ok())
            throw cms::Exception("DeepTauCache: unable to load graph_ from ") << graph_name << ". \n"
                                                                             << mmap_status.ToString();
        options.config.mutable_graph_options()->mutable_optimizer_options()->set_opt_level(::tensorflow::OptimizerOptions::L0);
        options.env = memmappedEnv_.get();

        session_ = tensorflow::createSession(graph_.get(), options);

    } else {
        graph_.reset(tensorflow::loadGraphDef(graph_name));
        session_ = tensorflow::createSession(graph_.get(), options);
      }
}

DeepTauCache::~DeepTauCache()
{
    tensorflow::closeSession(session_);
}

} // namespace deep_tau
