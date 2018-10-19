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
        for(size_t num_elem : num)
            x += pred.matrix<float>()(tau_index, num_elem);
        if(x != 0 && den.size() > 0) {
            float den_val = 0;
            for(size_t den_elem : den)
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

DeepTauBase::DeepTauBase(const edm::ParameterSet& cfg, const OutputCollection& outputCollection) :
    taus_token(consumes<TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
    graphName(edm::FileInPath(cfg.getParameter<std::string>("graph_file")).fullPath()),
    graph(tensorflow::loadGraphDef(graphName)),
    session(tensorflow::createSession(graph.get())),
    outputs(outputCollection)
{
    for(const auto& output_desc : outputs) {
        produces<TauDiscriminator>(output_desc.first);
        const auto& cut_pset = cfg.getParameter<edm::ParameterSet>(output_desc.first + "WP");
        for(const std::string& wp_name : cut_pset.getParameterNames()) {
            const auto& cut_str = cut_pset.getParameter<std::string>(wp_name);
            working_points[output_desc.first][wp_name] = std::make_unique<Cutter>(cut_str);
            produces<TauDiscriminator>(output_desc.first + wp_name);
        }
    }
}

DeepTauBase::~DeepTauBase()
{
    tensorflow::closeSession(session);
}

void DeepTauBase::produce(edm::Event& event, const edm::EventSetup& es)
{
    event.getByToken(taus_token, taus);
    const tensorflow::Tensor& pred = GetPredictions(event, es);
    CreateOutputs(event, pred);
}

void DeepTauBase::CreateOutputs(edm::Event& event, const tensorflow::Tensor& pred)
{
    for(const auto& output_desc : outputs) {
        auto result_map = output_desc.second.get_value(taus, pred, working_points.at(output_desc.first));
        for(auto& result : result_map)
            event.put(std::move(result.second), output_desc.first + result.first);
    }
}

} // namespace deep_tau
