// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      LeptonTagInfoCollectionProducer
//
/**\class LeptonTagInfoCollectionProducer LeptonTagInfoCollectionProducer.cc PhysicsTools/PatAlgos/plugins/PNETLeptonProducer.cc


*/
//
// Original Author:  Sergio Sanchez Cruz
//         Created:  Mon, 15 May 2023 08:32:03 GMT
//
//

#ifndef PhysicsTools_PatAlgos_LeptonTagInfoCollectionProducer_h
#define PhysicsTools_PatAlgos_LeptonTagInfoCollectionProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/BTauReco/interface/DeepBoostedJetFeatures.h" 
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h" // this is flexible enough for our purposes
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace pat {
template <class Features, class T>

class FeaturesLepInfo {
public:
FeaturesLepInfo() {}

FeaturesLepInfo(const Features& features, const edm::RefToBase<T>& lep_ref)
    : features_(features), lep_ref_(lep_ref) {} 
    
    
    edm::RefToBase<T> lep() const { return lep_ref_; }
    
    const Features& features() const { return features_; }
    
    ~FeaturesLepInfo() {}
    FeaturesLepInfo* clone(void) const { return new FeaturesLepInfo(*this); }
    
    CMS_CLASS_VERSION(3);

private:
Features features_;
edm::RefToBase<T> lep_ref_;
};
  
  template <typename T> 
  using LeptonTagInfo = FeaturesLepInfo<btagbtvdeep::DeepBoostedJetFeatures, T> ;
  
  template <typename T>
  using  LeptonTagInfoCollection = std::vector<LeptonTagInfo<T>> ;
  
  template<typename T2> 
  using varWithName = std::pair<std::string, StringObjectFunction<T2,true>> ;

  template<typename T2> 
  using extVarWithName = std::pair<std::string, edm::EDGetTokenT<edm::ValueMap<T2>>> ;
  
  template <typename T>  class LeptonTagInfoCollectionProducer : public edm::stream::EDProducer<> {
  public:
    explicit LeptonTagInfoCollectionProducer(const edm::ParameterSet& iConfig);
    ~LeptonTagInfoCollectionProducer() override{};
    
  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    template <typename T2> void parse_vars_into(const edm::ParameterSet &, std::vector<std::unique_ptr<varWithName<T2>>>&);
    template <typename T2> void parse_extvars_into(const edm::ParameterSet &, std::vector<std::unique_ptr<extVarWithName<T2>>>&);
    void fill_lepton_features(const T&, btagbtvdeep::DeepBoostedJetFeatures &);
    void fill_lepton_extfeatures(const edm::RefToBase<T>&, btagbtvdeep::DeepBoostedJetFeatures &, edm::Event &);
    void fill_pf_features(const T&, btagbtvdeep::DeepBoostedJetFeatures &);
    void fill_sv_features(const T&, btagbtvdeep::DeepBoostedJetFeatures &);

    edm::EDGetTokenT<edm::View<T>> src_token_;
    edm::EDGetTokenT<pat::PackedCandidateCollection> pf_token_;
    edm::EDGetTokenT<reco::VertexCompositePtrCandidateCollection> sv_token_;
    edm::EDGetTokenT<std::vector<reco::Vertex>> pv_token_;


    edm::ParameterSet lepton_varsPSet_;
    edm::ParameterSet lepton_varsExtPSet_;
    edm::ParameterSet pf_varsPSet_;
    edm::ParameterSet sv_varsPSet_;


    std::vector<std::unique_ptr<varWithName<T>>> lepton_vars_;
    std::vector<std::unique_ptr<varWithName<pat::PackedCandidate>>> pf_vars_;
    std::vector<std::unique_ptr<varWithName<reco::VertexCompositePtrCandidate>>> sv_vars_;
    edm::Handle<reco::VertexCompositePtrCandidateCollection> svs_;
    edm::Handle<pat::PackedCandidateCollection> pfs_;
    edm::Handle<std::vector<reco::Vertex>> pvs_;
    std::vector<std::unique_ptr<extVarWithName<float>>> extLepton_vars_;

  };
  
} // namespace pat

#endif
