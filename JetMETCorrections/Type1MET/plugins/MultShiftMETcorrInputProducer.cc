#include "JetMETCorrections/Type1MET/plugins/MultShiftMETcorrInputProducer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/View.h"

#include <TString.h>

MultShiftMETcorrInputProducer::MultShiftMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
//  token_      = consumes<edm::View<reco::MET> >(cfg.getParameter<edm::InputTag>("src"));
  pflowToken_ = consumes<std::vector<reco::PFCandidate> >(cfg.getParameter<edm::InputTag>("srcPFlow"));
  
  cfgCorrParameters_ = cfg.getParameter<std::vector<edm::ParameterSet> >("parameters");
  etaMin_.clear(); 
  etaMax_.clear();
  type_.clear(); 
  for (std::vector<edm::ParameterSet>::const_iterator v = cfgCorrParameters_.begin(); v!=cfgCorrParameters_.end(); v++) {
    produces<CorrMETData>(v->getParameter<std::string>("name"));
    TString corrPxFormula = v->getParameter<std::string>("fx");
    TString corrPyFormula = v->getParameter<std::string>("fy");
    std::vector<double> corrPxParams = v->getParameter<std::vector<double> >("px");
    std::vector<double> corrPyParams = v->getParameter<std::vector<double> >("py");
    TF1 * fx = new TF1(std::string(moduleLabel_).append("_").append(v->getParameter<std::string>("name")).append("_corrPx").c_str(), v->getParameter<std::string>("fx").c_str());    
    TF1 * fy = new TF1(std::string(moduleLabel_).append("_").append(v->getParameter<std::string>("name")).append("_corrPy").c_str(), v->getParameter<std::string>("fy").c_str());    
    for (unsigned i=0; i<corrPxParams.size();i++) fx->SetParameter(i, corrPxParams[i]);
    for (unsigned i=0; i<corrPyParams.size();i++) fy->SetParameter(i, corrPyParams[i]);
    formula_x_.push_back(fx);
    formula_y_.push_back(fy);
    counts_.push_back(0);
    etaMin_.push_back(v->getParameter<double>("etaMin"));
    etaMax_.push_back(v->getParameter<double>("etaMax"));
    type_.push_back(v->getParameter<int>("type"));
  }
}

MultShiftMETcorrInputProducer::~MultShiftMETcorrInputProducer()
{
  for (unsigned i=0; i<formula_x_.size();i++) delete formula_x_[i];
  for (unsigned i=0; i<formula_y_.size();i++) delete formula_y_[i];
}

void MultShiftMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{

//  typedef edm::View<reco::MET> METView;
//  edm::Handle<METView> met;
//  evt.getByToken(token_, met);
//  if ( met->size() != 1 ) 
//    throw cms::Exception("MultShiftMETcorrInputProducer::produce") 
//      << "Failed to find unique MET object !!\n";
  for (unsigned i=0;i<counts_.size();i++) counts_[i]=0;
//  typedef std::vector<reco::PFCandidate>  pfCand;
  edm::Handle<std::vector<reco::PFCandidate> > particleFlow;
  evt.getByToken(pflowToken_, particleFlow);
  for (unsigned i = 0; i < particleFlow->size(); ++i) {
    const reco::PFCandidate& c = particleFlow->at(i);
    for (unsigned j=0; j<type_.size(); j++) {
      if (c.particleId()==type_[j]) {
        if ((c.eta()>etaMin_[j]) and(c.eta()<etaMax_[j])) {
          counts_[j]+=1;
          continue;
        }
      }
    } 
  }
  for (std::vector<edm::ParameterSet>::const_iterator v = cfgCorrParameters_.begin(); v!=cfgCorrParameters_.end(); v++) {
    unsigned j=v-cfgCorrParameters_.begin();
    std::auto_ptr<CorrMETData> metCorr(new CorrMETData());
    metCorr->mex = -formula_x_[j]->Eval(counts_[j]);
    metCorr->mey = -formula_y_[j]->Eval(counts_[j]);  
//    std::cout<<v->getParameter<std::string>("name")<<" "<<counts_[j]<<" "<<-formula_x_[j]->Eval(counts_[j])<<" "<<-formula_y_[j]->Eval(counts_[j])<<std::endl;
    evt.put(metCorr, v->getParameter<std::string>("name"));
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MultShiftMETcorrInputProducer);
