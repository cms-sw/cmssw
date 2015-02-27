#include "JetMETCorrections/Type1MET/plugins/MultShiftMETcorrInputProducer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/View.h"

#include <TString.h>

int MultShiftMETcorrInputProducer::translateTypeToAbsPdgId( reco::PFCandidate::ParticleType type ) {
  switch( type ) {
  case reco::PFCandidate::ParticleType::h: return 211; // pi+
  case reco::PFCandidate::ParticleType::e: return 11;
  case reco::PFCandidate::ParticleType::mu: return 13;
  case reco::PFCandidate::ParticleType::gamma: return 22;
  case reco::PFCandidate::ParticleType::h0: return 130; // K_L0
  case reco::PFCandidate::ParticleType::h_HF: return 1; // dummy pdg code
  case reco::PFCandidate::ParticleType::egamma_HF: return 2; // dummy pdg code
  case reco::PFCandidate::ParticleType::X:
  default: return 0;
  }
}


MultShiftMETcorrInputProducer::MultShiftMETcorrInputProducer(const edm::ParameterSet& cfg):
  pflow_ ( cfg.getParameter< edm::InputTag >("srcPFlow") ),
  vertices_ ( cfg.getParameter< edm::InputTag >("vertexCollection") ),
  moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  
  cfgCorrParameters_ = cfg.getParameter<std::vector<edm::ParameterSet> >("parameters");
  etaMin_.clear(); 
  etaMax_.clear();
  type_.clear(); 
  varType_.clear(); 
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
    sumPt_.push_back(0.);
    etaMin_.push_back(v->getParameter<double>("etaMin"));
    etaMax_.push_back(v->getParameter<double>("etaMax"));
    type_.push_back(v->getParameter<int>("type"));
    varType_.push_back(v->getParameter<int>("varType"));
  }
}

MultShiftMETcorrInputProducer::~MultShiftMETcorrInputProducer()
{
  for (unsigned i=0; i<formula_x_.size();i++) delete formula_x_[i];
  for (unsigned i=0; i<formula_y_.size();i++) delete formula_y_[i];
}

void MultShiftMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  //get primary vertices
  edm::Handle<edm::View<reco::Vertex> > hpv;
  try {
    evt.getByLabel( vertices_, hpv );
  } catch ( cms::Exception & e ) {
    std::cout <<"[MultShiftMETcorrInputProducer] error: " << e.what() << std::endl;
  }
  std::vector<reco::Vertex> goodVertices;
  for (unsigned i = 0; i < hpv->size(); i++) {
    if ( (*hpv)[i].ndof() > 4 &&
       ( fabs((*hpv)[i].z()) <= 24. ) &&
       ( fabs((*hpv)[i].position().rho()) <= 2.0 ) )
       goodVertices.push_back((*hpv)[i]);
  }
  int ngoodVertices = goodVertices.size();

  for (unsigned i=0;i<counts_.size();i++) counts_[i]=0;
  for (unsigned i=0;i<sumPt_.size();i++) sumPt_[i]=0.;
//  typedef std::vector<reco::PFCandidate>  pfCand;
  edm::Handle<edm::View<reco::Candidate> > particleFlow;
  evt.getByLabel(pflow_, particleFlow);
  for (unsigned i = 0; i < particleFlow->size(); ++i) {
    const reco::Candidate& c = particleFlow->at(i);
    for (unsigned j=0; j<type_.size(); j++) {
//      if (c.particleId()==type_[j]) {
      if (abs(c.pdgId())== translateTypeToAbsPdgId(reco::PFCandidate::ParticleType(type_[j]))) {
        if ((c.eta()>etaMin_[j]) and (c.eta()<etaMax_[j])) {
          counts_[j]+=1;
          sumPt_[j]+=c.pt();
          continue;
        }
      }
    } 
  }
  for (std::vector<edm::ParameterSet>::const_iterator v = cfgCorrParameters_.begin(); v!=cfgCorrParameters_.end(); v++) {
    unsigned j=v-cfgCorrParameters_.begin();
    std::auto_ptr<CorrMETData> metCorr(new CorrMETData());
    double val(0.);
    if (varType_[j]==0) {
      val = counts_[j];
    } 
    if (varType_[j]==1) {
      val = ngoodVertices; 
    } 
    if (varType_[j]==2) {
      val = sumPt_[j];
    }
    metCorr->mex = -formula_x_[j]->Eval(val);
    metCorr->mey = -formula_y_[j]->Eval(val);  
//    std::cout<<v->getParameter<std::string>("name")<<" counts: "<<counts_[j]<<" sumPt: "<<sumPt_[j]<<" ngoodVertices: "<<ngoodVertices<<" "<<-formula_x_[j]->Eval(val)<<" "<<-formula_y_[j]->Eval(val)<<std::endl;
    evt.put(metCorr, v->getParameter<std::string>("name"));
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MultShiftMETcorrInputProducer);
