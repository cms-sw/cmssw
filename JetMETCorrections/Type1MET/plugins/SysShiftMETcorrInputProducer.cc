#include "JetMETCorrections/Type1MET/plugins/SysShiftMETcorrInputProducer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/View.h"

#include <TString.h>

SysShiftMETcorrInputProducer::SysShiftMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    corrPx_(0),
    corrPy_(0)
{
  src_ = cfg.getParameter<edm::InputTag>("src");
  
  edm::ParameterSet cfgCorrParameter = cfg.getParameter<edm::ParameterSet>("parameter");
  TString corrPxFormula = cfgCorrParameter.getParameter<std::string>("px");
  TString corrPyFormula = cfgCorrParameter.getParameter<std::string>("py").data();
  if ( corrPxFormula.Contains("Nvtx") || corrPyFormula.Contains("Nvtx") ) {
    srcVertices_ = cfg.getParameter<edm::InputTag>("srcVertices");
    if ( srcVertices_.label() == "" )
      throw cms::Exception("SysShiftMETcorrInputProducer")
	<< "Configuration Parameter 'srcVertices' must be non-empty !!\n";
  }
  
  corrPxFormula.ReplaceAll("sumEt", "x");
  corrPxFormula.ReplaceAll("Nvtx", "y");
  std::string corrPxName = std::string(moduleLabel_).append("_corrPx");
  corrPx_ = new TFormula(corrPxName.data(), corrPxFormula.Data());

  corrPyFormula.ReplaceAll("sumEt", "x");
  corrPyFormula.ReplaceAll("Nvtx", "y");
  std::string corrPyName = std::string(moduleLabel_).append("_corrPy");
  corrPy_ = new TFormula(corrPyName.data(), corrPyFormula.Data());
  
  produces<CorrMETData>();
}

SysShiftMETcorrInputProducer::~SysShiftMETcorrInputProducer()
{
// nothing to be done yet...
}

void SysShiftMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  //std::cout << "<SysShiftMETcorrInputProducer::produce>:" << std::endl;

  typedef edm::View<reco::MET> METView;
  edm::Handle<METView> met;
  evt.getByLabel(src_, met);
  if ( met->size() != 1 ) 
    throw cms::Exception("SysShiftMETcorrInputProducer::produce") 
      << "Failed to find unique MET object !!\n";

  double sumEt = met->front().sumEt();
  //std::cout << " sumEt = " << sumEt << std::endl;

  size_t Nvtx = 0;
  if ( srcVertices_.label() != "" ) {
    edm::Handle<reco::VertexCollection> vertices;
    evt.getByLabel(srcVertices_, vertices);
    Nvtx = vertices->size();
  }
  //std::cout << " Nvtx = " << Nvtx << std::endl;

  std::auto_ptr<CorrMETData> metCorr(new CorrMETData());
  metCorr->mex = -corrPx_->Eval(sumEt, Nvtx);
  metCorr->mey = -corrPy_->Eval(sumEt, Nvtx);
  //std::cout << "--> metCorr: Px = " << metCorr->mex << ", Py = " << metCorr->mey << std::endl;
  
  evt.put(metCorr);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SysShiftMETcorrInputProducer);
