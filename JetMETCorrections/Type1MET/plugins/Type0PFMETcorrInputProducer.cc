#include "JetMETCorrections/Type1MET/plugins/Type0PFMETcorrInputProducer.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "CommonTools/RecoUtils/interface/PFCand_AssoMapAlgos.h"

#include <TMath.h>

Type0PFMETcorrInputProducer::Type0PFMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    correction_(0)
{
  srcPFCandidateToVertexAssociations_ = cfg.getParameter<edm::InputTag>("srcPFCandidateToVertexAssociations");
  srcHardScatterVertex_ = cfg.getParameter<edm::InputTag>("srcHardScatterVertex");

  edm::ParameterSet cfgCorrection_function = cfg.getParameter<edm::ParameterSet>("correction");
  std::string corrFunctionName = std::string(moduleLabel_).append("correction");
  std::string corrFunctionFormula = cfgCorrection_function.getParameter<std::string>("formula");
  correction_ = new TFormula(corrFunctionName.data(), corrFunctionFormula.data());
  int numParameter = correction_->GetNpar();
  for ( int iParameter = 0; iParameter < numParameter; ++iParameter ) {
    std::string parName = Form("par%i", iParameter);
    double parValue = cfgCorrection_function.getParameter<double>(parName);
    correction_->SetParameter(iParameter, parValue);
  }

  minDz_ = cfg.getParameter<double>("minDz");

  produces<CorrMETData>();
}

Type0PFMETcorrInputProducer::~Type0PFMETcorrInputProducer()
{
  delete correction_;
}

void Type0PFMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<reco::VertexCollection> hardScatterVertex;
  evt.getByLabel(srcHardScatterVertex_, hardScatterVertex);

  edm::Handle<PFCandToVertexAssMap> pfCandidateToVertexAssociations;
  evt.getByLabel(srcPFCandidateToVertexAssociations_, pfCandidateToVertexAssociations);

  std::auto_ptr<CorrMETData> pfMEtCorrection(new CorrMETData());

  for ( PFCandToVertexAssMap::const_iterator pfCandidateToVertexAssociation = pfCandidateToVertexAssociations->begin();
	pfCandidateToVertexAssociation != pfCandidateToVertexAssociations->end(); ++pfCandidateToVertexAssociation ) {
    reco::VertexRef vertex = pfCandidateToVertexAssociation->key;
    const PFCandQualityPairVector& pfCandidates_vertex = pfCandidateToVertexAssociation->val;
    
    bool isHardScatterVertex = false;
    for ( reco::VertexCollection::const_iterator hardScatterVertex_i = hardScatterVertex->begin();
	  hardScatterVertex_i != hardScatterVertex->end(); ++hardScatterVertex_i ) {
      if ( TMath::Abs(vertex->position().z() - hardScatterVertex_i->position().z()) < minDz_ ) {
	isHardScatterVertex = true;
	break;
      }
    }
    
    if ( !isHardScatterVertex ) {
      reco::Candidate::LorentzVector sumChargedPFCandP4_vertex;
      for ( PFCandQualityPairVector::const_iterator pfCandidate_vertex = pfCandidates_vertex.begin();
	    pfCandidate_vertex != pfCandidates_vertex.end(); ++pfCandidate_vertex ) {
	const reco::PFCandidate& pfCandidate = (*pfCandidate_vertex->first);
	if ( pfCandidate.particleId() == reco::PFCandidate::h  ||
	     pfCandidate.particleId() == reco::PFCandidate::e  ||
	     pfCandidate.particleId() == reco::PFCandidate::mu ) {
	  sumChargedPFCandP4_vertex += pfCandidate.p4();
	}
      }
      
      double pt = sumChargedPFCandP4_vertex.pt();
      double phi = sumChargedPFCandP4_vertex.phi();
      double ptCorr = correction_->Eval(pt);
      double pxCorr = TMath::Cos(phi)*ptCorr;
      double pyCorr = TMath::Sin(phi)*ptCorr;

      pfMEtCorrection->mex += pxCorr;
      pfMEtCorrection->mey += pyCorr;
    }
  }

  evt.put(pfMEtCorrection);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(Type0PFMETcorrInputProducer);
