#include "JetMETCorrections/Type1MET/plugins/Type0PFMETcorrInputProducer.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "TauAnalysis/TauIdEfficiency/interface/tauPtResAuxFunctions.h"

#include <TMath.h>

typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::VertexCollection, reco::PFCandidateCollection, float> > 
  PFCandidateToVertexAssociationMap;
typedef std::vector<std::pair<reco::PFCandidateRef, float> > PFCandidateQualityPairVector;

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
  //std::cout << "<Type0PFMETcorrInputProducer::produce>:" << std::endl;

  edm::Handle<reco::VertexCollection> hardScatterVertex;
  evt.getByLabel(srcHardScatterVertex_, hardScatterVertex);

  edm::Handle<PFCandidateToVertexAssociationMap> pfCandidateToVertexAssociations;
  evt.getByLabel(srcPFCandidateToVertexAssociations_, pfCandidateToVertexAssociations);

  std::auto_ptr<CorrMETData> pfMEtCorrection(new CorrMETData());

  for ( PFCandidateToVertexAssociationMap::const_iterator pfCandidateToVertexAssociation = pfCandidateToVertexAssociations->begin();
	pfCandidateToVertexAssociation != pfCandidateToVertexAssociations->end(); ++pfCandidateToVertexAssociation ) {
    reco::VertexRef vertex = pfCandidateToVertexAssociation->key;
    const PFCandidateQualityPairVector& pfCandidates_vertex = pfCandidateToVertexAssociation->val;
    
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
      for ( PFCandidateQualityPairVector::const_iterator pfCandidate_vertex = pfCandidates_vertex.begin();
	    pfCandidate_vertex != pfCandidates_vertex.end(); ++pfCandidate_vertex ) {
	const reco::PFCandidate& pfCandidate = (*pfCandidate_vertex->first);
	if ( pfCandidate.particleId() == reco::PFCandidate::h  ||
	     pfCandidate.particleId() == reco::PFCandidate::e  ||
	     pfCandidate.particleId() == reco::PFCandidate::mu ) {
	  //std::cout << "adding " << getPFCandidateType(pfCandidate) << ": Pt = " << pfCandidate.pt() << "," 
	  //	      << " eta = " << pfCandidate.eta() << ", phi = " << pfCandidate.phi() 
	  //	      << " (px = " << pfCandidate.px() << ", py = " << pfCandidate.py() << ")" << std::endl;
	  sumChargedPFCandP4_vertex += pfCandidate.p4();
	}
      }
      
      //std::cout << "vertex: z = " << vertex->position().z() << " --> sumChargedPFCand: "
      //	  << " px = " << sumChargedPFCandP4_vertex.px() << ", py = " << sumChargedPFCandP4_vertex.py() << std::endl;
      
      double pt = sumChargedPFCandP4_vertex.pt();
      double phi = sumChargedPFCandP4_vertex.phi();
      double ptCorr = correction_->Eval(pt);
      double pxCorr = TMath::Cos(phi)*ptCorr;
      double pyCorr = TMath::Sin(phi)*ptCorr;
      //std::cout << "correction (vertex): px = " << pxCorr << ", py = " << pyCorr << std::endl;

      pfMEtCorrection->mex += pxCorr;
      pfMEtCorrection->mey += pyCorr;
    }
  }

  //std::cout << "--> correction: px = " << pfMEtCorrection->mex << ", py = " << pfMEtCorrection->mey << std::endl;

  evt.put(pfMEtCorrection);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(Type0PFMETcorrInputProducer);
