//--------------------------------------------------------------------------------------------------
//
// PileupJetIdAlgo
//
// Author: P. Musella, P. Harris
//--------------------------------------------------------------------------------------------------

#ifndef RecoJets_JetProducers_plugins_PileupJetIdAlgo_h
#define RecoJets_JetProducers_plugins_PileupJetIdAlgo_h

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"


// ----------------------------------------------------------------------------------------------------
class PileupJetIdAlgo {
public:
	enum version_t { USER=-1, PHILv0=0 };
	
	PileupJetIdAlgo(int version=PHILv0, const std::string & tmvaWeight="", const std::string & tmvaMethod="", 
			Float_t impactParTkThreshod_=1., const std::vector<std::string> & tmvaVariables= std::vector<std::string>());
	PileupJetIdAlgo(const edm::ParameterSet & ps); 
	~PileupJetIdAlgo(); 
	
	PileupJetIdentifier computeIdVariables(const reco::Jet * jet, 
					       float jec, const reco::Vertex *, const reco::VertexCollection &,
					       bool calculateMva=false);

	void set(const PileupJetIdentifier &);
	PileupJetIdentifier computeMva();
	const std::string method() const { return tmvaMethod_; }
	
	std::string dumpVariables() const;

	typedef std::map<std::string,std::pair<float *,float> > variables_list_t;

	std::pair<int,int> getJetIdKey(float jetPt, float jetEta);
	int computeCutIDflag(float betaStarClassic,float dR2Mean,float nvtx, float jetPt, float jetEta);
	int computeIDflag   (float mva, float jetPt, float jetEta);
	int computeIDflag   (float mva,int ptId,int etaId);

	/// const PileupJetIdentifier::variables_list_t & getVariables() const { return variables_; };
	const variables_list_t & getVariables() const { return variables_; };
	
protected:

	void setup(); 
	void runMva(); 
	void bookReader();	
	void resetVariables();
	void initVariables();

	
	PileupJetIdentifier internalId_;
	variables_list_t variables_;

	TMVA::Reader * reader_;
	std::string    tmvaWeights_, tmvaMethod_; 
	std::vector<std::string>  tmvaVariables_;
	std::vector<std::string>  tmvaSpectators_;
	std::map<std::string,std::string>  tmvaNames_;
	
	Int_t   version_;
	Float_t impactParTkThreshod_;
	bool    cutBased_;
	Float_t mvacut_     [3][4][4]; //Keep the array fixed
	Float_t rmsCut_     [3][4][4]; //Keep the array fixed
	Float_t betaStarCut_[3][4][4]; //Keep the array fixed
};

#endif
