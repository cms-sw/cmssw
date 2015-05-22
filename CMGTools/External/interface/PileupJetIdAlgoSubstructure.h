//--------------------------------------------------------------------------------------------------
// $Id $
//
// PileupJetIdAlgo
//
// Author: P. Musella, P. Harris
//--------------------------------------------------------------------------------------------------

#ifndef PileupJetIdAlgoSubstructure_h
#define PileupJetIdAlgoSubstructure_h

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "PileupJetIdentifierSubstructure.h"

// ----------------------------------------------------------------------------------------------------
class PileupJetIdAlgoSubstructure {
public:
	enum version_t { USER=-1, PHILv0=0 };
	
	PileupJetIdAlgoSubstructure(int version=PHILv0, const std::string & tmvaWeight="", const std::string & tmvaMethod="", 
			Float_t impactParTkThreshod_=1., const std::vector<std::string> & tmvaVariables= std::vector<std::string>());
	PileupJetIdAlgoSubstructure(const edm::ParameterSet & ps); 
	~PileupJetIdAlgoSubstructure(); 
	
	PileupJetIdentifierSubstructure computeIdVariables(const reco::Jet * jet, 
					       float jec, const reco::Vertex *, const reco::VertexCollection &,
					       bool calculateMva=false);
	
	void set(const PileupJetIdentifierSubstructure &);
	PileupJetIdentifierSubstructure computeMva();
	const std::string method() const { return tmvaMethod_; }
	
	std::string dumpVariables() const;

	typedef std::map<std::string,std::pair<float *,float> > variables_list_t;

	std::pair<int,int> getJetIdKey(float jetPt, float jetEta);
	int computeCutIDflag(float betaStarClassic,float dR2Mean,float nvtx, float jetPt, float jetEta);
	int computeIDflag   (float mva, float jetPt, float jetEta);
	int computeIDflag   (float mva,int ptId,int etaId);

	/// const PileupJetIdentifierSubstructure::variables_list_t & getVariables() const { return variables_; };
	const variables_list_t & getVariables() const { return variables_; };
	//Generic tools
	void  assign(const std::vector<float> & vec, float & a, float & b, float & c, float & d );
	void  setPtEtaPhi(const reco::Candidate & p, float & pt, float & eta, float &phi );

protected:

	void setup(); 
	void runMva(); 
	void bookReader();	
	void resetVariables();
	void initVariables();

	
	PileupJetIdentifierSubstructure internalId_;
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
