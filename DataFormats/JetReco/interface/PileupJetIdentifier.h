//--------------------------------------------------------------------------------------------------
//
// PileupJetIdentifier
//
// Author: P. Musella, P. Harris
//--------------------------------------------------------------------------------------------------

#include <string>
#include <map>

#ifndef RecoJets_JetProducers_interface_PileupJetIdentifier_h
#define RecoJets_JetProducers_interface_PileupJetIdentifier_h

#define DECLARE_VARIABLE(NAME,TYPE)		\
	private: \
	TYPE NAME ## _;			\
	public: \
	const TYPE & NAME() const { return NAME ## _; } \
	void NAME(const TYPE val) { NAME ## _ = val; } 

// ----------------------------------------------------------------------------------------------------
class StoredPileupJetIdentifier {
public:
	friend class PileupJetIdAlgo;
	friend class MVAJetPuId;

	StoredPileupJetIdentifier();
	~StoredPileupJetIdentifier(); 
	
	DECLARE_VARIABLE(jetEta   ,float);
	DECLARE_VARIABLE(jetPt    ,float);

	DECLARE_VARIABLE(nCharged   ,float);
	DECLARE_VARIABLE(nNeutrals,float);

	DECLARE_VARIABLE(dZ         ,float);  
	DECLARE_VARIABLE(nParticles ,float);  

	DECLARE_VARIABLE(dR2Mean    ,float);  /// a.k.a RMS
	void          RMS(const float val) { dR2Mean(val);     }
	const float & RMS() const          { return dR2Mean(); }
	
	DECLARE_VARIABLE(dRMean     ,float);  

	DECLARE_VARIABLE(frac01    ,float);  
	DECLARE_VARIABLE(frac02    ,float);  
	DECLARE_VARIABLE(frac03    ,float);  
	DECLARE_VARIABLE(frac04    ,float);  
	DECLARE_VARIABLE(frac05   ,float);  
	DECLARE_VARIABLE(frac06   ,float);  
	DECLARE_VARIABLE(frac07   ,float);  
	
	DECLARE_VARIABLE(beta       ,float);  
	DECLARE_VARIABLE(betaStar   ,float);  
	DECLARE_VARIABLE(betaClassic   ,float);  
	DECLARE_VARIABLE(betaStarClassic   ,float);  
	
	DECLARE_VARIABLE(ptD        ,float);
	DECLARE_VARIABLE(rho        ,float);
	DECLARE_VARIABLE(jetR       ,float);
	DECLARE_VARIABLE(jetRchg    ,float);
	DECLARE_VARIABLE(dRMatch    ,float);
	DECLARE_VARIABLE(nTrueInt   ,float);

	DECLARE_VARIABLE(nvtx   ,float);  
	DECLARE_VARIABLE(DR_weighted,float);
	DECLARE_VARIABLE(pull,float);
};

// ----------------------------------------------------------------------------------------------------
class PileupJetIdentifier : public StoredPileupJetIdentifier {
public:
	friend class PileupJetIdAlgo;
	friend class MVAJetPuId;

	PileupJetIdentifier();
	~PileupJetIdentifier(); 

	PileupJetIdentifier & operator= (const StoredPileupJetIdentifier & lhs) { ((StoredPileupJetIdentifier &)(*this)) = lhs; return *this;}

	enum Id {
	  kTight  = 0,
	  kMedium = 1,
	  kLoose  = 2
	};

	static bool passJetId(int flag, Id level) { return ( flag & (1 << level) ) != 0 ; }
	
	DECLARE_VARIABLE(idFlag      ,int);
	bool passJetId   (Id level) { return passJetId(idFlag_,level); }	

	DECLARE_VARIABLE(mva      ,float);

	DECLARE_VARIABLE(jetPhi   ,float);
	DECLARE_VARIABLE(jetM     ,float);

	DECLARE_VARIABLE(chgEMfrac  ,float);
	DECLARE_VARIABLE(neuEMfrac  ,float);
	DECLARE_VARIABLE(chgHadrfrac,float);
	DECLARE_VARIABLE(neuHadrfrac,float);
	
	DECLARE_VARIABLE(d0         ,float);   
	DECLARE_VARIABLE(leadPt     ,float);  
	DECLARE_VARIABLE(leadEta    ,float);  
	DECLARE_VARIABLE(leadPhi    ,float);  
	DECLARE_VARIABLE(secondPt   ,float);  
	DECLARE_VARIABLE(secondEta  ,float);  
	DECLARE_VARIABLE(secondPhi  ,float);  
	DECLARE_VARIABLE(leadNeutPt ,float);  
	DECLARE_VARIABLE(leadNeutEta,float);  
	DECLARE_VARIABLE(leadNeutPhi,float);  
	DECLARE_VARIABLE(leadEmPt   ,float);  
	DECLARE_VARIABLE(leadEmEta  ,float);  
	DECLARE_VARIABLE(leadEmPhi  ,float);  
	DECLARE_VARIABLE(leadChPt   ,float);  
	DECLARE_VARIABLE(leadChEta  ,float);  
	DECLARE_VARIABLE(leadChPhi  ,float);  

	DECLARE_VARIABLE(dRLeadCent ,float);  
	DECLARE_VARIABLE(dRLead2nd  ,float);  
	DECLARE_VARIABLE(dRMeanNeut ,float);  
	DECLARE_VARIABLE(dRMeanEm   ,float);  
	DECLARE_VARIABLE(dRMeanCh   ,float);  

	DECLARE_VARIABLE(ptMean     ,float);
	DECLARE_VARIABLE(ptRMS      ,float);
	DECLARE_VARIABLE(pt2A       ,float);
	DECLARE_VARIABLE(ptDCh      ,float);
	DECLARE_VARIABLE(ptDNe      ,float);
	DECLARE_VARIABLE(sumPt      ,float);
	DECLARE_VARIABLE(sumChPt    ,float);
	DECLARE_VARIABLE(sumNePt    ,float);

	DECLARE_VARIABLE(leadFrac    ,float);  
	DECLARE_VARIABLE(secondFrac  ,float);  
	DECLARE_VARIABLE(thirdFrac   ,float);  
	DECLARE_VARIABLE(fourthFrac  ,float);  

	DECLARE_VARIABLE(leadChFrac    ,float);  
	DECLARE_VARIABLE(secondChFrac  ,float);  
	DECLARE_VARIABLE(thirdChFrac   ,float);  
	DECLARE_VARIABLE(fourthChFrac  ,float);  

	DECLARE_VARIABLE(leadNeutFrac    ,float);  
	DECLARE_VARIABLE(secondNeutFrac  ,float);  
	DECLARE_VARIABLE(thirdNeutFrac   ,float);  
	DECLARE_VARIABLE(fourthNeutFrac  ,float);  

	DECLARE_VARIABLE(leadEmFrac    ,float);  
	DECLARE_VARIABLE(secondEmFrac  ,float);  
	DECLARE_VARIABLE(thirdEmFrac   ,float);  
	DECLARE_VARIABLE(fourthEmFrac  ,float);  

	DECLARE_VARIABLE(jetW  ,float);  
	DECLARE_VARIABLE(etaW  ,float);  
	DECLARE_VARIABLE(phiW  ,float);  

	DECLARE_VARIABLE(majW  ,float);  
	DECLARE_VARIABLE(minW  ,float);  

	DECLARE_VARIABLE(chFrac01    ,float);  
	DECLARE_VARIABLE(chFrac02    ,float);  
	DECLARE_VARIABLE(chFrac03    ,float);  
	DECLARE_VARIABLE(chFrac04    ,float);  
	DECLARE_VARIABLE(chFrac05   ,float);  
	DECLARE_VARIABLE(chFrac06   ,float);  
	DECLARE_VARIABLE(chFrac07   ,float);  

	DECLARE_VARIABLE(neutFrac01    ,float);  
	DECLARE_VARIABLE(neutFrac02    ,float);  
	DECLARE_VARIABLE(neutFrac03    ,float);  
	DECLARE_VARIABLE(neutFrac04    ,float);  
	DECLARE_VARIABLE(neutFrac05   ,float);  
	DECLARE_VARIABLE(neutFrac06   ,float);  
	DECLARE_VARIABLE(neutFrac07   ,float);  

	DECLARE_VARIABLE(emFrac01    ,float);  
	DECLARE_VARIABLE(emFrac02    ,float);  
	DECLARE_VARIABLE(emFrac03    ,float);  
	DECLARE_VARIABLE(emFrac04    ,float);  
	DECLARE_VARIABLE(emFrac05   ,float);  
	DECLARE_VARIABLE(emFrac06   ,float);  
	DECLARE_VARIABLE(emFrac07   ,float);  

};

#undef DECLARE_VARIABLE

#endif
