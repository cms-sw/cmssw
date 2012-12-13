//--------------------------------------------------------------------------------------------------
// $Id $
//
// PileupJetIdentifier
//
// Author: P. Musella, P. Harris
//--------------------------------------------------------------------------------------------------

#include <string>
#include <map>

#ifndef RecoJets_JetProducers_interface_PileupJetIdentifier_h
#define RecoJets_JetProducers_interface_PileupJetIdentifier_h


// ----------------------------------------------------------------------------------------------------
class StoredPileupJetIdentifier {
public:
	friend class PileupJetIdAlgo;

	StoredPileupJetIdentifier();
	~StoredPileupJetIdentifier(); 

private: 
	float jetEta_; 
	float jetPt_; 
	float nCharged_; 
	float nNeutrals_; 
	float dZ_; 
	float nParticles_; 
	float dR2Mean_; 
	float dRMean_; 
	float frac01_; 
	float frac02_; 
	float frac03_; 
	float frac04_; 
	float frac05_; 
	float frac06_; 
	float frac07_; 
	float beta_; 
	float betaStar_; 
	float betaClassic_; 
	float betaStarClassic_; 
	float ptD_; 
	float nvtx_; 

public: 
	const float & jetEta() const { return jetEta_; } 
	void jetEta(const float val) { jetEta_ = val; };

	const float & jetPt() const { return jetPt_; } 
	void jetPt(const float val) { jetPt_ = val; };

	const float & nCharged() const { return nCharged_; } 
	void nCharged(const float val) { nCharged_ = val; };

	const float & nNeutrals() const { return nNeutrals_; } 
	void nNeutrals(const float val) { nNeutrals_ = val; };

	const float & dZ() const { return dZ_; } 
	void dZ(const float val) { dZ_ = val; };

	const float & nParticles() const { return nParticles_; } 
	void nParticles(const float val) { nParticles_ = val; };

	const float & dR2Mean() const { return dR2Mean_; } 
	void dR2Mean(const float val) { dR2Mean_ = val; };

	void RMS(const float val) { dR2Mean(val); }
	const float & RMS() const { return dR2Mean(); }

	const float & dRMean() const { return dRMean_; } 
	void dRMean(const float val) { dRMean_ = val; };

	const float & frac01() const { return frac01_; } 
	void frac01(const float val) { frac01_ = val; };

	const float & frac02() const { return frac02_; } 
	void frac02(const float val) { frac02_ = val; };

	const float & frac03() const { return frac03_; } 
	void frac03(const float val) { frac03_ = val; };

	const float & frac04() const { return frac04_; } 
	void frac04(const float val) { frac04_ = val; };

	const float & frac05() const { return frac05_; } 
	void frac05(const float val) { frac05_ = val; };

	const float & frac06() const { return frac06_; } 
	void frac06(const float val) { frac06_ = val; };

	const float & frac07() const { return frac07_; } 
	void frac07(const float val) { frac07_ = val; };

	const float & beta() const { return beta_; } 
	void beta(const float val) { beta_ = val; };

	const float & betaStar() const { return betaStar_; } 
	void betaStar(const float val) { betaStar_ = val; };

	const float & betaClassic() const { return betaClassic_; } 
	void betaClassic(const float val) { betaClassic_ = val; };

	const float & betaStarClassic() const { return betaStarClassic_; } 
	void betaStarClassic(const float val) { betaStarClassic_ = val; };

	const float & ptD() const { return ptD_; } 
	void ptD(const float val) { ptD_ = val; };

	const float & nvtx() const { return nvtx_; } 
	void nvtx(const float val) { nvtx_ = val; };
};

	


// ----------------------------------------------------------------------------------------------------
class PileupJetIdentifier : public StoredPileupJetIdentifier {
public:
	friend class PileupJetIdAlgo;

	PileupJetIdentifier();
	~PileupJetIdentifier(); 

	PileupJetIdentifier & operator= (const StoredPileupJetIdentifier & lhs) { ((StoredPileupJetIdentifier &)(*this)) = lhs; return *this;}

	enum Id {
	  kTight  = 0,
	  kMedium = 1,
	  kLoose  = 2
	};	

private: 
	int idFlag_;        
	float mva_;         
	float jetPhi_;      
	float jetM_;        
	float chgEMfrac_;   
	float neuEMfrac_;   
	float chgHadrfrac_; 
	float neuHadrfrac_; 
	float d0_;          
	float leadPt_;      
	float leadEta_;     
	float leadPhi_;     
	float secondPt_;    
	float secondEta_;   
	float secondPhi_;   
	float leadNeutPt_;  
	float leadNeutEta_; 
	float leadNeutPhi_; 
	float leadEmPt_;    
	float leadEmEta_;   
	float leadEmPhi_;   
	float leadChPt_;    
	float leadChEta_;   
	float leadChPhi_;   
	float dRLeadCent_;  
	float dRLead2nd_;   
	float dRMeanNeut_;  
	float dRMeanEm_;    
	float dRMeanCh_;    
	float ptMean_;      
	float ptRMS_;       
	float pt2A_;        
	float ptDCh_;       
	float ptDNe_;       
	float sumPt_;       
	float sumChPt_;     
	float sumNePt_;     
	float leadFrac_;    
	float secondFrac_;  
	float thirdFrac_;   
	float fourthFrac_;  
	float leadChFrac_;  
	float secondChFrac_;  
	float thirdChFrac_;   
	float fourthChFrac_;  
	float leadNeutFrac_;  
	float secondNeutFrac_;
	float thirdNeutFrac_; 
	float fourthNeutFrac_;
	float leadEmFrac_;    
	float secondEmFrac_;  
	float thirdEmFrac_;   
	float fourthEmFrac_;  
	float jetW_;          
	float etaW_;          
	float phiW_;          
	float majW_;          
	float minW_;          
	float chFrac01_;      
	float chFrac02_;      
	float chFrac03_;      
	float chFrac04_;      
	float chFrac05_;      
	float chFrac06_;      
	float chFrac07_;      
	float neutFrac01_;    
	float neutFrac02_;    
	float neutFrac03_;    
	float neutFrac04_;    
	float neutFrac05_;    
	float neutFrac06_;    
	float neutFrac07_;    
	float emFrac01_;      
	float emFrac02_;      
	float emFrac03_;      
	float emFrac04_;      
	float emFrac05_;      
	float emFrac06_;      
	float emFrac07_;     
	

public: 
	static bool passJetId(int flag, Id level) { return ( flag & (1 << level) ) != 0 ; }
	bool passJetId   (Id level) { return passJetId(idFlag_,level); }

	const int & idFlag() const { return idFlag_; } 
	void idFlag(const int val) { idFlag_ = val; };			   

	const float & mva() const { return mva_; } 
	void mva(const float val) { mva_ = val; };				   

	const float & jetPhi() const { return jetPhi_; } 
	void jetPhi(const float val) { jetPhi_ = val; };			   

	const float & jetM() const { return jetM_; } 
	void jetM(const float val) { jetM_ = val; };				   

	const float & chgEMfrac() const { return chgEMfrac_; } 
	void chgEMfrac(const float val) { chgEMfrac_ = val; };	   

	const float & neuEMfrac() const { return neuEMfrac_; } 
	void neuEMfrac(const float val) { neuEMfrac_ = val; };	   

	const float & chgHadrfrac() const { return chgHadrfrac_; } 
	void chgHadrfrac(const float val) { chgHadrfrac_ = val; };  

	const float & neuHadrfrac() const { return neuHadrfrac_; } 
	void neuHadrfrac(const float val) { neuHadrfrac_ = val; };  

	const float & d0() const { return d0_; } 
	void d0(const float val) { d0_ = val; };					   

	const float & leadPt() const { return leadPt_; } 
	void leadPt(const float val) { leadPt_ = val; };			   

	const float & leadEta() const { return leadEta_; } 
	void leadEta(const float val) { leadEta_ = val; };		   

	const float & leadPhi() const { return leadPhi_; } 
	void leadPhi(const float val) { leadPhi_ = val; };		   

	const float & secondPt() const { return secondPt_; } 
	void secondPt(const float val) { secondPt_ = val; };		   

	const float & secondEta() const { return secondEta_; } 
	void secondEta(const float val) { secondEta_ = val; };	   

	const float & secondPhi() const { return secondPhi_; } 
	void secondPhi(const float val) { secondPhi_ = val; };	   

	const float & leadNeutPt() const { return leadNeutPt_; } 
	void leadNeutPt(const float val) { leadNeutPt_ = val; };	   

	const float & leadNeutEta() const { return leadNeutEta_; } 
	void leadNeutEta(const float val) { leadNeutEta_ = val; };  

	const float & leadNeutPhi() const { return leadNeutPhi_; } 
	void leadNeutPhi(const float val) { leadNeutPhi_ = val; };  

	const float & leadEmPt() const { return leadEmPt_; } 
	void leadEmPt(const float val) { leadEmPt_ = val; };		   

	const float & leadEmEta() const { return leadEmEta_; } 
	void leadEmEta(const float val) { leadEmEta_ = val; };	   

	const float & leadEmPhi() const { return leadEmPhi_; } 
	void leadEmPhi(const float val) { leadEmPhi_ = val; };	   

	const float & leadChPt() const { return leadChPt_; } 
	void leadChPt(const float val) { leadChPt_ = val; };		   

	const float & leadChEta() const { return leadChEta_; } 
	void leadChEta(const float val) { leadChEta_ = val; };	   

	const float & leadChPhi() const { return leadChPhi_; } 
	void leadChPhi(const float val) { leadChPhi_ = val; };	   

	const float & dRLeadCent() const { return dRLeadCent_; } 
	void dRLeadCent(const float val) { dRLeadCent_ = val; };	   

	const float & dRLead2nd() const { return dRLead2nd_; } 
	void dRLead2nd(const float val) { dRLead2nd_ = val; };	   

	const float & dRMeanNeut() const { return dRMeanNeut_; } 
	void dRMeanNeut(const float val) { dRMeanNeut_ = val; };	   

	const float & dRMeanEm() const { return dRMeanEm_; } 
	void dRMeanEm(const float val) { dRMeanEm_ = val; };		   

	const float & dRMeanCh() const { return dRMeanCh_; } 
	void dRMeanCh(const float val) { dRMeanCh_ = val; };		   

	const float & ptMean() const { return ptMean_; } 
	void ptMean(const float val) { ptMean_ = val; };			   

	const float & ptRMS() const { return ptRMS_; } 
	void ptRMS(const float val) { ptRMS_ = val; };			   

	const float & pt2A() const { return pt2A_; } 
	void pt2A(const float val) { pt2A_ = val; };				   

	const float & ptDCh() const { return ptDCh_; } 
	void ptDCh(const float val) { ptDCh_ = val; };			   

	const float & ptDNe() const { return ptDNe_; } 
	void ptDNe(const float val) { ptDNe_ = val; };			   

	const float & sumPt() const { return sumPt_; } 
	void sumPt(const float val) { sumPt_ = val; };			   

	const float & sumChPt() const { return sumChPt_; } 
	void sumChPt(const float val) { sumChPt_ = val; };		   

	const float & sumNePt() const { return sumNePt_; } 
	void sumNePt(const float val) { sumNePt_ = val; };		   

	const float & leadFrac() const { return leadFrac_; } 
	void leadFrac(const float val) { leadFrac_ = val; };		   

	const float & secondFrac() const { return secondFrac_; } 
	void secondFrac(const float val) { secondFrac_ = val; };	   

	const float & thirdFrac() const { return thirdFrac_; } 
	void thirdFrac(const float val) { thirdFrac_ = val; };	   

	const float & fourthFrac() const { return fourthFrac_; } 
	void fourthFrac(const float val) { fourthFrac_ = val; };	

	const float & leadChFrac() const { return leadChFrac_; } 
	void leadChFrac(const float val) { leadChFrac_ = val; };      


	const float & secondChFrac() const { return secondChFrac_; } 
	void secondChFrac(const float val) { secondChFrac_ = val; };	   

	const float & thirdChFrac() const { return thirdChFrac_; } 
	void thirdChFrac(const float val) { thirdChFrac_ = val; };
	   
	const float & fourthChFrac() const { return fourthChFrac_; } 
	void fourthChFrac(const float val) { fourthChFrac_ = val; };	   

	const float & leadNeutFrac() const { return leadNeutFrac_; } 
	void leadNeutFrac(const float val) { leadNeutFrac_ = val; };	   

	const float & secondNeutFrac() const { return secondNeutFrac_; } 
	void secondNeutFrac(const float val) { secondNeutFrac_ = val; };

	const float & thirdNeutFrac() const { return thirdNeutFrac_; } 
	void thirdNeutFrac(const float val) { thirdNeutFrac_ = val; };

	const float & fourthNeutFrac() const { return fourthNeutFrac_; } 
	void fourthNeutFrac(const float val) { fourthNeutFrac_ = val; };

	const float & leadEmFrac() const { return leadEmFrac_; } 
	void leadEmFrac(const float val) { leadEmFrac_ = val; };		   

	const float & secondEmFrac() const { return secondEmFrac_; } 
	void secondEmFrac(const float val) { secondEmFrac_ = val; };	   

	const float & thirdEmFrac() const { return thirdEmFrac_; } 
	void thirdEmFrac(const float val) { thirdEmFrac_ = val; };	   

	const float & fourthEmFrac() const { return fourthEmFrac_; } 
	void fourthEmFrac(const float val) { fourthEmFrac_ = val; };	   

	const float & jetW() const { return jetW_; } 
	void jetW(const float val) { jetW_ = val; };					   

	const float & etaW() const { return etaW_; } 
	void etaW(const float val) { etaW_ = val; };					   

	const float & phiW() const { return phiW_; } 
	void phiW(const float val) { phiW_ = val; };					   

	const float & majW() const { return majW_; } 
	void majW(const float val) { majW_ = val; };					   

	const float & minW() const { return minW_; } 
	void minW(const float val) { minW_ = val; };					   

	const float & chFrac01() const { return chFrac01_; } 
	void chFrac01(const float val) { chFrac01_ = val; };			   

	const float & chFrac02() const { return chFrac02_; } 
	void chFrac02(const float val) { chFrac02_ = val; };			   

	const float & chFrac03() const { return chFrac03_; } 
	void chFrac03(const float val) { chFrac03_ = val; };			   

	const float & chFrac04() const { return chFrac04_; } 
	void chFrac04(const float val) { chFrac04_ = val; };			
   
	const float & chFrac05() const { return chFrac05_; } 
	void chFrac05(const float val) { chFrac05_ = val; };			

	const float & chFrac06() const { return chFrac06_; } 
	void chFrac06(const float val) { chFrac06_ = val; };			
   
	const float & chFrac07() const { return chFrac07_; } 
	void chFrac07(const float val) { chFrac07_ = val; };			  

	const float & neutFrac01() const { return neutFrac01_; } 
	void neutFrac01(const float val) { neutFrac01_ = val; };		   

	const float & neutFrac02() const { return neutFrac02_; } 
	void neutFrac02(const float val) { neutFrac02_ = val; };		   

	const float & neutFrac03() const { return neutFrac03_; } 
	void neutFrac03(const float val) { neutFrac03_ = val; };		   

	const float & neutFrac04() const { return neutFrac04_; } 
	void neutFrac04(const float val) { neutFrac04_ = val; };		   

	const float & neutFrac05() const { return neutFrac05_; } 
	void neutFrac05(const float val) { neutFrac05_ = val; };		   

	const float & neutFrac06() const { return neutFrac06_; } 
	void neutFrac06(const float val) { neutFrac06_ = val; };		   

	const float & neutFrac07() const { return neutFrac07_; } 
	void neutFrac07(const float val) { neutFrac07_ = val; };		   

	const float & emFrac01() const { return emFrac01_; } 
	void emFrac01(const float val) { emFrac01_ = val; };			   

	const float & emFrac02() const { return emFrac02_; } 
	void emFrac02(const float val) { emFrac02_ = val; };			   

	const float & emFrac03() const { return emFrac03_; } 
	void emFrac03(const float val) { emFrac03_ = val; };			   

	const float & emFrac04() const { return emFrac04_; } 
	void emFrac04(const float val) { emFrac04_ = val; };			   

	const float & emFrac05() const { return emFrac05_; } 
	void emFrac05(const float val) { emFrac05_ = val; };			   

	const float & emFrac06() const { return emFrac06_; } 
	void emFrac06(const float val) { emFrac06_ = val; };			   

	const float & emFrac07() const { return emFrac07_; } 
	void emFrac07(const float val) { emFrac07_ = val; };                    

};									  

#endif
