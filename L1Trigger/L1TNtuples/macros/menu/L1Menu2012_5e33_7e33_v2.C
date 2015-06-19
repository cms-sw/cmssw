#include "L1Ntuple.h"
#include "hist.C"
#include "Style.C"

#include "TLegend.h"
#include "TMath.h"
#include "TText.h"
#include "TH2.h"
#include "TAxis.h"
#include "TString.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <set>

// -- Huge prescale value for seeds "for lower PU"
#define INFTY 10000

TH1F *h_Cross,*h_Cross_8TeV;
TH1F *h_Jets,*h_Jets_8TeV;
TH1F *h_Sums,*h_Sums_8TeV;
TH1F *h_Egamma,*h_Egamma_8TeV;
TH1F *h_Muons,*h_Muons_8TeV;

TH1F *h_Block;
TH2F *cor_Block;

Int_t NPAGS = 6;
TH2F *cor_PAGS;
TH1F *h_PAGS_pure;
TH1F *h_PAGS_shared;

const Int_t N128 = 128;			// could be > 128 for "test seeds"
Int_t kOFFSET = 0;
Bool_t TheTriggerBits[N128];	// contains the emulated triggers for each event
TH1F *h_All,*h_All_8TeV;		// one bin for each trigger. Fill bin i if event fires trigger i.
TH1F *h_Pure,*h_Pure_8TeV;		// one bin for each trigger. Fill bin i if event fires trigger i and NO OTHER TRIGGER.

// Methods to scale L1 jets for new HCAL LUTs and estimate the rate changes 

// correction by 5% overall (from HCAL January 2012)
Double_t CorrectedL1JetPtByFactor(Double_t JetPt, Bool_t theL1JetCorrection=false) {

	Double_t JetPtcorr = JetPt;

	if (theL1JetCorrection) {
		JetPtcorr = JetPt*1.05;
	}
	return JetPtcorr;
}

// correction by 8% for forward jets (from HCAL January 2012)
Double_t CorrectedL1FwdJetPtByFactor(Bool_t isFwdJet, Double_t JetPt, Bool_t theL1JetCorrection=false) {

	Double_t JetPtcorr = JetPt;

	if (theL1JetCorrection) {
		if (isFwdJet) { JetPtcorr = JetPt*1.08; }
	}
	return JetPtcorr;
}

// correction for HF bins (from HCAL January 2012)
Size_t   JetHFiEtabins   = 13;
Int_t    JetHFiEtabin[]  = {29,30,31,32,33,34,35,36,37,38,39,40,41};
Double_t JetHFiEtacorr[] = {0.982,0.962,0.952, 0.943,0.947,0.939, 0.938,0.935,0.934, 0.935,0.942,0.923,0.914};

Double_t CorrectedL1JetPtByHFtowers(Double_t JetiEta,Double_t JetPt, Bool_t theL1JetCorrection=false) {

	Double_t JetPtcorr   = JetPt;

	if (theL1JetCorrection) {
		Int_t    iJetiEtabin = 0;
		for (iJetiEtabin=0; iJetiEtabin<JetHFiEtabins; iJetiEtabin++) {
			if (JetHFiEtabin[iJetiEtabin]==JetiEta) {
				JetPtcorr = JetPt * (1+(1-JetHFiEtacorr[iJetiEtabin]));
			}
		}
	}
	return JetPtcorr;
}

// correction for RCT->GCT bins (from HCAL January 2012)
// HF from 29-41, first 3 HF trigger towers 3 iEtas, last highest eta HF trigger tower 4 iEtas; each trigger tower is 0.5 eta, RCT iEta from 0->21 (left->right)
Double_t JetRCTHFiEtacorr[]  = {0.965,0.943,0.936,0.929}; // from HF iEta=29 to 41 (smaller->higher HF iEta)

Double_t CorrectedL1JetPtByGCTregions(Double_t JetiEta,Double_t JetPt, Bool_t theL1JetCorrection=false) {

	Double_t JetPtcorr   = JetPt;

	if (theL1JetCorrection) {

		if ((JetiEta>=7 && JetiEta<=14)) {
			JetPtcorr = JetPt * 1.05;
		}

		if ((JetiEta>=4 && JetiEta<=6) || (JetiEta>=15 && JetiEta<=17)) {
			JetPtcorr = JetPt * 0.95;
		}

		if (JetiEta==0 || JetiEta==21) {
			JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[3]));
		}
		else if (JetiEta==1 || JetiEta==20) {
			JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[2]));
		}
		else if (JetiEta==2 || JetiEta==19) {
			JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[1]));
		}
		else if (JetiEta==3 || JetiEta==18) {
			JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[0]));
		}
	}

	return JetPtcorr;
}

// methods for the correlation conditions

size_t PHIBINS = 18;
Double_t PHIBIN[] = {10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350};

size_t ETABINS = 23;
Double_t ETABIN[] = {-5.,-4.5,-4.,-3.5,
	-3.,-2.172,-1.74,-1.392,-1.044,-0.696,-0.348,
	0,
	0.348,0.696,1.044,1.392,1.74,2.172,3.,
	3.5,4.,4.5,5.};

size_t ETAMUBINS = 65;
Double_t ETAMU[] = { -2.45,-2.4,-2.35,-2.3,-2.25,-2.2,-2.15,-2.1,-2.05,-2,-1.95,-1.9,-1.85,-1.8,-1.75,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.75,1.8,1.85,1.9,1.95,2,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45 };

Int_t etaMuIdx(Double_t eta) {
	size_t etaIdx = 0.;
	for (size_t idx=0; idx<ETAMUBINS; idx++) {
		if (eta>=ETAMU[idx] and eta<ETAMU[idx+1])
			etaIdx = idx;
	}
	return int(etaIdx);
}

Int_t etaINjetCoord(Double_t eta){
	size_t etaIdx = 0.;
	for (size_t idx=0; idx<ETABINS; idx++) {
		if (eta>=ETABIN[idx] and eta<ETABIN[idx+1])
			etaIdx = idx;
	}
	return int(etaIdx);
}

Double_t degree(Double_t radian) {
	if (radian<0)
		return 360.+(radian/TMath::Pi()*180.);
	else
		return radian/TMath::Pi()*180.;
}

Int_t phiINjetCoord(Double_t phi) {
	size_t phiIdx = 0;
	Double_t phidegree = degree(phi);
	for (size_t idx=0; idx<PHIBINS; idx++) {
		if (phidegree>=PHIBIN[idx] and phidegree<PHIBIN[idx+1])
			phiIdx = idx;
		else if (phidegree>=PHIBIN[PHIBINS-1] || phidegree<=PHIBIN[0])
			phiIdx = idx;
	}
	phiIdx = phiIdx + 1;
	if (phiIdx == 18)  phiIdx = 0;
	return int(phiIdx);
}

Bool_t correlateInPhi(Int_t jetphi, Int_t muphi, Int_t delta=1) {

	Bool_t correlationINphi = fabs(muphi-jetphi)<fabs(2 +delta-1) || fabs(muphi-jetphi)>fabs(PHIBINS-2 - (delta-1) );
	return correlationINphi;

}

Bool_t correlateInEta(Int_t mueta, Int_t jeteta, Int_t delta=1) {
	Bool_t correlationINeta = fabs(mueta-jeteta)<2 + delta-1;
	return correlationINeta;
}

// set the errors properly
void CorrectScale(TH1F* h, Float_t scal) {

	Int_t nbins = h -> GetNbinsX();

	for (Int_t i=1; i<= nbins; i++)  {
		Float_t val = h -> GetBinContent(i);
		Float_t er = sqrt(val);
		val = val * scal;
		er = er * scal;
		h -> SetBinContent(i,val);
		h -> SetBinError(i,er);
	}
}

class L1Menu2012 : public L1Ntuple
{
	public :

	L1Menu2012(Float_t aTargetLumi, Float_t aNumberOfUserdLumiSections, Float_t aLumiForThisSetOfLumiSections, string aL1NtupleFileName,Float_t aAveragePU, Float_t aZeroBiasPrescale,Bool_t aL1JetCorrection) : 
	theTargetLumi(aTargetLumi), 
		theNumberOfUserdLumiSections(aNumberOfUserdLumiSections),
		theLumiForThisSetOfLumiSections(aLumiForThisSetOfLumiSections),
		theL1NtupleFileName(aL1NtupleFileName),
		theAveragePU(aAveragePU),
		theZeroBiasPrescale(aZeroBiasPrescale),
		theL1JetCorrection(aL1JetCorrection)
		{}

	~L1Menu2012() {}

	// The luminosity for which we want the rates, in units 1e32 (this sets also the right prescales for the 2 menus and some descriptions correctly).
	// 70. for 7e33, 50 for 5e33, etc. Use 70.001 for the "emergency columns" to the corresponding target luminosity.
	// For the moment we have pre-scales for 5e33,6e33,7e33 plus emergency pre-scales for 5e33 and 7e33
	Float_t theTargetLumi;

	// the setting below are/will be specific for each L1Ntuple file used
	Float_t theNumberOfUserdLumiSections;
	Float_t theLumiForThisSetOfLumiSections;
	string theL1NtupleFileName;
	Float_t theAveragePU;
	Float_t theZeroBiasPrescale;
	Bool_t theL1JetCorrection;

	ostringstream output;
	TString GetPrintout() { return output.str(); };

	void MyInit();
	void FilL1Bits();

	std::map<std::string, int> Counts;
	std::map<std::string, int> Prescales;
	std::map<std::string, bool> Biased;

    std::map<std::string, int> BitMapping;

	std::map<std::string, float> WeightsPAGs;


	void InsertInMenu(string L1name, Bool_t value);

	int L1Bit(string l1name);

	Bool_t Cross();
	Bool_t Jets();
	Bool_t EGamma();
	Bool_t Muons();
	Bool_t Sums();

// -- Cross
	Bool_t Mu_EG(Float_t mucut, Float_t EGcut );
	Bool_t MuOpen_EG(Float_t mucut, Float_t EGcut );
	Bool_t Mu_JetCentral(Float_t mucut, Float_t jetcut );
	Bool_t Mu_DoubleJetCentral(Float_t mucut, Float_t jetcut );
	Bool_t Mu_JetCentral_LowerTauTh(Float_t mucut, Float_t jetcut, Float_t taucut );
	Bool_t Muer_JetCentral(Float_t mucut, Float_t jetcut );
	Bool_t Muer_JetCentral_LowerTauTh(Float_t mucut, Float_t jetcut, Float_t taucut );
	Bool_t Mu_HTT(Float_t mucut, Float_t HTcut );
	Bool_t Muer_ETM(Float_t mucut, Float_t ETMcut );
	Bool_t EG_FwdJet(Float_t EGcut, Float_t FWcut ) ;
	Bool_t EG_HT(Float_t EGcut, Float_t HTcut );
	Bool_t EG_DoubleJetCentral(Float_t EGcut, Float_t jetcut );
	Bool_t DoubleEG_HT(Float_t EGcut, Float_t HTcut );
	Bool_t EGEta2p1_JetCentral(Float_t EGcut, Float_t jetcut);		// delta
	Bool_t EGEta2p1_JetCentral_LowTauTh(Float_t EGcut, Float_t jetcut, Float_t taucut);          // delta
	Bool_t IsoEGEta2p1_JetCentral_LowTauTh(Float_t EGcut, Float_t jetcut, Float_t taucut);          // delta
	Bool_t EGEta2p1_DoubleJetCentral(Float_t EGcut, Float_t jetcut);	// delta
	Bool_t EGEta2p1_DoubleJetCentral_TripleJetCentral(Float_t EGcut, Float_t jetcut2, Float_t jetcut3);   

	Bool_t HTT_HTM(Float_t HTTcut, Float_t HTMcut);
	Bool_t JetCentral_ETM(Float_t jetcut, Float_t ETMcut);
	Bool_t DoubleJetCentral_ETM(Float_t jetcut1, Float_t jetcut2, Float_t ETMcut);
	Bool_t DoubleMu_EG(Float_t mucut, Float_t EGcut );
	Bool_t Mu_DoubleEG(Float_t mucut, Float_t EGcut);

	Bool_t Muer_TripleJetCentral(Float_t mucut, Float_t jet1, Float_t jet2, Float_t jet3);
	Bool_t Mia(Float_t mucut, Float_t jet1, Float_t jet2);	// delta
	Bool_t Mu_JetCentral_delta(Float_t mucut, Float_t ptcut);	// delta
	Bool_t Mu_JetCentral_deltaOut(Float_t mucut, Float_t ptcut); // delta


// -- Jets 
	Bool_t SingleJet(Float_t cut);
	Bool_t SingleJetCentral(Float_t cut);
	Bool_t DoubleJetCentral(Float_t cut1, Float_t cut2);
	Bool_t DoubleJet_Eta1p7_deltaEta4(Float_t cut1, Float_t cut2);
	Bool_t TripleJetCentral(Float_t cut1, Float_t cut2, Float_t cut3);
	Bool_t TripleJet_VBF(Float_t cut1, Float_t cut2, Float_t cut3);

	Bool_t QuadJetCentral(Float_t cut1, Float_t cut2, Float_t cut3, Float_t cut4);
	Bool_t DoubleTauJetEta2p17(Float_t cut1, Float_t cut2);

// -- Sums
	Bool_t ETT(Float_t ETTcut);
	Bool_t HTT(Float_t HTTcut);
	Bool_t ETM(Float_t ETMcut);

// -- Egamma
	Bool_t SingleEG(Float_t cut);
	Bool_t SingleEG_Eta2p1(Float_t cut);
	Bool_t SingleIsoEG_Eta2p1(Float_t cut);

	Bool_t DoubleEG(Float_t cut1, Float_t cut2);
	Bool_t TripleEG(Float_t cut1, Float_t cut2, Float_t cut3);

// -- Muons 
	Bool_t SingleMu(Float_t ptcut, Int_t qualmin=4);
	Bool_t SingleMuEta2p1(Float_t ptcut);
	Bool_t DoubleMu(Float_t cut1, Float_t cut2);	// on top of DoubleMu3
	Bool_t DoubleMuHighQEtaCut(Float_t ptcut, Float_t etacut);
	Bool_t TripleMu(Float_t cut1, Float_t cut2, Float_t cut3, Int_t qualmin);	// on top of DoubleMu3
	Bool_t DoubleMuXOpen(Float_t ptcut);	// on top of SingleMu7
	Bool_t Onia(Float_t ptcut1, Float_t ptcut2, Float_t etacut, Int_t delta);   

	void Loop();

	private :

	Bool_t PhysicsBits[128];
	Bool_t first;

	Int_t insert_ibin;
	Bool_t insert_val[100];
	string insert_names[100];

	Int_t NBITS_MUONS;
	Int_t NBITS_EGAMMA;
	Int_t NBITS_JETS;
	Int_t NBITS_SUMS;
	Int_t NBITS_CROSS;

	set<string> setTOP;
	set<string> setHIGGS;
	set<string> setEXO;
	set<string> setSMP;
	set<string> setBPH;
	set<string> setSUSY;

};

void L1Menu2012::InsertInMenu(string L1name, Bool_t value) {

	Bool_t post_prescale = false;

	Int_t prescale = 1;

	map<string, int>::const_iterator it = Prescales.find(L1name);
	if (it == Prescales.end() ) {
		cout << " --- NO PRESCALE DEFINED FOR " << L1name << " ---  SET P = 1 " << endl;
	}
	else {
		prescale = Prescales[L1name];
	}

	if (prescale >0) {
		Counts[L1name] ++;
		Int_t n = Counts[L1name];
		if ( n % prescale == 0) post_prescale = value; 
	}

	insert_names[insert_ibin] = L1name;
	insert_val[insert_ibin] = post_prescale ;

	insert_ibin ++;

}

int L1Menu2012::L1Bit(string l1name) {

	map<string, int>::const_iterator it = BitMapping.find(l1name);
	if (it == BitMapping.end() ) {
		cout << " Wrong L1 name, not in BitMapping " << l1name << endl;
		return -1;
	}

	return BitMapping[l1name];
}

void L1Menu2012::FilL1Bits() {
	for (Int_t ibit=0; ibit < 128; ibit++) {
		PhysicsBits[ibit] = 0;
		if (ibit<64) {
			PhysicsBits[ibit] = (gt_->tw1[2]>>ibit)&1;
		}
		else {
			PhysicsBits[ibit] = (gt_->tw2[2]>>(ibit-64))&1;
		}
	}
}       

void L1Menu2012::MyInit() {

// --- The seeds per group
	setTOP.insert("L1_HTT150") ;
	setTOP.insert("L1_HTT175") ;
	setTOP.insert("L1_HTT200") ;
	setTOP.insert("L1_SingleEG18er") ;
	setTOP.insert("L1_SingleIsoEG18er") ;
	setTOP.insert("L1_SingleEG20") ;
	setTOP.insert("L1_SingleIsoEG20er") ;
	setTOP.insert("L1_SingleEG22") ;
	setTOP.insert("L1_SingleEG24") ;
	setTOP.insert("L1_SingleEG30") ;
	setTOP.insert("L1_DoubleEG_13_7") ;
	setTOP.insert("L1_QuadJetC36") ;
	setTOP.insert("L1_QuadJetC40") ;
	setTOP.insert("L1_Mu12_EG7") ;
	setTOP.insert("L1_Mu3p5_EG12") ;
	setTOP.insert("L1_SingleMu14er") ;
	setTOP.insert("L1_SingleMu16er") ;
	setTOP.insert("L1_SingleMu18er") ;
	setTOP.insert("L1_SingleMu20er") ;
	setTOP.insert("L1_SingleMu25er") ;
	setTOP.insert("L1_DoubleMu_12_5") ;
	setTOP.insert("L1_DoubleMu_10_3p5") ;


	setHIGGS.insert("L1_ETM30") ;
	setHIGGS.insert("L1_ETM36") ;
	setHIGGS.insert("L1_ETM40") ;
	setHIGGS.insert("L1_SingleEG7") ;
	setHIGGS.insert("L1_SingleEG12") ;
	setHIGGS.insert("L1_SingleEG18er") ;
	setHIGGS.insert("L1_SingleIsoEG18er") ;
	setHIGGS.insert("L1_SingleEG20") ;
	setHIGGS.insert("L1_SingleIsoEG20er") ;
	setHIGGS.insert("L1_SingleEG22") ;
	setHIGGS.insert("L1_DoubleEG_13_7") ;
	setHIGGS.insert("L1_TripleEG_12_7_5") ;
	setHIGGS.insert("L1_SingleJet128") ;
	setHIGGS.insert("L1_DoubleJetC36") ;
	setHIGGS.insert("L1_DoubleJetC44_Eta1p74_WdEta4") ;
	setHIGGS.insert("L1_DoubleJetC52") ;
	setHIGGS.insert("L1_DoubleJetC56_Eta1p74_WdEta4") ;
	setHIGGS.insert("L1_DoubleJetC56") ;
	setHIGGS.insert("L1_DoubleJetC64") ;
	setHIGGS.insert("L1_TripleJet_64_44_24_VBF") ;
	setHIGGS.insert("L1_TripleJet_64_48_28_VBF") ;
	setHIGGS.insert("L1_TripleJet_68_48_32_VBF") ;
	setHIGGS.insert("L1_DoubleTauJet44er") ;
	setHIGGS.insert("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12") ;
	setHIGGS.insert("L1_Mu10er_JetC32") ;
	setHIGGS.insert("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12") ;
	setHIGGS.insert("L1_EG18er_JetC_Cen28_Tau20_dPhi1") ;
	setHIGGS.insert("L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1") ;
	setHIGGS.insert("L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1") ;
	setHIGGS.insert("L1_EG18er_JetC_Cen36_Tau28_dPhi1") ;
	setHIGGS.insert("L1_Mu12_EG7") ;
	setHIGGS.insert("L1_MuOpen_EG12") ;
	setHIGGS.insert("L1_Mu3p5_EG12") ;
	setHIGGS.insert("L1_SingleMu3") ;
	setHIGGS.insert("L1_SingleMu7") ;
	setHIGGS.insert("L1_SingleMu18er") ;
	setHIGGS.insert("L1_SingleMu20er") ;
	setHIGGS.insert("L1_DoubleMu_10_Open") ;
	setHIGGS.insert("L1_DoubleMu_10_3p5") ;

	setSUSY.insert("L1_HTT100") ;
	setSUSY.insert("L1_HTT125") ;
	setSUSY.insert("L1_HTT150") ;
	setSUSY.insert("L1_HTT175") ;
	setSUSY.insert("L1_HTT200") ;
	setSUSY.insert("L1_SingleEG20") ;
	setSUSY.insert("L1_SingleIsoEG20er") ;
	setSUSY.insert("L1_SingleEG22") ;
	setSUSY.insert("L1_SingleEG24") ;
	setSUSY.insert("L1_DoubleEG_13_7") ;
	setSUSY.insert("L1_TripleEG7") ;
	setSUSY.insert("L1_SingleJet128") ;
	setSUSY.insert("L1_DoubleJetC52") ;
	setSUSY.insert("L1_DoubleJetC56") ;
	setSUSY.insert("L1_DoubleJetC64") ;
	setSUSY.insert("L1_QuadJetC32") ;
	setSUSY.insert("L1_QuadJetC36") ;
	setSUSY.insert("L1_QuadJetC40") ;
	setSUSY.insert("L1_Mu0_HTT50") ;
	setSUSY.insert("L1_Mu0_HTT100") ;
	setSUSY.insert("L1_Mu4_HTT125") ;
	setSUSY.insert("L1_Mu8_DoubleJetC20") ;
	setSUSY.insert("L1_DoubleEG6_HTT100") ;
	setSUSY.insert("L1_DoubleEG6_HTT125") ;
	setSUSY.insert("L1_EG8_DoubleJetC20") ;
	setSUSY.insert("L1_Mu12_EG7") ;
	setSUSY.insert("L1_MuOpen_EG12") ;
	setSUSY.insert("L1_Mu3p5_EG12") ;
	setSUSY.insert("L1_DoubleMu3p5_EG5") ;
	setSUSY.insert("L1_DoubleMu5_EG5") ;
	setSUSY.insert("L1_Mu5_DoubleEG5") ;
	setSUSY.insert("L1_Mu5_DoubleEG6") ;
	setSUSY.insert("L1_DoubleJetC36_ETM30") ;
	setSUSY.insert("L1_DoubleJetC44_ETM30") ;
	setSUSY.insert("L1_SingleMu14er") ;
	setSUSY.insert("L1_SingleMu16er") ;
	setSUSY.insert("L1_SingleMu18er") ;
	setSUSY.insert("L1_SingleMu20er") ;
	setSUSY.insert("L1_TripleMu0") ;
	setSUSY.insert("L1_TripleMu0_HighQ") ;

	setEXO.insert("L1_ETM36") ;
	setEXO.insert("L1_ETM40") ;
	setEXO.insert("L1_ETM50") ;
	setEXO.insert("L1_ETM70") ;
	setEXO.insert("L1_ETM100") ;
	setEXO.insert("L1_HTT150") ;
	setEXO.insert("L1_HTT175") ;
	setEXO.insert("L1_HTT200") ;
	setEXO.insert("L1_ETT300") ;
	setEXO.insert("L1_SingleEG20") ;
	setEXO.insert("L1_SingleIsoEG20er") ;
	setEXO.insert("L1_SingleEG22") ;
	setEXO.insert("L1_SingleEG24") ;
	setEXO.insert("L1_SingleEG30") ;
	setEXO.insert("L1_DoubleEG_13_7") ;
	setEXO.insert("L1_SingleJet36") ;
	setEXO.insert("L1_SingleJet52") ;
	setEXO.insert("L1_SingleJet68") ;
	setEXO.insert("L1_SingleJet92") ;
	setEXO.insert("L1_SingleJet128") ;
	setEXO.insert("L1_QuadJetC36") ;
	setEXO.insert("L1_QuadJetC40") ;
	setEXO.insert("L1_Mu12_EG7") ;
	setEXO.insert("L1_Mu3p5_EG12") ;
	setEXO.insert("L1_SingleMu16er") ;
	setEXO.insert("L1_SingleMu18er") ;
	setEXO.insert("L1_SingleMu20er") ;
	setEXO.insert("L1_SingleMu25er") ;
	setEXO.insert("L1_DoubleMu0er_HighQ") ;
	setEXO.insert("L1_DoubleMu_10_3p5") ;
	setEXO.insert("L1_TripleMu0_HighQ") ;
	setEXO.insert("L1_SingleJetC32_NotBptxOR") ;
	setEXO.insert("L1_SingleJetC20_NotBptxOR") ;
	setEXO.insert("L1_SingleMu6_NotBptxOR") ;
	setEXO.insert("L1_DoubleMu3er_HighQ_WdEta22");

	setSMP.insert("L1_SingleEG7") ;
	setSMP.insert("L1_SingleEG12") ;
	setSMP.insert("L1_SingleIsoEG18er") ;
	setSMP.insert("L1_SingleEG20") ;
	setSMP.insert("L1_SingleIsoEG20er") ;
	setSMP.insert("L1_SingleEG22") ;
	setSMP.insert("L1_SingleEG24") ;
	setSMP.insert("L1_SingleEG30") ;
	setSMP.insert("L1_DoubleEG_13_7") ;
	setSMP.insert("L1_SingleJet16") ;
	setSMP.insert("L1_SingleJet36") ;
	setSMP.insert("L1_SingleJet52") ;
	setSMP.insert("L1_SingleJet68") ;
	setSMP.insert("L1_SingleJet92") ;
	setSMP.insert("L1_SingleJet128") ;
	setSMP.insert("L1_EG22_ForJet24") ;
	setSMP.insert("L1_EG22_ForJet32") ;
	setSMP.insert("L1_Mu12_EG7") ;
	setSMP.insert("L1_MuOpen_EG12") ;
	setSMP.insert("L1_Mu3p5_EG12") ;
	setSMP.insert("L1_SingleMu7") ;
	setSMP.insert("L1_SingleMu16er") ;
	setSMP.insert("L1_SingleMu18er") ;
	setSMP.insert("L1_SingleMu20er") ;
	setSMP.insert("L1_DoubleMu_12_5") ;
	setSMP.insert("L1_DoubleMu_10_Open") ;
	setSMP.insert("L1_DoubleMu_10_3p5") ;

	setBPH.insert("L1_SingleMu3") ;
	setBPH.insert("L1_DoubleMu0er_HighQ") ;
	setBPH.insert("L1_DoubleMu3er_HighQ_WdEta22") ;
	setBPH.insert("L1_DoubleMu_5er_0er_HighQ_WdEta22") ;
	setBPH.insert("L1_TripleMu0_HighQ") ;

// ---- The bit mapping
	//  be carefull with Trigger name != Trigger alias for a few bits that were already in 2011 menu :
	// SingleMu16_Eta2p1
	// SingleMu14_Eta2p1
	// DoubleJet52_Central
	// L1_DoubleTauJet44_Eta2p17
	// L1_DoubleMu0_HighQ_EtaCuts 
	// L1_DoubleMu5_v1
	// L1_DoubleJet36_Central
	// L1_DoubleJet64_Central
	
	BitMapping["L1_ZeroBias"] = 0 ;
	BitMapping["L1_ZeroBias_Instance1"] = 1 ;
	BitMapping["L1_BeamGas_Hf_BptxPlusPostQuiet"] = 2 ;
	BitMapping["L1_BeamGas_Hf_BptxMinusPostQuiet"] = 4 ;
	BitMapping["L1_InterBunch_Bptx"] = 5 ;
	BitMapping["L1_BeamHalo"] = 8 ;
	BitMapping["L1_TripleMu0"] = 9 ;
	BitMapping["L1_Mu4_HTT125"] = 10 ;
	BitMapping["L1_Mu3p5_EG12"] = 11 ;
	BitMapping["L1_Mu12er_ETM20"] = 12 ;
	BitMapping["L1_MuOpen_EG12"] = 13 ;
	BitMapping["L1_Mu12_EG7"] = 14 ;
	BitMapping["L1_SingleJet16"] = 15 ;
	BitMapping["L1_SingleJet36"] = 16 ;
	BitMapping["L1_SingleJet52"] = 17 ;
	BitMapping["L1_SingleJet68"] = 18 ;
	BitMapping["L1_SingleJet92"] = 19 ;
	BitMapping["L1_SingleJet128"] = 20 ;
	BitMapping["L1_DoubleEG6_HTT100"] = 21 ;
	BitMapping["L1_DoubleEG6_HTT125"] = 22 ;
	BitMapping["L1_Mu5_DoubleEG5"] = 23 ;
	BitMapping["L1_DoubleMu3p5_EG5"] = 24 ;
	BitMapping["L1_DoubleMu5_EG5"] = 25 ;
	BitMapping["L1_DoubleMu0er_HighQ"] = 26 ;
	BitMapping["L1_Mu5_DoubleEG6"] = 27 ;
	BitMapping["L1_DoubleJetC44_ETM30"] = 28 ;
	BitMapping["L1_Mu3_JetC16_WdEtaPhi2"] = 29 ;
	BitMapping["L1_Mu3_JetC52_WdEtaPhi2"] = 30 ;
	BitMapping["L1_SingleEG7"] = 31 ;
	BitMapping["L1_SingleIsoEG20er"] = 32 ;
	BitMapping["L1_EG22_ForJet24"] = 33 ;
	BitMapping["L1_EG22_ForJet32"] = 34 ;
	BitMapping["L1_DoubleJetC44_Eta1p74_WdEta4"] = 35 ;
	BitMapping["L1_DoubleJetC56_Eta1p74_WdEta4"] = 36 ;
	BitMapping["L1_DoubleTauJet44er"] = 37 ;
	BitMapping["L1_DoubleEG_13_7"] = 38 ;
	BitMapping["L1_TripleEG_12_7_5"] = 39 ;
	BitMapping["L1_HTT125"] = 40 ;
	BitMapping["L1_DoubleJetC52"] = 41 ;
	BitMapping["L1_SingleMu14er"] = 42 ;
	BitMapping["L1_SingleIsoEG18er"] = 43 ;
	BitMapping["L1_DoubleMu_10_Open"] = 44 ;
	BitMapping["L1_DoubleMu_10_3p5"] = 45 ;
	BitMapping["L1_ETT80"] = 46 ;
	BitMapping["L1_SingleEG5"] = 47 ;
	BitMapping["L1_SingleEG18er"] = 48 ;
	BitMapping["L1_SingleEG22"] = 49 ;
	BitMapping["L1_SingleEG12"] = 50 ;
	BitMapping["L1_SingleEG24"] = 51 ;
	BitMapping["L1_SingleEG20"] = 52 ;
	BitMapping["L1_SingleEG30"] = 53 ;
	BitMapping["L1_DoubleMu3er_HighQ_WdEta22"] = 54 ;
	BitMapping["L1_SingleMuOpen"] = 55 ;
	BitMapping["L1_SingleMu16"] = 56 ;
	BitMapping["L1_SingleMu3"] = 57 ;
	BitMapping["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 58 ;
	BitMapping["L1_SingleMu7"] = 59 ;
	BitMapping["L1_SingleMu20er"] = 60 ;
	BitMapping["L1_SingleMu12"] = 61 ;
	BitMapping["L1_SingleMu20"] = 62 ;
	BitMapping["L1_SingleMu25er"] = 63 ;
	BitMapping["L1_ETM100"] = 64 ;
	BitMapping["L1_ETM36"] = 65 ;
	BitMapping["L1_ETM30"] = 66 ;
	BitMapping["L1_ETM50"] = 67 ;
	BitMapping["L1_ETM70"] = 68 ;
	BitMapping["L1_ETT300"] = 69 ;
	BitMapping["L1_HTT100"] = 70 ;
	BitMapping["L1_HTT150"] = 71 ;
	BitMapping["L1_HTT175"] = 72 ;
	BitMapping["L1_HTT200"] = 73 ;
	BitMapping["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 74 ;
	BitMapping["L1_Mu10er_JetC32"] = 75 ;
	BitMapping["L1_DoubleJetC64"] = 76 ;
	BitMapping["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 77 ;
	BitMapping["L1_SingleJetC32_NotBptxOR"] = 78 ;
	BitMapping["L1_ETM40"] = 79 ;
	BitMapping["L1_Mu0_HTT50"] = 80 ;
	BitMapping["L1_Mu0_HTT100"] = 81 ;
	BitMapping["L1_DoubleEG5"] = 82 ;
	BitMapping["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 83 ;
	BitMapping["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 84 ;
	BitMapping["L1_SingleMu16er"] = 86 ;
	BitMapping["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 87 ;
	BitMapping["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 88 ;
	BitMapping["L1_SingleMu6_NotBptxOR"] = 89 ;
	BitMapping["L1_Mu8_DoubleJetC20"] = 90 ;
	BitMapping["L1_DoubleMu0"] = 92 ;
	BitMapping["L1_EG8_DoubleJetC20"] = 94 ;
	BitMapping["L1_DoubleMu5"] = 95 ;
	BitMapping["L1_DoubleJetC56"] = 96 ;
	BitMapping["L1_TripleMu0_HighQ"] = 97 ;
	BitMapping["L1_TripleMu_5_5_0"] = 98 ;
	BitMapping["L1_ETT140"] = 99 ;
	BitMapping["L1_DoubleJetC36"] = 100 ;
	BitMapping["L1_DoubleJetC36_ETM30"] = 101 ;
	BitMapping["L1_SingleJet36_FwdVeto5"] = 102 ;
	BitMapping["L1_TripleJet_64_44_24_VBF"] = 103 ;
	BitMapping["L1_TripleJet_64_48_28_VBF"] = 104 ;
	BitMapping["L1_TripleJet_68_48_32_VBF"] = 105 ;
	BitMapping["L1_QuadJetC40"] = 106 ;
	BitMapping["L1_QuadJetC36"] = 107 ;
	BitMapping["L1_TripleJetC_52_28_28"] = 108 ;
	BitMapping["L1_QuadJetC32"] = 109 ;
	BitMapping["L1_DoubleForJet16_EtaOpp"] = 110 ;
	BitMapping["L1_DoubleEG3_FwdVeto"] = 111 ;
	BitMapping["L1_SingleJet20_Central_NotBptxOR"] = 112 ;
	BitMapping["L1_SingleJet16_FwdVeto5"] = 113 ;
	BitMapping["L1_SingleForJet16"] = 114 ;
	BitMapping["L1_DoubleJetC36_RomanPotsOR"] = 115 ;
	BitMapping["L1_SingleMu20_RomanPotsOR"] = 116 ;
	BitMapping["L1_SingleEG20_RomanPotsOR"] = 117 ;
	BitMapping["L1_DoubleMu5_RomanPotsOR"] = 118 ;
	BitMapping["L1_DoubleEG5_RomanPotsOR"] = 119 ;
	BitMapping["L1_SingleJet52_RomanPotsOR"] = 120 ;
	BitMapping["L1_SingleMu18er"] = 122 ;
	BitMapping["L1_MuOpen_EG5"] = 123 ;
	BitMapping["L1_DoubleMu_12_5"] = 124 ;
	BitMapping["L1_TripleEG7"] = 125 ;

// target lumi = 1e32
	if (theTargetLumi == 1) {

// -- Cross 
		Prescales["L1_Mu0_HTT50"] = 1;
		Prescales["L1_Mu0_HTT100"] = 1;
		Prescales["L1_Mu4_HTT125"] = 1;

		Prescales["L1_Mu12er_ETM20"] = 1;

		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 1;
		Prescales["L1_Mu10er_JetC32"] = 1 ;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

		Prescales["L1_Mu3_JetC16_WdEtaPhi2"] = 1;  
		Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 1;
		Prescales["L1_Mu8_DoubleJetC20"] = 1;
// Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;

		Prescales["L1_EG22_ForJet24"] = 1;
		Prescales["L1_EG22_ForJet32"] = 1;
		Prescales["L1_DoubleEG6_HTT100"] = 1;
		Prescales["L1_DoubleEG6_HTT125"] = 1;

		Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 1;
// Prescales["L1_EG18er_CentralJet32_or_TauJet24_deltaPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
		Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 1;

		Prescales["L1_EG8_DoubleJetC20"] = 15 ;  

		Prescales["L1_Mu12_EG7"] = 1;

		Prescales["L1_DoubleMu3p5_EG5"] = 1;
		Prescales["L1_DoubleMu5_EG5"] = 1;

// Prescales["L1_Mu3p5_DoubleEG5"] = 0;
		Prescales["L1_Mu5_DoubleEG5"] = 1 ;
		Prescales["L1_Mu5_DoubleEG6"] = 1;

		Prescales["L1_MuOpen_EG12"] = 1;
		Prescales["L1_Mu3p5_EG12"] = 1;

		Prescales["L1_DoubleJetC36_ETM30"] = 1;
		Prescales["L1_DoubleJetC44_ETM30"] = 1;


// -- Jets 
		Prescales["L1_SingleJet16"] = 1111;
		Prescales["L1_SingleJet36"] = 11;
		Prescales["L1_SingleJet52"] = 1;
		Prescales["L1_SingleJet68"] = 1;
		Prescales["L1_SingleJet92"] = 1 ;
		Prescales["L1_SingleJet128"] =1;

		Prescales["L1_DoubleJetC36"] = 1;
		Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 1;
		Prescales["L1_DoubleJetC52"] = 1;
		Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
		Prescales["L1_DoubleJetC56"] = 1;
		Prescales["L1_DoubleJetC64"] = 1;

		Prescales["L1_TripleJet_64_44_24_VBF"] = 1;
		Prescales["L1_TripleJet_64_48_28_VBF"] = 1;
		Prescales["L1_TripleJet_68_48_32_VBF"] = 1;

// Prescales["L1_TripleJet28_Central"] = 500; 
		Prescales["L1_TripleJetC_52_28_28"] = 1;

		Prescales["L1_QuadJetC32"] = 1;            
		Prescales["L1_QuadJetC36"] = 1;
		Prescales["L1_QuadJetC40"] = 1;

		Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
		Prescales["L1_ETM30"] = 1;
		Prescales["L1_ETM36"] = 1;
		Prescales["L1_ETM40"] = 1;
		Prescales["L1_ETM50"] = 1;
		Prescales["L1_ETM70"] = 1;
		Prescales["L1_ETM100"] = 1;

		Prescales["L1_HTT100"] = 1;
		Prescales["L1_HTT125"] = 1;
		Prescales["L1_HTT150"] = 1;
		Prescales["L1_HTT175"] = 1;
		Prescales["L1_HTT200"] = 1;

		Prescales["L1_ETT300"] = 1;


// -- Egamma
		Prescales["L1_SingleEG5"] = 123;
		Prescales["L1_SingleEG7"] = 41;
		Prescales["L1_SingleEG12"] = 1;
		Prescales["L1_SingleEG18er"] = 1;
// Prescales["L1_SingleIsoEG18er"] = 10;
		Prescales["L1_SingleIsoEG18er"] = 1;

		Prescales["L1_SingleEG20"] = 1;
		Prescales["L1_SingleIsoEG20er"] = 0;
		Prescales["L1_SingleEG22"] = 1;
		Prescales["L1_SingleEG24"] = 1;
		Prescales["L1_SingleEG30"] = 1;

// Prescales["L1_DoubleEG_15_5"] = 1;
		Prescales["L1_DoubleEG_13_7"] = 1;

		Prescales["L1_TripleEG7"] = 1;
		Prescales["L1_TripleEG_12_7_5"] = 1;


// -- Muons 

		Prescales["L1_SingleMu12"] = 0;

		Prescales["L1_SingleMu20"] = 1;
		Prescales["L1_SingleMu16"] = 1;
		Prescales["L1_SingleMu12"] = 1;
		Prescales["L1_SingleMu7"] = 1;
		Prescales["L1_SingleMu3"] = 53;
		Prescales["L1_SingleMuOpen"] = 106;
		Prescales["L1_DoubleMu0"] = 1;

		Prescales["L1_SingleMu25er"] = 1;
		Prescales["L1_SingleMu20er"] = 1;
// Prescales["L1_SingleMu14er"] = 50;
		Prescales["L1_SingleMu14er"] =  1;
		Prescales["L1_SingleMu18er"] = 1;
		Prescales["L1_SingleMu16er"] = 1;
		Prescales["L1_DoubleMu_12_5"] = 1;
		Prescales["L1_DoubleMu5"] = 1;
		Prescales["L1_DoubleMu0er_HighQ"] = 1;
		Prescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_10_Open"] = 1 ;
		Prescales["L1_DoubleMu_10_3p5"] = 1;
		Prescales["L1_TripleMu0"] = 1;
		Prescales["L1_TripleMu0_HighQ"] = 1;
		Prescales["L1_TripleMu_5_5_0"] = 1;

	}

// target lumi = 2e32
	if (theTargetLumi == 2) {

// -- Cross 
		Prescales["L1_Mu0_HTT50"] = 1;
		Prescales["L1_Mu0_HTT100"] = 1;
		Prescales["L1_Mu4_HTT125"] = 1;

		Prescales["L1_Mu12er_ETM20"] = 1;

		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 1;
		Prescales["L1_Mu10er_JetC32"] = 1 ;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

		Prescales["L1_Mu3_JetC16_WdEtaPhi2"] = 1;  
		Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 1;
		Prescales["L1_Mu8_DoubleJetC20"] = 1;
// Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;

		Prescales["L1_EG22_ForJet24"] = 1;
		Prescales["L1_EG22_ForJet32"] = 1;
		Prescales["L1_DoubleEG6_HTT100"] = 1;
		Prescales["L1_DoubleEG6_HTT125"] = 1;

		Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 1;
// Prescales["L1_EG18er_CentralJet32_or_TauJet24_deltaPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
		Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 1;

		Prescales["L1_EG8_DoubleJetC20"] = 30 ;  

		Prescales["L1_Mu12_EG7"] = 1;

		Prescales["L1_DoubleMu3p5_EG5"] = 1;
		Prescales["L1_DoubleMu5_EG5"] = 1;

// Prescales["L1_Mu3p5_DoubleEG5"] = 0;
		Prescales["L1_Mu5_DoubleEG5"] = 1 ;
		Prescales["L1_Mu5_DoubleEG6"] = 1;

		Prescales["L1_MuOpen_EG12"] = 1;
		Prescales["L1_Mu3p5_EG12"] = 1;

		Prescales["L1_DoubleJetC36_ETM30"] = 1;
		Prescales["L1_DoubleJetC44_ETM30"] = 1;


// -- Jets 
		Prescales["L1_SingleJet16"] = 1111;
		Prescales["L1_SingleJet36"] = 12;
		Prescales["L1_SingleJet52"] = 1;
		Prescales["L1_SingleJet68"] = 1;
		Prescales["L1_SingleJet92"] = 1 ;
		Prescales["L1_SingleJet128"] =1;

		Prescales["L1_DoubleJetC36"] = 1;
		Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 1;
		Prescales["L1_DoubleJetC52"] = 1;
		Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
		Prescales["L1_DoubleJetC56"] = 1;
		Prescales["L1_DoubleJetC64"] = 1;

		Prescales["L1_TripleJet_64_44_24_VBF"] = 1;
		Prescales["L1_TripleJet_64_48_28_VBF"] = 1;
		Prescales["L1_TripleJet_68_48_32_VBF"] = 1;

// Prescales["L1_TripleJet28_Central"] = 500; 
		Prescales["L1_TripleJetC_52_28_28"] = 1;

		Prescales["L1_QuadJetC32"] = 1;            
		Prescales["L1_QuadJetC36"] = 1;
		Prescales["L1_QuadJetC40"] = 1;

		Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
		Prescales["L1_ETM30"] = 1;
		Prescales["L1_ETM36"] = 1;
		Prescales["L1_ETM40"] = 1;
		Prescales["L1_ETM50"] = 1;
		Prescales["L1_ETM70"] = 1;
		Prescales["L1_ETM100"] = 1;

		Prescales["L1_HTT100"] = 1;
		Prescales["L1_HTT125"] = 1;
		Prescales["L1_HTT150"] = 1;
		Prescales["L1_HTT175"] = 1;
		Prescales["L1_HTT200"] = 1;

		Prescales["L1_ETT300"] = 1;


// -- Egamma
		Prescales["L1_SingleEG5"] = 160;
		Prescales["L1_SingleEG7"] = 40;
		Prescales["L1_SingleEG12"] = 1;
		Prescales["L1_SingleEG18er"] = 1;
// Prescales["L1_SingleIsoEG18er"] = 10;
		Prescales["L1_SingleIsoEG18er"] = 1;

		Prescales["L1_SingleEG20"] = 1;
		Prescales["L1_SingleIsoEG20er"] = 0;
		Prescales["L1_SingleEG22"] = 1;
		Prescales["L1_SingleEG24"] = 1;
		Prescales["L1_SingleEG30"] = 1;

// Prescales["L1_DoubleEG_15_5"] = 1;
		Prescales["L1_DoubleEG_13_7"] = 1;

		Prescales["L1_TripleEG7"] = 1;
		Prescales["L1_TripleEG_12_7_5"] = 1;


// -- Muons 

		Prescales["L1_SingleMu12"] = 0;

		Prescales["L1_SingleMu20"] = 1;
		Prescales["L1_SingleMu16"] = 1;
		Prescales["L1_SingleMu12"] = 1;
		Prescales["L1_SingleMu7"] = 1;
		Prescales["L1_SingleMu3"] = 100;
		Prescales["L1_SingleMuOpen"] = 150;
		Prescales["L1_DoubleMu0"] = 1;

		Prescales["L1_SingleMu25er"] = 1;
		Prescales["L1_SingleMu20er"] = 1;
// Prescales["L1_SingleMu14er"] = 50;
		Prescales["L1_SingleMu14er"] =  1;
		Prescales["L1_SingleMu18er"] = 1;
		Prescales["L1_SingleMu16er"] = 1;
		Prescales["L1_DoubleMu_12_5"] = 1;
		Prescales["L1_DoubleMu5"] = 1;
		Prescales["L1_DoubleMu0er_HighQ"] = 1;
		Prescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_10_Open"] = 1 ;
		Prescales["L1_DoubleMu_10_3p5"] = 1;
		Prescales["L1_TripleMu0"] = 1;
		Prescales["L1_TripleMu0_HighQ"] = 1;
		Prescales["L1_TripleMu_5_5_0"] = 1;

	}

// target lumi = 2e33
	if (theTargetLumi == 20) {

	// -- Cross 
			Prescales["L1_Mu0_HTT50"] = INFTY;
			Prescales["L1_Mu0_HTT100"] = 1;
			Prescales["L1_Mu4_HTT125"] = 1;

			Prescales["L1_Mu12er_ETM20"] = 1;

			Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 1;
			Prescales["L1_Mu10er_JetC32"] = 1 ;
			Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

			Prescales["L1_Mu3_JetC16_WdEtaPhi2"] = 20;  
			Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 1;
			Prescales["L1_Mu8_DoubleJetC20"] = 50;
	// Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;

			Prescales["L1_EG22_ForJet24"] = 1;
			Prescales["L1_EG22_ForJet32"] = 1;
			Prescales["L1_DoubleEG6_HTT100"] = 1;
			Prescales["L1_DoubleEG6_HTT125"] = 1;

			Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 1;
	// Prescales["L1_EG18er_CentralJet32_or_TauJet24_deltaPhi1"] = 1;
			Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
			Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
			Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 1;

			Prescales["L1_EG8_DoubleJetC20"] = 50 ;  

			Prescales["L1_Mu12_EG7"] = 1;

			Prescales["L1_DoubleMu3p5_EG5"] = 1;
			Prescales["L1_DoubleMu5_EG5"] = 1;

	// Prescales["L1_Mu3p5_DoubleEG5"] = 0;
			Prescales["L1_Mu5_DoubleEG5"] = 1 ;
			Prescales["L1_Mu5_DoubleEG6"] = 1;

			Prescales["L1_MuOpen_EG12"] = 1;
			Prescales["L1_Mu3p5_EG12"] = 1;

			Prescales["L1_DoubleJetC36_ETM30"] = 1;
			Prescales["L1_DoubleJetC44_ETM30"] = 1;


	// -- Jets 
			Prescales["L1_SingleJet16"] = 15000;
			Prescales["L1_SingleJet36"] = 1000;
			Prescales["L1_SingleJet52"] = 5;
			Prescales["L1_SingleJet68"] = 1;
			Prescales["L1_SingleJet92"] = 1;
			Prescales["L1_SingleJet128"] =1;

			Prescales["L1_DoubleJetC36"] = 40;
			Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 1;
			Prescales["L1_DoubleJetC52"] = INFTY;
			Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
			Prescales["L1_DoubleJetC56"] = 1;
			Prescales["L1_DoubleJetC64"] = 1;

			Prescales["L1_TripleJet_64_44_24_VBF"] = 1;
			Prescales["L1_TripleJet_64_48_28_VBF"] = 1;
			Prescales["L1_TripleJet_68_48_32_VBF"] = 1;

	// Prescales["L1_TripleJet28_Central"] = 500; 
			Prescales["L1_TripleJetC_52_28_28"] = 1;

			Prescales["L1_QuadJetC32"] = INFTY;            
			Prescales["L1_QuadJetC36"] = 1;
			Prescales["L1_QuadJetC40"] = 1;

			Prescales["L1_DoubleTauJet44er"] = 1;

	// -- Sums
			Prescales["L1_ETM30"] = 1;
			Prescales["L1_ETM36"] = 1;
			Prescales["L1_ETM40"] = 1;
			Prescales["L1_ETM50"] = 1;
			Prescales["L1_ETM70"] = 1;
			Prescales["L1_ETM100"] = 1;

			Prescales["L1_HTT100"] = 1;
			Prescales["L1_HTT125"] = 1;
			Prescales["L1_HTT150"] = 1;
			Prescales["L1_HTT175"] = 1;
			Prescales["L1_HTT200"] = 1;

			Prescales["L1_ETT300"] = 1;


	// -- Egamma
			Prescales["L1_SingleEG5"] = 200;
			Prescales["L1_SingleEG7"] = 20;
			Prescales["L1_SingleEG12"] = 2;
			Prescales["L1_SingleEG18er"] = 1;
			// Prescales["L1_SingleIsoEG18er"] = 10;
			Prescales["L1_SingleIsoEG18er"] = 1;

			Prescales["L1_SingleEG20"] = 1;
			Prescales["L1_SingleIsoEG20er"] = 0;
			Prescales["L1_SingleEG22"] = 1;
			Prescales["L1_SingleEG24"] = 1;
			Prescales["L1_SingleEG30"] = 1;

			// Prescales["L1_DoubleEG_15_5"] = 1;
			Prescales["L1_DoubleEG_13_7"] = 1;

			Prescales["L1_TripleEG7"] = 1;
			Prescales["L1_TripleEG_12_7_5"] = 1;


	// -- Muons 

			Prescales["L1_SingleMu12"] = 0;

			Prescales["L1_SingleMu20"] = 1;
			Prescales["L1_SingleMu16"] = 1;
			Prescales["L1_SingleMu12"] = 10;
			Prescales["L1_SingleMu7"] = 50;
			Prescales["L1_SingleMu3"] = 500;
			Prescales["L1_SingleMuOpen"] = 1000;
			Prescales["L1_DoubleMu0"] = 5;

			Prescales["L1_SingleMu25er"] = 1;
			Prescales["L1_SingleMu20er"] = 1;
	// Prescales["L1_SingleMu14er"] = 50;
			Prescales["L1_SingleMu14er"] =  1;
			Prescales["L1_SingleMu18er"] = 1;
			Prescales["L1_SingleMu16er"] = 1;
			Prescales["L1_DoubleMu_12_5"] = 1;
			Prescales["L1_DoubleMu5"] = 5;
			Prescales["L1_DoubleMu0er_HighQ"] = 1;
			Prescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
			Prescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
			Prescales["L1_DoubleMu_10_Open"] = 1 ;
			Prescales["L1_DoubleMu_10_3p5"] = 1;
			Prescales["L1_TripleMu0"] = INFTY;
			Prescales["L1_TripleMu0_HighQ"] = 1;
			Prescales["L1_TripleMu_5_5_0"] = 1;



		}

// target lumi = 3e33
	if (theTargetLumi == 30) {

		// -- Cross 
		Prescales["L1_Mu0_HTT50"] = INFTY;
		Prescales["L1_Mu0_HTT100"] = 1;
		Prescales["L1_Mu4_HTT125"] = 1;

		Prescales["L1_Mu12er_ETM20"] = 1;

		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 1;
		Prescales["L1_Mu10er_JetC32"] = 1 ;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

		Prescales["L1_Mu3_JetC16_WdEtaPhi2"] =40;  
		Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 1;
		Prescales["L1_Mu8_DoubleJetC20"] = 50;
		// Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;

		Prescales["L1_EG22_ForJet24"] = 1;
		Prescales["L1_EG22_ForJet32"] = 1;
		Prescales["L1_DoubleEG6_HTT100"] = 1;
		Prescales["L1_DoubleEG6_HTT125"] = 1;

		Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 1;
		// Prescales["L1_EG18er_CentralJet32_or_TauJet24_deltaPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
		Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 1;

		Prescales["L1_EG8_DoubleJetC20"] = 100 ;  

		Prescales["L1_Mu12_EG7"] = 1;

		Prescales["L1_DoubleMu3p5_EG5"] = 1;
		Prescales["L1_DoubleMu5_EG5"] = 1;

		// Prescales["L1_Mu3p5_DoubleEG5"] = 0;
		Prescales["L1_Mu5_DoubleEG5"] = 1 ;
		Prescales["L1_Mu5_DoubleEG6"] = 1;

		Prescales["L1_MuOpen_EG12"] = 1;
		Prescales["L1_Mu3p5_EG12"] = 1;

		Prescales["L1_DoubleJetC36_ETM30"] = 1;
		Prescales["L1_DoubleJetC44_ETM30"] = 1;


		// -- Jets 
		Prescales["L1_SingleJet16"] = 20000;
		Prescales["L1_SingleJet36"] = 1200;
		Prescales["L1_SingleJet52"] = 100;
		Prescales["L1_SingleJet68"] = 1;
		Prescales["L1_SingleJet92"] = 1;
		Prescales["L1_SingleJet128"] =1;

		Prescales["L1_DoubleJetC36"] = 80;
		Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 1;
		Prescales["L1_DoubleJetC52"] = INFTY;
		Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
		Prescales["L1_DoubleJetC56"] = 1;
		Prescales["L1_DoubleJetC64"] = 1;

		Prescales["L1_TripleJet_64_44_24_VBF"] = 1;
		Prescales["L1_TripleJet_64_48_28_VBF"] = 1;
		Prescales["L1_TripleJet_68_48_32_VBF"] = 1;

		// Prescales["L1_TripleJet28_Central"] = 500; 
		Prescales["L1_TripleJetC_52_28_28"] = 1;

		Prescales["L1_QuadJetC32"] = INFTY;            
		Prescales["L1_QuadJetC36"] = 1;
		Prescales["L1_QuadJetC40"] = 1;

		Prescales["L1_DoubleTauJet44er"] = 1;

		// -- Sums
		Prescales["L1_ETM30"] = 100;
		Prescales["L1_ETM36"] = 1;
		Prescales["L1_ETM40"] = 1;
		Prescales["L1_ETM50"] = 1;
		Prescales["L1_ETM70"] = 1;
		Prescales["L1_ETM100"] = 1;

		Prescales["L1_HTT100"] = 1;
		Prescales["L1_HTT125"] = 1;
		Prescales["L1_HTT150"] = 1;
		Prescales["L1_HTT175"] = 1;
		Prescales["L1_HTT200"] = 1;

		Prescales["L1_ETT300"] = 1;


		// -- Egamma
		Prescales["L1_SingleEG5"] = 500;
		Prescales["L1_SingleEG7"] = 50;
		Prescales["L1_SingleEG12"] = 10;
		Prescales["L1_SingleEG18er"] = 1;
		// Prescales["L1_SingleIsoEG18er"] = 10;
		Prescales["L1_SingleIsoEG18er"] = 1;

		Prescales["L1_SingleEG20"] = 1;
		Prescales["L1_SingleIsoEG20er"] = 0;
		Prescales["L1_SingleEG22"] = 1;
		Prescales["L1_SingleEG24"] = 1;
		Prescales["L1_SingleEG30"] = 1;

		// Prescales["L1_DoubleEG_15_5"] = 1;
		Prescales["L1_DoubleEG_13_7"] = 1;

		Prescales["L1_TripleEG7"] = 1;
		Prescales["L1_TripleEG_12_7_5"] = 1;


		// -- Muons 

		Prescales["L1_SingleMu12"] = 0;

		Prescales["L1_SingleMu20"] = 1;
		Prescales["L1_SingleMu16"] = 100;
		Prescales["L1_SingleMu12"] = 100;
		Prescales["L1_SingleMu7"] = 100;
		Prescales["L1_SingleMu3"] = 500;
		Prescales["L1_SingleMuOpen"] = 2000;
		Prescales["L1_DoubleMu0"] = 50;

		Prescales["L1_SingleMu25er"] = 1;
		Prescales["L1_SingleMu20er"] = 1;
		// Prescales["L1_SingleMu14er"] = 50;
		Prescales["L1_SingleMu14er"] =  1;
		Prescales["L1_SingleMu18er"] = 1;
		Prescales["L1_SingleMu16er"] = 1;
		Prescales["L1_DoubleMu_12_5"] = 1;
		Prescales["L1_DoubleMu5"] = 50;
		Prescales["L1_DoubleMu0er_HighQ"] = 1;
		Prescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_10_Open"] = 1 ;
		Prescales["L1_DoubleMu_10_3p5"] = 1;
		Prescales["L1_TripleMu0"] = INFTY;
		Prescales["L1_TripleMu0_HighQ"] = 1;
		Prescales["L1_TripleMu_5_5_0"] = 1;



	}

// target lumi = 5e33
	if (theTargetLumi == 50) {

// -- Cross 
		Prescales["L1_Mu0_HTT50"] = INFTY;
		Prescales["L1_Mu0_HTT100"] = 1;
		Prescales["L1_Mu4_HTT125"] = 1;

		Prescales["L1_Mu12er_ETM20"] = 1;

		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 1;
		Prescales["L1_Mu10er_JetC32"] = 1 ;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

		Prescales["L1_Mu3_JetC16_WdEtaPhi2"] =80;  
		Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 1;
		Prescales["L1_Mu8_DoubleJetC20"] = 100;
// Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;

		Prescales["L1_EG22_ForJet24"] = 1;
		Prescales["L1_EG22_ForJet32"] = 1;
		Prescales["L1_DoubleEG6_HTT100"] = 1;
		Prescales["L1_DoubleEG6_HTT125"] = 1;

		Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 1;
// Prescales["L1_EG18er_CentralJet32_or_TauJet24_deltaPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
		Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 1;

		Prescales["L1_EG8_DoubleJetC20"] = 650 ;  

		Prescales["L1_Mu12_EG7"] = 1;

		Prescales["L1_DoubleMu3p5_EG5"] = 1;
		Prescales["L1_DoubleMu5_EG5"] = 1;

// Prescales["L1_Mu3p5_DoubleEG5"] = 0;
		Prescales["L1_Mu5_DoubleEG5"] = 1 ;
		Prescales["L1_Mu5_DoubleEG6"] = 1;

		Prescales["L1_MuOpen_EG12"] = 1;
		Prescales["L1_Mu3p5_EG12"] = 1;

		Prescales["L1_DoubleJetC36_ETM30"] = 1;
		Prescales["L1_DoubleJetC44_ETM30"] = 1;


// -- Jets 
		Prescales["L1_SingleJet16"] = 40000;
		Prescales["L1_SingleJet36"] = 2400;
		Prescales["L1_SingleJet52"] = 200;
		Prescales["L1_SingleJet68"] = 50;
		Prescales["L1_SingleJet92"] = 10 ;
		Prescales["L1_SingleJet128"] =1;

		Prescales["L1_DoubleJetC36"] = 160;
		Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 3;
		Prescales["L1_DoubleJetC52"] = INFTY;
		Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
		Prescales["L1_DoubleJetC56"] = 1;
		Prescales["L1_DoubleJetC64"] = 1;

		Prescales["L1_TripleJet_64_44_24_VBF"] = 1;
		Prescales["L1_TripleJet_64_48_28_VBF"] = 1;
		Prescales["L1_TripleJet_68_48_32_VBF"] = 1;

// Prescales["L1_TripleJet28_Central"] = 500; 
		Prescales["L1_TripleJetC_52_28_28"] = 40;

		Prescales["L1_QuadJetC32"] = INFTY;            
		Prescales["L1_QuadJetC36"] = 1;
		Prescales["L1_QuadJetC40"] = 1;

		Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
		Prescales["L1_ETM30"] = 100;
		Prescales["L1_ETM36"] = 1;
		Prescales["L1_ETM40"] = 1;
		Prescales["L1_ETM50"] = 1;
		Prescales["L1_ETM70"] = 1;
		Prescales["L1_ETM100"] = 1;

		Prescales["L1_HTT100"] = INFTY;
		Prescales["L1_HTT125"] = INFTY;
		Prescales["L1_HTT150"] = 1;
		Prescales["L1_HTT175"] = 1;
		Prescales["L1_HTT200"] = 1;

		Prescales["L1_ETT300"] = 1;


// -- Egamma
		Prescales["L1_SingleEG5"] = 3000;
		Prescales["L1_SingleEG7"] = 400;
		Prescales["L1_SingleEG12"] = 200;
		Prescales["L1_SingleEG18er"] = 10;
// Prescales["L1_SingleIsoEG18er"] = 10;
		Prescales["L1_SingleIsoEG18er"] = 1;

		Prescales["L1_SingleEG20"] = 1;
		Prescales["L1_SingleIsoEG20er"] = 0;
		Prescales["L1_SingleEG22"] = 1;
		Prescales["L1_SingleEG24"] = 1;
		Prescales["L1_SingleEG30"] = 1;

// Prescales["L1_DoubleEG_15_5"] = 1;
		Prescales["L1_DoubleEG_13_7"] = 1;

		Prescales["L1_TripleEG7"] = 1;
		Prescales["L1_TripleEG_12_7_5"] = 1;


// -- Muons 

		Prescales["L1_SingleMu12"] = 0;

		Prescales["L1_SingleMu20"] = 100;
		Prescales["L1_SingleMu16"] = 100;
		Prescales["L1_SingleMu12"] = 200;
		Prescales["L1_SingleMu7"] = 400;
		Prescales["L1_SingleMu3"] = 2000;
		Prescales["L1_SingleMuOpen"] = 3500;
		Prescales["L1_DoubleMu0"] = 150;

		Prescales["L1_SingleMu25er"] = 1;
		Prescales["L1_SingleMu20er"] = 1;
// Prescales["L1_SingleMu14er"] = 50;
		Prescales["L1_SingleMu14er"] =  1;
		Prescales["L1_SingleMu18er"] = 1;
		Prescales["L1_SingleMu16er"] = 1;
		Prescales["L1_DoubleMu_12_5"] = 1;
		Prescales["L1_DoubleMu5"] = 50;
		Prescales["L1_DoubleMu0er_HighQ"] = 1;
		Prescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_10_Open"] = 1 ;
		Prescales["L1_DoubleMu_10_3p5"] = 1;
		Prescales["L1_TripleMu0"] = INFTY;
		Prescales["L1_TripleMu0_HighQ"] = 1;
		Prescales["L1_TripleMu_5_5_0"] = 1;



	}

// target lumi = 5e33 (emergency)
	if (theTargetLumi == 50.001 ) {

// -- Cross 
		Prescales["L1_Mu0_HTT50"] = INFTY;
		Prescales["L1_Mu0_HTT100"] = INFTY;
		Prescales["L1_Mu4_HTT125"] = 1;

		Prescales["L1_Mu12er_ETM20"] = 2;

		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 40;
		Prescales["L1_Mu10er_JetC32"] = INFTY ;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

		Prescales["L1_Mu3_JetC16_WdEtaPhi2"] = 150 *2;
		Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 2 *5;
		Prescales["L1_Mu8_DoubleJetC20"] = 200;
// Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;
		Prescales["L1_EG22_ForJet24"] = 1;
		Prescales["L1_EG22_ForJet32"] = 1;
		Prescales["L1_DoubleEG6_HTT100"] = INFTY;
		Prescales["L1_DoubleEG6_HTT125"] = 1;

		Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = INFTY;
// Prescales["L1_EG18er_CentralJet32_or_TauJet24_deltaPhi1"] = INFTY;
		Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = INFTY;
		Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
		Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = INFTY;

		Prescales["L1_EG8_DoubleJetC20"] = 1000 ;


		Prescales["L1_Mu12_EG7"] = 1;

		Prescales["L1_DoubleMu3p5_EG5"] = INFTY ;
		Prescales["L1_DoubleMu5_EG5"] = 1;

// Prescales["L1_Mu3p5_DoubleEG5"] = 0 ;
		Prescales["L1_Mu5_DoubleEG5"] = INFTY ;
		Prescales["L1_Mu5_DoubleEG6"] = 1;

		Prescales["L1_MuOpen_EG12"] = INFTY;
		Prescales["L1_Mu3p5_EG12"] = 1;
		Prescales["L1_DoubleJetC36_ETM30"] = INFTY;
		Prescales["L1_DoubleJetC44_ETM30"] = 1;

// -- Jets 
		Prescales["L1_SingleJet16"] = 40000 * 2;
		Prescales["L1_SingleJet36"] = 6000 *2;
		Prescales["L1_SingleJet52"] = 500 *2;
		Prescales["L1_SingleJet68"] = 100 *2;
		Prescales["L1_SingleJet92"] = 20 *2;
		Prescales["L1_SingleJet128"] =1;

		Prescales["L1_DoubleJetC36"] = 320 *2;

// Prescales["L1_DoubleJet44_Central"] = 0;
// Prescales["L1_DoubleJetC52"] = 40;
// Prescales["L1_DoubleJetC56"] = 1;

// Prescales["L1_DoubleJet44_Central"] = 0;
		Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;

		Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 8;
		Prescales["L1_DoubleJetC52"] = 0;
		Prescales["L1_DoubleJetC56"] = INFTY;

		Prescales["L1_DoubleJetC64"] = 1;

		Prescales["L1_TripleJet_64_44_24_VBF"] = 0;
		Prescales["L1_TripleJet_64_48_28_VBF"] = INFTY;
		Prescales["L1_TripleJet_68_48_32_VBF"] = 1;
// Prescales["L1_TripleJet28_Central"] = 1000;         
		Prescales["L1_TripleJetC_52_28_28"] = 100 *2;

		Prescales["L1_QuadJetC32"] = INFTY;             // better estimate on 178208
		Prescales["L1_QuadJetC36"] = INFTY;
		Prescales["L1_QuadJetC40"] = 1;

		Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
		Prescales["L1_ETM30"] = 300;
		Prescales["L1_ETM36"] = INFTY;
		Prescales["L1_ETM40"] = 1;
		Prescales["L1_ETM50"] = 1;
		Prescales["L1_ETM70"] = 1;
		Prescales["L1_ETM100"] = 1;

		Prescales["L1_HTT50"] = 0;
// Prescales["L1_HTT100"] = 40;
// Prescales["L1_HTT125"] = 20;
		Prescales["L1_HTT100"] = INFTY;
		Prescales["L1_HTT125"] = INFTY;
		Prescales["L1_HTT150"] = INFTY;
		Prescales["L1_HTT175"] = INFTY;
// Prescales["L1_HTT175"] = 1;
		Prescales["L1_HTT200"] = 1;

		Prescales["L1_ETT300"] = 1;


// -- Egamma
		Prescales["L1_SingleEG5"] = 4500;
		Prescales["L1_SingleEG7"] = 800;
		Prescales["L1_SingleEG12"] = 300;
		Prescales["L1_SingleEG18er"] = 80; 
		Prescales["L1_SingleIsoEG18er"] = 15;
		Prescales["L1_SingleEG20"] = INFTY;
		Prescales["L1_SingleIsoEG20er"] = 0;
		Prescales["L1_SingleEG22"] = 1;
		Prescales["L1_SingleEG24"] = 1;
		Prescales["L1_SingleEG30"] = 1;

// Prescales["L1_DoubleEG_15_5"] = 8;
		Prescales["L1_DoubleEG_13_7"] = 1;

		Prescales["L1_TripleEG7"] = 1;
		Prescales["L1_TripleEG_12_7_5"] = 1;


// -- Muons 

		Prescales["L1_SingleMu12"] = 0;

		Prescales["L1_SingleMu20"] = 150;
		Prescales["L1_SingleMu16"] = 150;
		Prescales["L1_SingleMu12"] = 300;
		Prescales["L1_SingleMu7"] = 600;
		Prescales["L1_SingleMu3"] = 4000;
		Prescales["L1_SingleMuOpen"] = 7000;
		Prescales["L1_DoubleMu0"] = 300;

		Prescales["L1_SingleMu25er"] = 1;
		Prescales["L1_SingleMu20er"] = 1;
		Prescales["L1_SingleMu14er"] = 75;
// Prescales["L1_SingleMu18er"] = 1;
		Prescales["L1_SingleMu18er"] = INFTY;
		Prescales["L1_SingleMu16er"] = INFTY;
// Prescales["L1_SingleMu16er"] = 1;
		Prescales["L1_DoubleMu_12_5"] = 1;
		Prescales["L1_DoubleMu5"] = 50; 
// Prescales["L1_DoubleMu0_HighQ_Eta1p7"] = 1;
// Prescales["L1_DoubleMu3p5_HighQ_Eta2p1"] = 1;
		Prescales["L1_DoubleMu0er_HighQ"] = 50;
		Prescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_10_Open"] = INFTY ;

		Prescales["L1_DoubleMu_10_3p5"] = 1;
// Prescales["L1_DoubleMu_10_3p5"] = INFTY;
		Prescales["L1_TripleMu0"] = INFTY;
		Prescales["L1_TripleMu0_HighQ"] = 1;
		Prescales["L1_TripleMu_5_5_0"] = 1;

	}

// target lumi = 6e33
	if (theTargetLumi == 60) {

// -- Cross 
		Prescales["L1_Mu0_HTT50"] = INFTY;
		Prescales["L1_Mu0_HTT100"] = INFTY;
		Prescales["L1_Mu4_HTT125"] = 1;

		Prescales["L1_Mu12er_ETM20"] = 2;

// Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = INFTY;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 40;
		Prescales["L1_Mu10er_JetC32"] = INFTY ;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

		Prescales["L1_Mu3_JetC16_WdEtaPhi2"] = 150 *2;  
		Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 2 *5;
		Prescales["L1_Mu8_DoubleJetC20"] = 200;
// Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;
		Prescales["L1_EG22_ForJet24"] = 1;
		Prescales["L1_EG22_ForJet32"] = 1;
		Prescales["L1_DoubleEG6_HTT100"] = INFTY;
		Prescales["L1_DoubleEG6_HTT125"] = 1;

		Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = INFTY;
		Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
		Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = INFTY;

		Prescales["L1_EG8_DoubleJetC20"] = 1000 ;	


		Prescales["L1_Mu12_EG7"] = 1;

		Prescales["L1_DoubleMu3p5_EG5"] = INFTY ;
		Prescales["L1_DoubleMu5_EG5"] = 1;

// Prescales["L1_Mu3p5_DoubleEG5"] = 0 ;
		Prescales["L1_Mu5_DoubleEG5"] = INFTY ;
		Prescales["L1_Mu5_DoubleEG6"] = 1;

		Prescales["L1_MuOpen_EG12"] = INFTY;
		Prescales["L1_Mu3p5_EG12"] = 1;
		Prescales["L1_DoubleJetC36_ETM30"] = INFTY;
		Prescales["L1_DoubleJetC44_ETM30"] = 1;

// -- Jets 
		Prescales["L1_SingleJet16"] = 50000;
		Prescales["L1_SingleJet36"] = 6000;
		Prescales["L1_SingleJet52"] = 500;
		Prescales["L1_SingleJet68"] = 100;
		Prescales["L1_SingleJet92"] = 20 ;
		Prescales["L1_SingleJet128"] =1;

		Prescales["L1_DoubleJetC36"] = 320;
		Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 6;
		Prescales["L1_DoubleJetC52"] = INFTY;
		Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
		Prescales["L1_DoubleJetC56"] = 1;
		Prescales["L1_DoubleJetC64"] = 1;

		Prescales["L1_TripleJet_64_44_24_VBF"] = 0;
		Prescales["L1_TripleJet_64_48_28_VBF"] = INFTY;
		Prescales["L1_TripleJet_68_48_32_VBF"] = 1;

// Prescales["L1_TripleJet28_Central"] = 500; 
		Prescales["L1_TripleJetC_52_28_28"] = 100;

		Prescales["L1_QuadJetC32"] = INFTY;            
		Prescales["L1_QuadJetC36"] = INFTY;
		Prescales["L1_QuadJetC40"] = 1;

		Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
		Prescales["L1_ETM30"] = 300;
		Prescales["L1_ETM36"] = INFTY;
		Prescales["L1_ETM40"] = 1;
		Prescales["L1_ETM50"] = 1;
		Prescales["L1_ETM70"] = 1;
		Prescales["L1_ETM100"] = 1;

		Prescales["L1_HTT100"] = INFTY;
		Prescales["L1_HTT125"] = INFTY;
		Prescales["L1_HTT150"] = 1;
		Prescales["L1_HTT175"] = 1;
		Prescales["L1_HTT200"] = 1;

		Prescales["L1_ETT300"] = 1;


// -- Egamma
		Prescales["L1_SingleEG5"] = 4000;
		Prescales["L1_SingleEG7"] = 800;
		Prescales["L1_SingleEG12"] = 300;
		Prescales["L1_SingleEG18er"] = 80;
		Prescales["L1_SingleIsoEG18er"] = 15;

		Prescales["L1_SingleEG20"] = 1;
		Prescales["L1_SingleIsoEG20er"] = 0;
		Prescales["L1_SingleEG22"] = 1;
		Prescales["L1_SingleEG24"] = 1;
		Prescales["L1_SingleEG30"] = 1;

// Prescales["L1_DoubleEG_15_5"] = 1;
		Prescales["L1_DoubleEG_13_7"] = 1;

		Prescales["L1_TripleEG7"] = 1;
		Prescales["L1_TripleEG_12_7_5"] = 1;


// -- Muons 

		Prescales["L1_SingleMu12"] = 0;

		Prescales["L1_SingleMu20"] = 150;
		Prescales["L1_SingleMu16"] = 150;
		Prescales["L1_SingleMu12"] = 300;
		Prescales["L1_SingleMu7"] = 600;
		Prescales["L1_SingleMu3"] = 4000;
		Prescales["L1_SingleMuOpen"] = 7000;
		Prescales["L1_DoubleMu0"] = 300;

		Prescales["L1_SingleMu25er"] = 1;
		Prescales["L1_SingleMu20er"] = 1;
		Prescales["L1_SingleMu14er"] = 75;
		Prescales["L1_SingleMu18er"] = 1;
		Prescales["L1_SingleMu16er"] = 1;
		Prescales["L1_DoubleMu_12_5"] = 1;
		Prescales["L1_DoubleMu5"] = 50;
		Prescales["L1_DoubleMu0er_HighQ"] = 1;
		Prescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_10_Open"] = INFTY ;
		Prescales["L1_DoubleMu_10_3p5"] = 1;
		Prescales["L1_TripleMu0"] = INFTY;
		Prescales["L1_TripleMu0_HighQ"] = 1;
		Prescales["L1_TripleMu_5_5_0"] = 1;



	}

// target lumi = 7e33
	if (theTargetLumi == 70 ) {

// -- Cross 
		Prescales["L1_Mu0_HTT50"] = INFTY;
		Prescales["L1_Mu0_HTT100"] = INFTY;
		Prescales["L1_Mu4_HTT125"] = 1;

		Prescales["L1_Mu12er_ETM20"] = 2;

// Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = INFTY;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 40;
		Prescales["L1_Mu10er_JetC32"] = INFTY ;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

		Prescales["L1_Mu3_JetC16_WdEtaPhi2"] = 150 *2;  
		Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 2 *5;
		Prescales["L1_Mu8_DoubleJetC20"] = 200;
// Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;
		Prescales["L1_EG22_ForJet24"] = 1;
		Prescales["L1_EG22_ForJet32"] = 1;
		Prescales["L1_DoubleEG6_HTT100"] = INFTY;
		Prescales["L1_DoubleEG6_HTT125"] = 1;

		Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = INFTY;
		Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
		Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
		Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = INFTY;

		Prescales["L1_EG8_DoubleJetC20"] = 1000 ;	


		Prescales["L1_Mu12_EG7"] = 1;

		Prescales["L1_DoubleMu3p5_EG5"] = INFTY ;
		Prescales["L1_DoubleMu5_EG5"] = 1;

// Prescales["L1_Mu3p5_DoubleEG5"] = 0 ;
		Prescales["L1_Mu5_DoubleEG5"] = INFTY ;
		Prescales["L1_Mu5_DoubleEG6"] = 1;

		Prescales["L1_MuOpen_EG12"] = INFTY;
		Prescales["L1_Mu3p5_EG12"] = 1;
		Prescales["L1_DoubleJetC36_ETM30"] = INFTY;
		Prescales["L1_DoubleJetC44_ETM30"] = 1;

// -- Jets 
		Prescales["L1_SingleJet16"] = 50000;
		Prescales["L1_SingleJet36"] = 6000;
		Prescales["L1_SingleJet52"] = 500;
		Prescales["L1_SingleJet68"] = 100;
		Prescales["L1_SingleJet92"] = 20 ;
		Prescales["L1_SingleJet128"] =1;

		Prescales["L1_DoubleJetC36"] = 320;

// Prescales["L1_DoubleJet44_Central"] = 0;
// Prescales["L1_DoubleJetC52"] = 40;
// Prescales["L1_DoubleJetC56"] = 1;

// Prescales["L1_DoubleJet44_Central"] = 0;

		Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
		Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 6;


		Prescales["L1_DoubleJetC52"] = 0;
		Prescales["L1_DoubleJetC56"] = INFTY;

		Prescales["L1_DoubleJetC64"] = 1;

		Prescales["L1_TripleJet_64_44_24_VBF"] = 0;
		Prescales["L1_TripleJet_64_48_28_VBF"] = INFTY;
		Prescales["L1_TripleJet_68_48_32_VBF"] = 1;

		Prescales["L1_TripleJetC_52_28_28"] = 100;

		Prescales["L1_QuadJetC32"] = INFTY;		// better estimate on 178208
		Prescales["L1_QuadJetC36"] = INFTY;
		Prescales["L1_QuadJetC40"] = 1;

		Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
		Prescales["L1_ETM30"] = 300;
		Prescales["L1_ETM36"] = INFTY;
		Prescales["L1_ETM40"] = 1;
		Prescales["L1_ETM50"] = 1;
		Prescales["L1_ETM70"] = 1;
		Prescales["L1_ETM100"] = 1;

		Prescales["L1_HTT50"] = 0;
// Prescales["L1_HTT100"] = 40;
// Prescales["L1_HTT125"] = 20;
		Prescales["L1_HTT100"] = INFTY;
		Prescales["L1_HTT125"] = INFTY;
		Prescales["L1_HTT150"] = INFTY;
// Prescales["L1_HTT175"] = INFTY;
		Prescales["L1_HTT175"] = 1;
		Prescales["L1_HTT200"] = 1;

		Prescales["L1_ETT300"] = 1;


// -- Egamma
		Prescales["L1_SingleEG5"] = 4500;
		Prescales["L1_SingleEG7"] = 800;
		Prescales["L1_SingleEG12"] = 300;
		Prescales["L1_SingleEG18er"] = 80; 
		Prescales["L1_SingleIsoEG18er"] = 15;
		Prescales["L1_SingleEG20"] = INFTY;
		Prescales["L1_SingleIsoEG20er"] = 0;
		Prescales["L1_SingleEG22"] = 1;
		Prescales["L1_SingleEG24"] = 1;
		Prescales["L1_SingleEG30"] = 1;

// Prescales["L1_DoubleEG_15_5"] = 8;
		Prescales["L1_DoubleEG_13_7"] = 1;

		Prescales["L1_TripleEG7"] = 1;
		Prescales["L1_TripleEG_12_7_5"] = 1;



// -- Muons 

		Prescales["L1_SingleMu12"] = 0;

		Prescales["L1_SingleMu20"] = 150;
		Prescales["L1_SingleMu16"] = 150;
		Prescales["L1_SingleMu12"] = 300;
		Prescales["L1_SingleMu7"] = 600;
		Prescales["L1_SingleMu3"] = 4000;
		Prescales["L1_SingleMuOpen"] = 7000;
		Prescales["L1_DoubleMu0"] = 300;

		Prescales["L1_SingleMu25er"] = 1;
		Prescales["L1_SingleMu20er"] = 1;
		Prescales["L1_SingleMu14er"] = 75;
		Prescales["L1_SingleMu18er"] = 1;
// Prescales["L1_SingleMu16er"] = INFTY;
		Prescales["L1_SingleMu16er"] = 1;
		Prescales["L1_DoubleMu_12_5"] = 1;
		Prescales["L1_DoubleMu5"] = 50;
// Prescales["L1_DoubleMu0_HighQ_Eta1p7"] = 1;
// Prescales["L1_DoubleMu3p5_HighQ_Eta2p1"] = 1;

		Prescales["L1_DoubleMu0er_HighQ"] = 50;
		Prescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;

		Prescales["L1_DoubleMu_10_Open"] = INFTY ;
		Prescales["L1_DoubleMu_10_3p5"] = 1;
		Prescales["L1_TripleMu0"] = INFTY;
		Prescales["L1_TripleMu0_HighQ"] = 1;
		Prescales["L1_TripleMu_5_5_0"] = 1;

	}

// target lumi = 7e33  (emergency)
	if (theTargetLumi == 70.001 ) {

// -- Cross 
		Prescales["L1_Mu0_HTT50"] = INFTY;
		Prescales["L1_Mu0_HTT100"] = INFTY;
		Prescales["L1_Mu4_HTT125"] = 1;

		Prescales["L1_Mu12er_ETM20"] = 2;

		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 40;
		Prescales["L1_Mu10er_JetC32"] = INFTY ;
		Prescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;

		Prescales["L1_Mu3_JetC16_WdEtaPhi2"] = 150 *2;
		Prescales["L1_Mu3_JetC52_WdEtaPhi2"] = 2 *5;
		Prescales["L1_Mu8_DoubleJetC20"] = 200;
// Prescales["L1_Mu16_Jet24_Central_dPhi1"] = 1;
		Prescales["L1_EG22_ForJet24"] = 1;
		Prescales["L1_EG22_ForJet32"] = 1;
		Prescales["L1_DoubleEG6_HTT100"] = INFTY;
		Prescales["L1_DoubleEG6_HTT125"] = 1;

		Prescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = INFTY;
// Prescales["L1_EG18er_CentralJet32_or_TauJet24_deltaPhi1"] = INFTY;
		Prescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = INFTY;
		Prescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
		Prescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = INFTY;

		Prescales["L1_EG8_DoubleJetC20"] = 1000 ;


		Prescales["L1_Mu12_EG7"] = 1;

		Prescales["L1_DoubleMu3p5_EG5"] = INFTY ;
		Prescales["L1_DoubleMu5_EG5"] = 1;

// Prescales["L1_Mu3p5_DoubleEG5"] = 0 ;
		Prescales["L1_Mu5_DoubleEG5"] = INFTY ;
		Prescales["L1_Mu5_DoubleEG6"] = 1;

		Prescales["L1_MuOpen_EG12"] = INFTY;
		Prescales["L1_Mu3p5_EG12"] = 1;
		Prescales["L1_DoubleJetC36_ETM30"] = INFTY;
		Prescales["L1_DoubleJetC44_ETM30"] = 1;

// -- Jets 
		Prescales["L1_SingleJet16"] = 50000 * 2;
		Prescales["L1_SingleJet36"] = 6000 *2;
		Prescales["L1_SingleJet52"] = 500 *2;
		Prescales["L1_SingleJet68"] = 100 *2;
		Prescales["L1_SingleJet92"] = 20 *2;
		Prescales["L1_SingleJet128"] =1;

		Prescales["L1_DoubleJetC36"] = 320 *2;

// Prescales["L1_DoubleJet44_Central"] = 0;
// Prescales["L1_DoubleJetC52"] = 40;
// Prescales["L1_DoubleJetC56"] = 1;

// Prescales["L1_DoubleJet44_Central"] = 0;
		Prescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;

		Prescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 8;
		Prescales["L1_DoubleJetC52"] = 0;
		Prescales["L1_DoubleJetC56"] = INFTY;

		Prescales["L1_DoubleJetC64"] = 1;

		Prescales["L1_TripleJet_64_44_24_VBF"] = 0;
		Prescales["L1_TripleJet_64_48_28_VBF"] = INFTY;
		Prescales["L1_TripleJet_68_48_32_VBF"] = 1;
// Prescales["L1_TripleJet28_Central"] = 1000;         
		Prescales["L1_TripleJetC_52_28_28"] = 100 *2;

		Prescales["L1_QuadJetC32"] = INFTY;             // better estimate on 178208
		Prescales["L1_QuadJetC36"] = INFTY;
		Prescales["L1_QuadJetC40"] = 1;

		Prescales["L1_DoubleTauJet44er"] = 1;

// -- Sums
		Prescales["L1_ETM30"] = 300;
		Prescales["L1_ETM36"] = INFTY;
		Prescales["L1_ETM40"] = 1;
		Prescales["L1_ETM50"] = 1;
		Prescales["L1_ETM70"] = 1;
		Prescales["L1_ETM100"] = 1;

		Prescales["L1_HTT50"] = 0;
// Prescales["L1_HTT100"] = 40;
// Prescales["L1_HTT125"] = 20;
		Prescales["L1_HTT100"] = INFTY;
		Prescales["L1_HTT125"] = INFTY;
		Prescales["L1_HTT150"] = INFTY;
		Prescales["L1_HTT175"] = INFTY;
// Prescales["L1_HTT175"] = 1;
		Prescales["L1_HTT200"] = 1;

		Prescales["L1_ETT300"] = 1;


// -- Egamma
		Prescales["L1_SingleEG5"] = 4500;
		Prescales["L1_SingleEG7"] = 800;
		Prescales["L1_SingleEG12"] = 300;
		Prescales["L1_SingleEG18er"] = 80; 
		Prescales["L1_SingleIsoEG18er"] = 15;
		Prescales["L1_SingleEG20"] = INFTY;
		Prescales["L1_SingleIsoEG20er"] = 0;
		Prescales["L1_SingleEG22"] = 1;
		Prescales["L1_SingleEG24"] = 1;
		Prescales["L1_SingleEG30"] = 1;

// Prescales["L1_DoubleEG_15_5"] = 8;
		Prescales["L1_DoubleEG_13_7"] = 1;

		Prescales["L1_TripleEG7"] = 1;
		Prescales["L1_TripleEG_12_7_5"] = 1;


// -- Muons 

		Prescales["L1_SingleMu12"] = 0;

		Prescales["L1_SingleMu20"] = 150;
		Prescales["L1_SingleMu16"] = 150;
		Prescales["L1_SingleMu12"] = 300;
		Prescales["L1_SingleMu7"] = 600;
		Prescales["L1_SingleMu3"] = 4000;
		Prescales["L1_SingleMuOpen"] = 7000;
		Prescales["L1_DoubleMu0"] = 300;

		Prescales["L1_SingleMu25er"] = 1;
		Prescales["L1_SingleMu20er"] = 1;
		Prescales["L1_SingleMu14er"] = 75;
// Prescales["L1_SingleMu18er"] = 1;
		Prescales["L1_SingleMu18er"] = INFTY;
		Prescales["L1_SingleMu16er"] = INFTY;
// Prescales["L1_SingleMu16er"] = 1;
		Prescales["L1_DoubleMu_12_5"] = 1;
		Prescales["L1_DoubleMu5"] = 50; 
// Prescales["L1_DoubleMu0_HighQ_Eta1p7"] = 1;
// Prescales["L1_DoubleMu3p5_HighQ_Eta2p1"] = 1;
		Prescales["L1_DoubleMu0er_HighQ"] = 50;
		Prescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
		Prescales["L1_DoubleMu_10_Open"] = INFTY ;

		Prescales["L1_DoubleMu_10_3p5"] = 1;
// Prescales["L1_DoubleMu_10_3p5"] = INFTY;
		Prescales["L1_TripleMu0"] = INFTY;
		Prescales["L1_TripleMu0_HighQ"] = 1;
		Prescales["L1_TripleMu_5_5_0"] = 1;

	}

/*
// -- test  to see where we stand if we would get rid of all p'ed seeds
// -- (in 2011 we spent ~ 20% of the rate in monitoring / control p'ed seeds..)

for (map<string, int>::iterator it=Prescales.begin(); it != Prescales.end(); it++) {
string name = it -> first;
Int_t p = it -> second;
if (p > 1 ) Prescales[name] = 0;
}
*/


// -- Each seed gets a "weight" according to how many PAGS are using it

for (map<string, int>::iterator it=Prescales.begin(); it != Prescales.end(); it++) {
	string name = it -> first;
	Int_t UsedPernPAG = 0;
	if ( setTOP.count(name) > 0) UsedPernPAG ++;
	if ( setHIGGS.count(name) > 0) UsedPernPAG ++;
	if ( setSUSY.count(name) > 0) UsedPernPAG ++;
	if ( setEXO.count(name) > 0) UsedPernPAG ++;
	if ( setSMP.count(name) > 0) UsedPernPAG ++;
	if ( setBPH.count(name) > 0) UsedPernPAG ++;
	WeightsPAGs[name] = 1./(float)UsedPernPAG;
}


for (map<string, int>::iterator it=Prescales.begin(); it != Prescales.end(); it++) {
	string name = it -> first;
	Counts[name] = 0;
	Biased[name] = false; 
}


// -- The "Biased" table is only used for the final print-out
// -- set true for seeds for which the rate estimation is biased by
// -- the sample (because of the seeds enabled in the high PU run)

// Biased["L1_TripleMu0"] = true;
// Biased["L1_DoubleMu_10_Open"] = true;
// Biased["L1_SingleEG5"] = true;
// Biased["L1_TripleEG7"] = true;
// Biased["L1_TripleEG_10_7_5"] = true;
// Biased["L1_SingleJet36"] = true;
// Biased["L1_DoubleJetC36"] = true;
// Biased["L1_DoubleJetC44_Eta1p74_WdEta4"] = true;
// Biased["L1_QuadJetC36"] = true;
// Biased["L1_QuadJetC40"] = true;
// Biased["L1_DoubleTauJet44er"] = true;
// Biased["L1_Mu3p5_DoubleEG5"] = true;
// Biased["L1_QuadJetC32"] = true;
// Biased["L1_TripleJet28_Central"] = true;



}

Bool_t L1Menu2012::SingleMuEta2p1(Float_t ptcut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t muon = false;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) { 
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		Float_t eta = gmt_  -> Eta[imu];        
		if (fabs(eta) > 2.1) continue;
		if (pt >= ptcut) muon = true;
	}

	Bool_t ok = muon;
	return ok;

}

Bool_t L1Menu2012::SingleMu(Float_t ptcut, Int_t qualmin) {

	Bool_t raw = PhysicsBits[0];  // ZeroBias
	if (! raw) return false;

	Bool_t muon = false;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		//    BX = 0, +/- 1 or +/- 2
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];		
		if ( qual < qualmin) continue;
		if (pt >= ptcut) muon = true;
	}

	Bool_t ok = muon;
	return ok;

}

Bool_t L1Menu2012::DoubleMuHighQEtaCut(Float_t ptcut, Float_t etacut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Int_t nmu=0;
	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		//    BX = 0, +/- 1 or +/- 2
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];		
		if ( qual < 4) continue;
		Float_t eta = gmt_  -> Eta[imu];		
		if (fabs(eta) > etacut) continue;
		if (pt >= ptcut) nmu ++;
	}

	Bool_t ok = (nmu >= 2 ) ;
	return ok;

}

Bool_t L1Menu2012::Onia(Float_t ptcut1, Float_t ptcut2, Float_t etacut, Int_t delta) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Int_t Nmu = gmt_ -> N;
	Int_t n1=0;
	Int_t n2=0;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];		
		if ( qual < 4) continue;
		Float_t eta = gmt_  -> Eta[imu];		
		if (fabs(eta) > etacut) continue;
		if (pt >= ptcut1) n1 ++;
		if (pt >= ptcut2) n2++;
	}

	Bool_t ok = (n1 >=1 && n2 >= 2 ) ;
	if (! ok) return false;

	// -- now the CORRELATION condition
	Bool_t CORREL = false;

	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		Float_t eta = gmt_  -> Eta[imu];        
		if (fabs(eta) > etacut) continue;
		if (pt < ptcut1) continue;
		Int_t ieta1 = etaMuIdx(eta);

		for (Int_t imu2=0; imu2 < Nmu; imu2++) {
			if (imu2 == imu) continue;
			Int_t bx2 = gmt_ -> CandBx[imu2];		
			if (bx2 != 0) continue;
			Float_t pt2 = gmt_ -> Pt[imu2];			
			Int_t qual2 = gmt_ -> Qual[imu2];        
			if ( qual2 < 4) continue;
			Float_t eta2 = gmt_  -> Eta[imu2];        
			if (fabs(eta2) > etacut) continue;
			if (pt2 < ptcut2) continue;
			Int_t ieta2 = etaMuIdx(eta2);

			Float_t deta = ieta1 - ieta2; 
		// cout << "eta 1 2 delta " << ieta1 << " " << ieta2 << " " << deta << endl;
			if ( fabs(deta) <= delta)  CORREL = true;
		// if (fabs ( eta - eta2) <=  1.7) CORREL = true; 
		}

	}

	return CORREL;

}

Bool_t L1Menu2012::DoubleMu(Float_t cut1, Float_t cut2) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias
	if (! raw) return false;  

	Int_t n1=0;
	Int_t n2=0;
	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		// if ( qual < 4) continue;
		if (qual < 4  && qual != 3 ) continue;
		if (pt >= cut1) n1 ++;
		if (pt >= cut2) n2 ++;
	}

	Bool_t ok = (n1 >= 1 && n2 >= 2 );
	return ok;

}

Bool_t L1Menu2012::TripleMu(Float_t cut1, Float_t cut2, Float_t cut3, Int_t qualmin) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias
	if (! raw) return false;

	Int_t n1=0;
	Int_t n2=0;
	Int_t n3=0;
	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < qualmin) continue;
		if (pt >= cut1) n1 ++;
		if (pt >= cut2) n2 ++;
		if (pt >= cut3) n3 ++;
	}

	Bool_t ok = ( n1 >= 1 && n2 >= 2 && n3 >= 3 );
	return ok;

}

Bool_t L1Menu2012::DoubleMuXOpen(Float_t cut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias
	if (! raw) return false;

	Int_t n1=0;
	Int_t n2=0;
	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( (qual >= 5 || qual == 3 ) && pt >= cut ) n1 ++;
		if ( pt >= 0 ) n2 ++;
	}

	Bool_t ok = ( n1 >= 1 && n2 >= 2 );
	return ok;
}

Bool_t L1Menu2012::Muons() {

	insert_ibin = 0;
	InsertInMenu("L1_SingleMuOpen",SingleMu(0.,0));
	InsertInMenu("L1_SingleMu3",SingleMu(3.));
	InsertInMenu("L1_SingleMu7",SingleMu(7.));
	InsertInMenu("L1_SingleMu12",SingleMu(12.));
	InsertInMenu("L1_SingleMu16",SingleMu(16.));
	InsertInMenu("L1_SingleMu20",SingleMu(20.));

	InsertInMenu("L1_SingleMu14er",SingleMuEta2p1(14.));
	InsertInMenu("L1_SingleMu16er",SingleMuEta2p1(16.));
	InsertInMenu("L1_SingleMu18er",SingleMuEta2p1(18.));
	InsertInMenu("L1_SingleMu20er",SingleMuEta2p1(20.));
	InsertInMenu("L1_SingleMu25er",SingleMuEta2p1(25.));

	InsertInMenu("L1_DoubleMu0",DoubleMu(0.,0.));
	InsertInMenu("L1_DoubleMu0er_HighQ",DoubleMuHighQEtaCut(0.,2.1));
	InsertInMenu("L1_DoubleMu3er_HighQ_WdEta22",Onia(3.,3.,2.1,22));
	InsertInMenu("L1_DoubleMu_5er_0er_HighQ_WdEta22",Onia(5.,0.,2.1,22));

	InsertInMenu("L1_DoubleMu5",DoubleMu(5.,5.));
	InsertInMenu("L1_DoubleMu_12_5",DoubleMu(12.,5.));
	InsertInMenu("L1_DoubleMu_10_Open",DoubleMuXOpen(10.));
	InsertInMenu("L1_DoubleMu_10_3p5",DoubleMu(10.,3.5));

	InsertInMenu("L1_TripleMu0",TripleMu(0.,0.,0.,3));
	InsertInMenu("L1_TripleMu0_HighQ",TripleMu(0.,0.,0.,4));
	InsertInMenu("L1_TripleMu_5_5_0",TripleMu(5.,5.,0.,3));

	Int_t NN = insert_ibin;

	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += insert_ibin;

	if (first) {

		NBITS_MUONS = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = (TString)insert_names[ibin];
			h_Muons -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
		}
		h_Muons -> GetXaxis() -> SetBinLabel(NN+1, "MUONS") ;

		for (Int_t k=1; k <= kOFFSET - kOFFSET_old ; k++) {
			h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Muons -> GetXaxis() -> GetBinLabel(k) );
		}
	}

	Bool_t res = false;
	for (Int_t i=0; i < NN; i++) {
		res = res || insert_val[i] ;
		if (insert_val[i]) h_Muons -> Fill(i);
	}
	if (res) h_Muons -> Fill(NN);

	return res;
}

Bool_t L1Menu2012::Mu_EG(Float_t mucut, Float_t EGcut ) {

	Bool_t raw = PhysicsBits[0];    // ZeroBias
	if (! raw) return false;


	Bool_t eg =false;
	Bool_t muon = false;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {   
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt >= mucut) muon = true;
	}

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;
	}  // end loop over EM objects

	Bool_t ok = muon && eg;
	return ok;

}

Bool_t L1Menu2012::DoubleMu_EG(Float_t mucut, Float_t EGcut ) {

	Bool_t raw = PhysicsBits[0]; 	// ZeroBias
	if (! raw) return false;

	Bool_t eg =false;
	Bool_t muon = false;
	Int_t  Nmuons = 0;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		// if ( qual < 4) continue;
		if (qual < 4 && qual !=3 ) continue;
		if (pt >= mucut) Nmuons ++;
	}
	if (Nmuons >= 2) muon = true;

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;
	}  // end loop over EM objects

	Bool_t ok = muon && eg;
	return ok;

}

Bool_t L1Menu2012::Mu_DoubleEG(Float_t mucut, Float_t EGcut ) {

	Bool_t raw = PhysicsBits[0];  // ZeroBias..
	if (! raw) return false;

	Bool_t eg =false;
	Bool_t muon = false;
	Int_t  Nmuons = 0;
	Int_t Nelectrons = 0;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt >= mucut) Nmuons ++;
	}
	if (Nmuons >= 1) muon = true;

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) Nelectrons ++;
	}  // end loop over EM objects
	if (Nelectrons >= 2) eg = true;

	Bool_t ok = muon && eg;
	return ok;

}

Bool_t L1Menu2012::MuOpen_EG(Float_t mucut, Float_t EGcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;


	Bool_t eg =false;
	Bool_t muon = false;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		if (pt >= mucut) muon = true;
	}

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;
	}  // end loop over EM objects

	Bool_t ok = muon && eg;
	return ok;

}

Bool_t L1Menu2012::Mu_JetCentral(Float_t mucut, Float_t jetcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;
	Bool_t muon = false;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut) jet = true;
	}

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt >= mucut) muon = true;
	}

	Bool_t ok = muon && jet;
	return ok;

}

Bool_t L1Menu2012::Mu_DoubleJetCentral(Float_t mucut, Float_t jetcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;
	Bool_t muon = false;

	Int_t n1 = 0;
	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut) n1 ++;
	}
	jet = ( n1 >= 2 );

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt >= mucut) muon = true;
	}

	Bool_t ok = muon && jet;
	return ok;

}

Bool_t L1Menu2012::Mu_JetCentral_LowerTauTh(Float_t mucut, Float_t jetcut, Float_t taucut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;
	Bool_t central = false;
	Bool_t tau = false;
	Bool_t muon = false;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		Bool_t isTauJet = gt_ -> Taujet[ue];
		if (! isTauJet) {  	// look at CentralJet
			if (pt >= jetcut) central = true;
		}
		else   {		// look at TauJets
			if (pt >= taucut) tau = true;
		}
	}
	jet = central || tau  ;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt >= mucut) muon = true;
	}

	Bool_t ok = muon && jet;
	return ok;

}

Bool_t L1Menu2012::Muer_JetCentral(Float_t mucut, Float_t jetcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;
	Bool_t muon = false;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut) jet = true;
	}

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		Float_t eta = gmt_  -> Eta[imu];        
		if (fabs(eta) > 2.1) continue;

		if (pt >= mucut) muon = true;
	}

	Bool_t ok = muon && jet;
	return ok;

}

Bool_t L1Menu2012::Muer_JetCentral_LowerTauTh(Float_t mucut, Float_t jetcut, Float_t taucut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;
	Bool_t central = false;
	Bool_t tau = false;
	Bool_t muon = false;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		Bool_t isTauJet = gt_ -> Taujet[ue];
		if (! isTauJet) {       // look at CentralJet
			if (pt >= jetcut) central = true;
		}
		else   {                // look at TauJets
			if (pt >= taucut) tau = true;
		}
	}
	jet = central || tau  ;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		Float_t eta = gmt_  -> Eta[imu];        
		if (fabs(eta) > 2.1) continue;
		if (pt >= mucut) muon = true;
	}

	Bool_t ok = muon && jet;
	return ok;

}

Bool_t L1Menu2012::Mia(Float_t mucut, Float_t jet1, Float_t jet2) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;
	Bool_t muon = false;
	Int_t n1 = 0;
	Int_t n2 = 0;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jet1) n1 ++;
		if (pt >= jet2) n2 ++;
	}       
	jet = (n1 >= 1 && n2 >= 2 );

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		Float_t eta = gmt_  -> Eta[imu];        
		if (fabs(eta) > 2.1) continue;        
		if (pt >= mucut) muon = true;
	} 

	Bool_t ok = muon && jet;
	if (! ok) return false;

	// now the CORREL condition


	Bool_t CORREL = false;

	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt < mucut) continue;
		Float_t eta = gmt_  -> Eta[imu];        
		if (fabs(eta) > 2.1) continue;

		Float_t phimu = gmt_ -> Phi[imu];
		Int_t iphi_mu = phiINjetCoord(phimu);
		Float_t etamu = gmt_ -> Eta[imu];
		Int_t ieta_mu = etaINjetCoord(etamu);

		for (Int_t ue=0; ue < Nj; ue++) {
			Int_t bxj = gt_ -> Bxjet[ue];        		
			if (bxj != 0) continue;
			Bool_t isFwdJet = gt_ -> Fwdjet[ue];
			if (isFwdJet) continue;
			Float_t rank = gt_ -> Rankjet[ue];
			Float_t ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
			if (ptj < jet2) continue;
			Float_t phijet = gt_ -> Phijet[ue];
			Int_t iphi_jet = (int)phijet;
			Float_t etajet = gt_ -> Etajet[ue];
			Int_t ieta_jet = (int)etajet;

			Bool_t corr_phi = correlateInPhi(iphi_jet, iphi_mu);
			Bool_t corr_eta = correlateInEta(ieta_jet, ieta_mu);
			Bool_t corr = corr_phi && corr_eta;
			if (corr) CORREL = true ;
		}
	}

	return CORREL;

}

Bool_t L1Menu2012::Mu_JetCentral_delta(Float_t mucut, Float_t jetcut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;
	Bool_t muon = false;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut) jet = true;
	}

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt >= mucut) muon = true;
	}

	Bool_t ok = muon && jet;
	if (! ok) return false;

		//  -- now evaluate the delta condition :

	Bool_t CORREL = false;

	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt < mucut) continue;

		Float_t phimu = gmt_ -> Phi[imu];
		Int_t iphi_mu = phiINjetCoord(phimu);
		Float_t etamu = gmt_ -> Eta[imu];
		Int_t ieta_mu = etaINjetCoord(etamu);

		for (Int_t ue=0; ue < Nj; ue++) {
			Int_t bxj = gt_ -> Bxjet[ue];        		
			if (bxj != 0) continue;
			Bool_t isFwdJet = gt_ -> Fwdjet[ue];
			if (isFwdJet) continue;
			Float_t rank = gt_ -> Rankjet[ue];
			Float_t ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
			if (ptj < jetcut) continue;
			Float_t phijet = gt_ -> Phijet[ue];
			Int_t iphi_jet = (int)phijet;
			Float_t etajet = gt_ -> Etajet[ue];
			Int_t ieta_jet = (int)etajet;

			Bool_t corr = correlateInPhi(iphi_jet, iphi_mu, 2) && correlateInEta(ieta_jet, ieta_mu, 2);
			if (corr) CORREL = true ;
		}
	}

	return CORREL;

}

Bool_t L1Menu2012::Mu_JetCentral_deltaOut(Float_t mucut, Float_t jetcut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;
	Bool_t muon = false;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut) jet = true;
	}

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt >= mucut) muon = true;
	}

	Bool_t ok = muon && jet;
	if (! ok) return false;

		//  -- now evaluate the delta condition :

	Bool_t CORREL = false;

	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt < mucut) continue;

		Float_t phimu = gmt_ -> Phi[imu];
		Int_t iphi_mu = phiINjetCoord(phimu);
//          Float_t etamu = gmt_ -> Eta[imu];
//          Int_t ieta_mu = etaINjetCoord(etamu);

		Int_t PhiOut[3];
		PhiOut[0] = iphi_mu;
		if (iphi_mu< 17) PhiOut[1] = iphi_mu+1;
		if (iphi_mu == 17) PhiOut[1] = 0;
		if (iphi_mu > 0) PhiOut[2] = iphi_mu - 1;
		if (iphi_mu == 0) PhiOut[2] = 17;

		for (Int_t ue=0; ue < Nj; ue++) {
			Int_t bxj = gt_ -> Bxjet[ue];        		
			if (bxj != 0) continue;
			Bool_t isFwdJet = gt_ -> Fwdjet[ue];
			if (isFwdJet) continue;
			Float_t rank = gt_ -> Rankjet[ue];
			Float_t ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
			if (ptj < jetcut) continue;
			Float_t phijet = gt_ -> Phijet[ue];
			Int_t iphi_jet = (int)phijet;
//                  Float_t etajet = gt_ -> Etajet[ue];
//                  Int_t ieta_jet = (int)etajet;

			if (! correlateInPhi(iphi_jet, iphi_mu, 8)) CORREL = true;


		}
	}

	return CORREL;

}

Bool_t L1Menu2012::Muer_TripleJetCentral(Float_t mucut, Float_t jet1, Float_t jet2, Float_t jet3)  {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;
	Bool_t muon = false;

	Int_t n1=0;
	Int_t n2=0;
	Int_t n3=0;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jet1) n1 ++;
		if (pt >= jet2) n2 ++;
		if (pt >= jet3) n3 ++;
	}

	jet = ( n1 >= 1 && n2 >= 2 && n3 >= 3 ) ;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		Float_t eta = gmt_ -> Eta[imu] ;
		if (fabs(eta) > 2.1 ) continue;
		if (pt >= mucut) muon = true;
	}

	Bool_t ok = muon && jet;
	return ok;

}

Bool_t L1Menu2012::Mu_HTT(Float_t mucut, Float_t HTcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t ht=false;
	Bool_t muon = false;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		if (pt >= mucut) muon = true;
	}

	Float_t adc = gt_ -> RankHTT ;
	Float_t TheHTT = adc / 2. ;
	ht = (TheHTT >= HTcut) ;

	Bool_t ok = muon && ht;
	return ok;

}

Bool_t L1Menu2012::Muer_ETM(Float_t mucut, Float_t ETMcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t etm = false;
	Bool_t muon = false;

	Int_t Nmu = gmt_ -> N;
	for (Int_t imu=0; imu < Nmu; imu++) {
		Int_t bx = gmt_ -> CandBx[imu];		
		if (bx != 0) continue;
		Float_t pt = gmt_ -> Pt[imu];			
		Int_t qual = gmt_ -> Qual[imu];        
		if ( qual < 4) continue;
		Float_t eta = gmt_  -> Eta[imu];        
		if (fabs(eta) > 2.1) continue;

		if (pt >= mucut) muon = true;
	}

	Float_t adc = gt_ -> RankETM ;
	Float_t TheETM = adc / 2. ;
	etm = (TheETM >= ETMcut);

	Bool_t ok = muon && etm;
	return ok;

}

Bool_t L1Menu2012::EG_FwdJet(Float_t EGcut, Float_t FWcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t eg = false;
	Bool_t jet = false;

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;
	}  // end loop over EM objects

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {        
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (!isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= FWcut) jet = true;
	}

	Bool_t ok = ( eg && jet);
	return ok;

}

Bool_t L1Menu2012::EG_DoubleJetCentral(Float_t EGcut, Float_t jetcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t eg = false;
	Bool_t jet = false;

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;
	}  // end loop over EM objects

	Int_t Nj = gt_ -> Njet ;
	Int_t njets = 0;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut) njets ++;
	}
	jet = ( njets >= 2 );

	Bool_t ok = ( eg && jet);
	return ok;

}

Bool_t L1Menu2012::EG_HT(Float_t EGcut, Float_t HTcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t eg = false;
	Bool_t ht = false;

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;
	}  // end loop over EM objects

	Float_t adc = gt_ -> RankHTT ;
	Float_t TheHTT = adc / 2. ;
	ht = (TheHTT >= HTcut) ;

	Bool_t ok = ( eg && ht);
	return ok;

}

Bool_t L1Menu2012::DoubleEG_HT(Float_t EGcut, Float_t HTcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t eg = false;
	Int_t n1 = 0;
	Bool_t ht = false;

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) n1 ++;
	}  // end loop over EM objects
	eg = ( n1 >= 2 );

	Float_t adc = gt_ -> RankHTT ;
	Float_t TheHTT = adc / 2. ;
	ht = (TheHTT >= HTcut) ;

	Bool_t ok = ( eg && ht);
	return ok;

}

Bool_t L1Menu2012::EGEta2p1_JetCentral(Float_t EGcut, Float_t jetcut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;
	
	Bool_t eg = false;
	Bool_t jet = false;

	Int_t Nele = gt_ -> Nele; 
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;
	}  // end loop over EM objects

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut) jet = true;
	}

	Bool_t ok = (eg && jet);
	if (! ok) return false;


	//  -- now evaluate the delta condition :

	Bool_t CORREL = false;
	Int_t PhiOut[3];

	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt < EGcut) continue;

		Float_t phiel = gt_ -> Phiel[ue];
		Int_t iphiel = (int)phiel;

		PhiOut[0]=0; PhiOut[1]=0; PhiOut[2]=0;   

		PhiOut[0] = iphiel;
		if (iphiel< 17) PhiOut[1] = iphiel+1;
		if (iphiel == 17) PhiOut[1] = 0;
		if (iphiel > 0) PhiOut[2] = iphiel - 1;
		if (iphiel == 0) PhiOut[2] = 17;

		for (Int_t uj=0; uj < Nj; uj++) {
			Int_t bxj = gt_ -> Bxjet[uj];        		
			if (bxj != 0) continue;
			Bool_t isFwdJet = gt_ -> Fwdjet[uj];
			if (isFwdJet) continue;
			Float_t rankj = gt_ -> Rankjet[uj];
			// Float_t ptj = rankj * 4;
			Float_t ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[uj],rankj*4.,theL1JetCorrection);
			if (ptj < jetcut) continue;
			Float_t phijet = gt_ -> Phijet[uj];
			Int_t iphijet = (int)phijet; 

			if ( iphijet != PhiOut[0] && 
				iphijet != PhiOut[1] &&
				iphijet != PhiOut[2] ) CORREL = true;
		}  // loop over jets

	}  // end loop over EM objects

	return CORREL;
	
}

Bool_t L1Menu2012::EGEta2p1_JetCentral_LowTauTh(Float_t EGcut, Float_t jetcut, Float_t taucut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t eg = false;
	Bool_t jet = false;
	Bool_t central = false;
	Bool_t tau = false;

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;
	}  // end loop over EM objects

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Bool_t isTauJet = gt_ -> Taujet[ue];
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (! isTauJet) {
			if (pt >= jetcut) central = true;
		}
		else {
			if (pt >= taucut) tau = true;
		}
	}
	jet = tau || central;

	Bool_t ok = (eg && jet);
	if (! ok) return false;

	//  -- now evaluate the delta condition :

	Bool_t CORREL_CENTRAL = false;
	Bool_t CORREL_TAU = false;
	Int_t PhiOut[3];

	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt < EGcut) continue;

		Float_t phiel = gt_ -> Phiel[ue];
		Int_t iphiel = (int)phiel;

		PhiOut[0]=0; PhiOut[1]=0; PhiOut[2]=0;   

		PhiOut[0] = iphiel;
		if (iphiel< 17) PhiOut[1] = iphiel+1;
		if (iphiel == 17) PhiOut[1] = 0;
		if (iphiel > 0) PhiOut[2] = iphiel - 1;
		if (iphiel == 0) PhiOut[2] = 17;

		for (Int_t uj=0; uj < Nj; uj++) {
			Int_t bxj = gt_ -> Bxjet[uj];        		
			if (bxj != 0) continue;
			Bool_t isFwdJet = gt_ -> Fwdjet[uj];
			if (isFwdJet) continue;
			Bool_t isTauJet = gt_ -> Taujet[uj];
			Float_t rankj = gt_ -> Rankjet[uj];
			Float_t ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[uj],rankj*4.,theL1JetCorrection);
			Float_t phijet = gt_ -> Phijet[uj];
			Int_t iphijet = (int)phijet;

			if (! isTauJet) {

				if (ptj >= jetcut) { 
					if ( iphijet != PhiOut[0] &&
						iphijet != PhiOut[1] &&
						iphijet != PhiOut[2] ) CORREL_CENTRAL = true;
				}

			}
			else {
				if (ptj >= taucut) {
					if ( iphijet != PhiOut[0] &&
						iphijet != PhiOut[1] &&
						iphijet != PhiOut[2] ) CORREL_TAU = true;
				}

			}


		}  // loop over jets

	}  // end loop over EM objects

	Bool_t CORREL = CORREL_CENTRAL || CORREL_TAU ;
	return CORREL;

}

Bool_t L1Menu2012::IsoEGEta2p1_JetCentral_LowTauTh(Float_t EGcut, Float_t jetcut, Float_t taucut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t eg = false;
	Bool_t jet = false;
	Bool_t central = false;
	Bool_t tau = false;

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Bool_t iso = gt_ -> Isoel[ue];
		if ( ! iso) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;
	}  // end loop over EM objects

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Bool_t isTauJet = gt_ -> Taujet[ue];
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (! isTauJet) {
			if (pt >= jetcut) central = true;
		}
		else {
			if (pt >= taucut) tau = true;
		}
	}
	jet = tau || central;

	Bool_t ok = (eg && jet);
	if (! ok) return false;

		//  -- now evaluate the delta condition :

	Bool_t CORREL_CENTRAL = false;
	Bool_t CORREL_TAU = false;
	Int_t PhiOut[3];

	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Bool_t iso = gt_ -> Isoel[ue];
		if ( ! iso) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt < EGcut) continue;

		Float_t phiel = gt_ -> Phiel[ue];
		Int_t iphiel = (int)phiel;

		PhiOut[0]=0; PhiOut[1]=0; PhiOut[2]=0;   

		PhiOut[0] = iphiel; 
		if (iphiel< 17) PhiOut[1] = iphiel+1;
		if (iphiel == 17) PhiOut[1] = 0;
		if (iphiel > 0) PhiOut[2] = iphiel - 1;
		if (iphiel == 0) PhiOut[2] = 17;

		for (Int_t uj=0; uj < Nj; uj++) {
			Int_t bxj = gt_ -> Bxjet[uj];        		
			if (bxj != 0) continue;
			Bool_t isFwdJet = gt_ -> Fwdjet[uj];
			if (isFwdJet) continue;
			Bool_t isTauJet = gt_ -> Taujet[uj];
			Float_t rankj = gt_ -> Rankjet[uj];
			Float_t ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[uj],rankj*4.,theL1JetCorrection);
			Float_t phijet = gt_ -> Phijet[uj];
			Int_t iphijet = (int)phijet;

			if (! isTauJet) {

				if (ptj >= jetcut) {
					if ( iphijet != PhiOut[0] &&
						iphijet != PhiOut[1] &&
						iphijet != PhiOut[2] ) CORREL_CENTRAL = true;
				}

			}
			else {
				if (ptj >= taucut) {
					if ( iphijet != PhiOut[0] &&
						iphijet != PhiOut[1] &&
						iphijet != PhiOut[2] ) CORREL_TAU = true;
				}

			}


		}  // loop over jets

	}  // end loop over EM objects

	Bool_t CORREL = CORREL_CENTRAL || CORREL_TAU ;

	return CORREL;

}

Bool_t L1Menu2012::EGEta2p1_DoubleJetCentral(Float_t EGcut, Float_t jetcut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t eg = false;
	Bool_t jet = false;
	Int_t n2=0;

	Int_t Nele = gt_ -> Nele; 
	for (Int_t ue=0; ue < Nele; ue++) { 
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true; 
	}  // end loop over EM objects

	Int_t Nj = gt_ -> Njet ;               
	for (Int_t ue=0; ue < Nj; ue++) {      
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue; 
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut) n2 ++;
	}

	jet = (n2 >= 2);

	Bool_t ok = (eg && jet);
	if (! ok) return false;

		//  -- now evaluate the delta condition :

	Bool_t CORREL = false;
	Int_t PhiOut[3];

	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt < EGcut) continue;

		Float_t phiel = gt_ -> Phiel[ue];
		Int_t iphiel = (int)phiel;

		PhiOut[0]=0; PhiOut[1]=0; PhiOut[2]=0; 

		PhiOut[0] = iphiel;
		if (iphiel< 17) PhiOut[1] = iphiel+1;
		if (iphiel == 17) PhiOut[1] = 0;
		if (iphiel > 0) PhiOut[2] = iphiel - 1;
		if (iphiel == 0) PhiOut[2] = 17;

		Int_t npair = 0;

		for (Int_t uj=0; uj < Nj; uj++) {
			Int_t bxj = gt_ -> Bxjet[uj];        		
			if (bxj != 0) continue; 
			Bool_t isFwdJet = gt_ -> Fwdjet[uj];
			if (isFwdJet) continue;
			Float_t rankj = gt_ -> Rankjet[uj];
										// Float_t ptj = rankj * 4;
			Float_t ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[uj],rankj*4.,theL1JetCorrection);
			if (ptj < jetcut) continue;
			Float_t phijet = gt_ -> Phijet[uj];
			Int_t iphijet = (int)phijet;

			if ( iphijet != PhiOut[0] &&
				iphijet != PhiOut[1] &&
				iphijet != PhiOut[2] ) npair ++;

		}  // loop over jets

		if (npair >= 2 ) CORREL = true ;

	}  // end loop over EM objects

	return CORREL;

}

Bool_t L1Menu2012::EGEta2p1_DoubleJetCentral_TripleJetCentral(Float_t EGcut, Float_t jetcut2, Float_t jetcut3) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t eg = false;
	Bool_t jet = false;
	Int_t n2=0;       
	Int_t n3=0;

	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue; 
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= EGcut) eg = true;  
	}  // end loop over EM objects

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut2) n2 ++;
		if (pt >= jetcut3) n3 ++;
	}

	jet = (n2 >= 2 && n3 >= 3 );

	Bool_t ok = (eg && jet);
	return ok;

}

Bool_t L1Menu2012::HTT_HTM(Float_t HTTcut, Float_t HTMcut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t htt = false;
	Bool_t htm = false;
	Float_t adc = gt_ -> RankHTT;   
	Float_t TheHTT = (float)adc / 2.   ;          
	htt = ( TheHTT >= HTTcut ) ;

	Int_t adc_HTM  = gt_  -> RankHTM ; 
	Float_t TheHTM = adc_HTM * 2.  ;           
	htm = ( TheHTM >= HTMcut );

	Bool_t ok = (htt && htm);
	return ok;

}

Bool_t L1Menu2012::JetCentral_ETM(Float_t jetcut, Float_t ETMcut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t etm = false;
	Bool_t jet = false;

	Float_t adc = gt_ -> RankETM ;
	Float_t TheETM = adc / 2. ;
	etm = (TheETM >= ETMcut);

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut) jet = true;
	}

	Bool_t ok = ( jet && etm );
	return ok;
	
}

Bool_t L1Menu2012::DoubleJetCentral_ETM(Float_t jetcut1, Float_t jetcut2, Float_t ETMcut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t etm = false; 
	Bool_t jet = false;
	Int_t n1=0;
	Int_t n2=0;

	Float_t adc = gt_ -> RankETM ;
	Float_t TheETM = adc / 2. ;
	etm = (TheETM >= ETMcut);

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= jetcut1) n1 ++;
		if (pt >= jetcut2) n2 ++;
	}       
	jet = (n1 >= 1 && n2 >= 2);

	Bool_t ok = ( jet && etm );
	return ok;

}

Bool_t L1Menu2012::Cross() {

	insert_ibin = 0;

	InsertInMenu("L1_Mu0_HTT50", Mu_HTT(0.,50.) );
	InsertInMenu("L1_Mu0_HTT100", Mu_HTT(0.,100.) );
	InsertInMenu("L1_Mu4_HTT125", Mu_HTT(4.,125.) );

	InsertInMenu("L1_Mu12er_ETM20", Muer_ETM(12.,20.) );
	InsertInMenu("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12", Mia(10.,20.,12.) );
	InsertInMenu("L1_Mu10er_JetC32", Muer_JetCentral(10.,32.) );
	InsertInMenu("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12", Mia(10.,32.,12.) );


	InsertInMenu("L1_Mu3_JetC16_WdEtaPhi2", Mu_JetCentral_delta(3.,16.) );
	InsertInMenu("L1_Mu3_JetC52_WdEtaPhi2", Mu_JetCentral_delta(3.,52.) );
	InsertInMenu("L1_Mu8_DoubleJetC20", Mu_DoubleJetCentral(8.,20.) );

	InsertInMenu("L1_EG22_ForJet24", EG_FwdJet(22.,24.) );
	InsertInMenu("L1_EG22_ForJet32", EG_FwdJet(22.,32.) );
	InsertInMenu("L1_DoubleEG6_HTT100", DoubleEG_HT(6.,100.) );
	InsertInMenu("L1_DoubleEG6_HTT125", DoubleEG_HT(6.,125.) );

	InsertInMenu("L1_EG18er_JetC_Cen28_Tau20_dPhi1", EGEta2p1_JetCentral_LowTauTh(18.,28.,20.) );
	InsertInMenu("L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1", IsoEGEta2p1_JetCentral_LowTauTh(18.,32.,24.) );
	InsertInMenu("L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1", IsoEGEta2p1_JetCentral_LowTauTh(18.,36.,28.) );
	InsertInMenu("L1_EG18er_JetC_Cen36_Tau28_dPhi1", EGEta2p1_JetCentral_LowTauTh(18.,36.,28.) );

	InsertInMenu("L1_EG8_DoubleJetC20", EG_DoubleJetCentral(8.,20.) );

	InsertInMenu("L1_Mu12_EG7", Mu_EG(12.,7.) );
	InsertInMenu("L1_MuOpen_EG12", MuOpen_EG(0.,12.) );
	InsertInMenu("L1_Mu3p5_EG12", Mu_EG(3.5,12.) );

	InsertInMenu("L1_DoubleMu3p5_EG5", DoubleMu_EG(3.5,5.) );
	InsertInMenu("L1_DoubleMu5_EG5", DoubleMu_EG(5.,5.) );

	InsertInMenu("L1_Mu5_DoubleEG5", Mu_DoubleEG(5., 5.) );
	InsertInMenu("L1_Mu5_DoubleEG6", Mu_DoubleEG(5., 6.) );

	InsertInMenu("L1_DoubleJetC36_ETM30", DoubleJetCentral_ETM(36., 36., 30.) );
	InsertInMenu("L1_DoubleJetC44_ETM30", DoubleJetCentral_ETM(44., 44., 30.) );

	Int_t NN = insert_ibin;
	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += NN;

	if (first) {

		NBITS_CROSS = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = (TString)insert_names[ibin];
			h_Cross -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
		}
		h_Cross-> GetXaxis() -> SetBinLabel(NN+1,"CROSS");

		for (Int_t k=1; k <= kOFFSET - kOFFSET_old; k++) {
			h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Cross -> GetXaxis() -> GetBinLabel(k) );
		}

	}

	Bool_t res = false;
	for (Int_t i=0; i < NN; i++) {
		res = res || insert_val[i] ;
		if (insert_val[i]) h_Cross -> Fill(i);
	}
	if (res) h_Cross -> Fill(NN);

	return res;
}

Bool_t L1Menu2012::SingleJetCentral(Float_t cut ) {

	Bool_t raw = PhysicsBits[0];  // ZeroBias
	if (! raw) return false;

	Bool_t ok=false;
	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue; 
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= cut) ok = true;
	} 

	return ok;

}

Bool_t L1Menu2012::SingleJet(Float_t cut ) {

	Bool_t raw = PhysicsBits[0];  // ZeroBias
	if (! raw) return false;

	Bool_t ok=false;
	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= cut) ok = true;
	}

	return ok;

}

Bool_t L1Menu2012::DoubleJetCentral(Float_t cut1, Float_t cut2 ) {

	Bool_t raw = PhysicsBits[0];  // ZeroBias
	if (! raw) return false;


	Int_t n1=0;
	Int_t n2=0;
	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= cut1) n1++;
		if (pt >= cut2) n2++;
	}
	Bool_t ok = ( n1 >=1 && n2 >= 2);
	return ok;

}

Bool_t L1Menu2012::DoubleJet_Eta1p7_deltaEta4(Float_t cut1, Float_t cut2 ) {

	Bool_t raw = PhysicsBits[0];  // ZeroBias
	if (! raw) return false;

	Int_t n1=0;
	Int_t n2=0;
	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		Float_t eta = gt_ -> Etajet[ue];
		if (eta < 5.5 || eta > 15.5) continue;  // eta = 6 - 15
		if (pt >= cut1) n1++;
		if (pt >= cut2) n2++;
	}
	Bool_t ok = ( n1 >=1 && n2 >= 2);
	if (! ok) return false;

	// -- now the correlation

	Bool_t CORREL = false;

	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		Float_t eta1 = gt_ -> Etajet[ue];
		if (eta1 < 5.5 || eta1 > 15.5) continue;  // eta = 6 - 15
		if (pt < cut1) continue;

		for (Int_t ve=0; ve < Nj; ve++) {
			if (ve == ue) continue;
			Int_t bx2 = gt_ -> Bxjet[ve];        		
			if (bx2 != 0) continue;
			Bool_t isFwdJet2 = gt_ -> Fwdjet[ve];
			if (isFwdJet2) continue;
			Float_t rank2 = gt_ -> Rankjet[ve];
			Float_t pt2 = rank2 * 4;
			Float_t eta2 = gt_ -> Etajet[ve];
			if (eta2 < 5.5 || eta2 > 15.5) continue;  // eta = 6 - 15
			if (pt2 < cut2) continue;

			Bool_t corr = correlateInEta((int)eta1, (int)eta2, 4);
			if (corr) CORREL = true;
		}


	}

	return CORREL ;

}

Bool_t L1Menu2012::DoubleTauJetEta2p17(Float_t cut1, Float_t cut2) {

	Bool_t raw = PhysicsBits[0];  // ZeroBias
	if (! raw) return false;

	Int_t n1=0;
	Int_t n2=0;
	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue; 
		Bool_t isTauJet = gt_ -> Taujet[ue];
		if (! isTauJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];    // the rank of the electron
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		Float_t eta = gt_ -> Etajet[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		if (pt >= cut1) n1++;
		if (pt >= cut2) n2++;
	}  // end loop over jets

	Bool_t ok = ( n1 >=1 && n2 >= 2);
	return ok;

}

Bool_t L1Menu2012::TripleJetCentral(Float_t cut1, Float_t cut2, Float_t cut3 ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Int_t n1=0;
	Int_t n2=0;
	Int_t n3=0;
	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= cut1) n1++;
		if (pt >= cut2) n2++;
		if (pt >= cut3) n3++;
	}

	Bool_t ok = ( n1 >=1 && n2 >= 2 && n3 >= 3 );
	return ok;

}

Bool_t L1Menu2012::TripleJet_VBF(Float_t jet1, Float_t jet2, Float_t jet3 ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t jet=false;        
	Bool_t jetf1=false;           
	Bool_t jetf2=false;   

	Int_t n1=0;
	Int_t n2=0;
	Int_t n3=0;

	Int_t f1=0;
	Int_t f2=0;
	Int_t f3=0;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);

		if (isFwdJet) {
			if (pt >= jet1) f1 ++;
			if (pt >= jet2) f2 ++;
			if (pt >= jet3) f3 ++;              
		} 
		else {
			if (pt >= jet1) n1 ++;
			if (pt >= jet2) n2 ++;
			if (pt >= jet3) n3 ++;
		}    
	}

	jet   = ( n1 >= 1 && n2 >= 2 && n3 >= 3 ) ;        
	jetf1 = ( f1 >= 1 && n2 >= 1 && n3 >= 2 ) ;  // numbers change ofcourse    
	jetf2 = ( n1 >= 1 && f2 >= 1 && n3 >= 2 ) ;  

	Bool_t ok = false;

	if( jet || jetf1 || jetf2 ) ok =true;

	return ok;
}

Bool_t L1Menu2012::QuadJetCentral(Float_t cut1, Float_t cut2, Float_t cut3, Float_t cut4 ) {

// cut1 >= cut2  >= cut3 >= cut4

	// ZeroBias
// Bool_t raw = PhysicsBits[16];  // SingleJet36
// if (! raw) return false;
	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;


	Int_t n1=0;
	Int_t n2=0;
	Int_t n3=0;
	Int_t n4=0;

	Int_t Nj = gt_ -> Njet ;
	for (Int_t ue=0; ue < Nj; ue++) {
		Int_t bx = gt_ -> Bxjet[ue];        		
		if (bx != 0) continue;
		Bool_t isFwdJet = gt_ -> Fwdjet[ue];
		if (isFwdJet) continue;
		Float_t rank = gt_ -> Rankjet[ue];
		Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.,theL1JetCorrection);
		if (pt >= cut1) n1++;
		if (pt >= cut2) n2++;
		if (pt >= cut3) n3++;
		if (pt >= cut4) n4++;
	}

	Bool_t ok = ( n1 >=1 && n2 >= 2 && n3 >= 3 && n4 >= 4);
	return ok;

}

Bool_t L1Menu2012::Jets() {

	insert_ibin = 0;

	InsertInMenu("L1_SingleJet16", SingleJet(16.) );
	InsertInMenu("L1_SingleJet36", SingleJet(36.) );
	InsertInMenu("L1_SingleJet52", SingleJet(52.) );
	InsertInMenu("L1_SingleJet68", SingleJet(68.) );
	InsertInMenu("L1_SingleJet92", SingleJet(92.) );
	InsertInMenu("L1_SingleJet128", SingleJet(128.) );

	InsertInMenu("L1_DoubleJetC36", DoubleJetCentral(36.,36.) );
	InsertInMenu("L1_DoubleJetC44_Eta1p74_WdEta4", DoubleJet_Eta1p7_deltaEta4(44.,44.) );
	InsertInMenu("L1_DoubleJetC52", DoubleJetCentral(52.,52.) );
	InsertInMenu("L1_DoubleJetC56_Eta1p74_WdEta4", DoubleJet_Eta1p7_deltaEta4(56.,56.) );
	InsertInMenu("L1_DoubleJetC56", DoubleJetCentral(56.,56.) );
	InsertInMenu("L1_DoubleJetC64", DoubleJetCentral(64.,64.) );

	InsertInMenu("L1_TripleJet_64_44_24_VBF", TripleJet_VBF(64.,44.,24.) );
	InsertInMenu("L1_TripleJet_64_48_28_VBF", TripleJet_VBF(64.,48.,28.) );
	InsertInMenu("L1_TripleJet_68_48_32_VBF", TripleJet_VBF(68.,48.,32.) );
	InsertInMenu("L1_TripleJetC_52_28_28", TripleJetCentral(52.,28.,28.) );

	InsertInMenu("L1_QuadJetC32", QuadJetCentral(32.,32.,32.,32.) );
	InsertInMenu("L1_QuadJetC36", QuadJetCentral(36.,36.,36.,36.) );
	InsertInMenu("L1_QuadJetC40", QuadJetCentral(40.,40.,40.,40.) );

	InsertInMenu("L1_DoubleTauJet44er", DoubleTauJetEta2p17(44.,44.) );

	Int_t NN = insert_ibin;

	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += insert_ibin;


	if (first) {

		NBITS_JETS = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = (TString)insert_names[ibin];
			h_Jets -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
		}

		h_Jets-> GetXaxis() -> SetBinLabel(NN+1,"JETS");

		for (Int_t k=1; k <= kOFFSET -kOFFSET_old; k++) {
			h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Jets -> GetXaxis() -> GetBinLabel(k) );
		}

	}

	Bool_t res = false;
	for (Int_t i=0; i < NN; i++) {
		res = res || insert_val[i] ;
		if (insert_val[i]) h_Jets -> Fill(i);
	}
	if (res) h_Jets -> Fill(NN);

	return res;
}

Bool_t L1Menu2012::ETM(Float_t ETMcut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Float_t adc = gt_ -> RankETM ;
	Float_t TheETM = adc / 2. ;

	if (TheETM < ETMcut) return false;
	return true;

}

Bool_t L1Menu2012::HTT(Float_t HTTcut) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Float_t adc = gt_ -> RankHTT ;
	Float_t TheHTT = adc / 2. ;

	if (TheHTT < HTTcut) return false;
	return true;

}

Bool_t L1Menu2012::ETT(Float_t ETTcut) {

	Float_t adc = gt_ -> RankETT ;
	Float_t TheETT = adc / 2. ;

	if (TheETT < ETTcut) return false;

	return true;

}

Bool_t L1Menu2012::Sums() {

	insert_ibin = 0;

	InsertInMenu("L1_ETM30", ETM(30.) );
	InsertInMenu("L1_ETM36", ETM(36.) );
	InsertInMenu("L1_ETM40", ETM(40.) );
	InsertInMenu("L1_ETM50", ETM(50.) );
	InsertInMenu("L1_ETM70", ETM(70.) );
	InsertInMenu("L1_ETM100", ETM(100.) );

	InsertInMenu("L1_HTT100", HTT(100.) );
	InsertInMenu("L1_HTT125", HTT(125.) );
	InsertInMenu("L1_HTT150", HTT(150.) );
	InsertInMenu("L1_HTT175", HTT(175.) );
	InsertInMenu("L1_HTT200", HTT(200.) );

	InsertInMenu("L1_ETT300", ETT(300.) );

	Int_t NN = insert_ibin;

	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += insert_ibin;


	if (first) {

		NBITS_SUMS = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = (TString)insert_names[ibin]; 
			h_Sums -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
		}

		h_Sums -> GetXaxis() -> SetBinLabel(NN+1,"SUMS");

		for (Int_t k=1; k <= kOFFSET -kOFFSET_old; k++) {
			h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Sums -> GetXaxis() -> GetBinLabel(k) );
		}
	}

	Bool_t res = false; 
	for (Int_t i=0; i < NN; i++) {
		res = res || insert_val[i] ; 
		if (insert_val[i]) h_Sums -> Fill(i);
	}
	if (res) h_Sums -> Fill(NN);

	return res;
}

Bool_t L1Menu2012::SingleEG(Float_t cut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t ok=false; 
	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {               
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ; 
		if (pt >= cut) ok = true;
	}  // end loop over EM objects

	return ok; 

}

Bool_t L1Menu2012::SingleIsoEG_Eta2p1(Float_t cut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t ok=false;
	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Bool_t iso = gt_ -> Isoel[ue];
		if (! iso) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= cut) ok = true;
	}  // end loop over EM objects

	return ok;

}

Bool_t L1Menu2012::SingleEG_Eta2p1(Float_t cut ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Bool_t ok=false;
	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t eta = gt_ -> Etael[ue];
		if (eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= cut) ok = true;
	}  // end loop over EM objects

	return ok;

}

Bool_t L1Menu2012::DoubleEG(Float_t cut1, Float_t cut2 ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Int_t n1=0;
	Int_t n2=0;
	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {               
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= cut1) n1++;
		if (pt >= cut2) n2++;
	}  // end loop over EM objects

	Bool_t ok = ( n1 >= 1 && n2 >= 2) ;
	return ok;

}

Bool_t L1Menu2012::TripleEG(Float_t cut1, Float_t cut2, Float_t cut3 ) {

	Bool_t raw = PhysicsBits[0];   // ZeroBias  
	if (! raw) return false;

	Int_t n1=0;
	Int_t n2=0;
	Int_t n3=0;
	Int_t Nele = gt_ -> Nele;
	for (Int_t ue=0; ue < Nele; ue++) {
		Int_t bx = gt_ -> Bxel[ue];        		
		if (bx != 0) continue;
		Float_t rank = gt_ -> Rankel[ue];    // the rank of the electron
		Float_t pt = rank ;
		if (pt >= cut1) n1++;
		if (pt >= cut2) n2++;
		if (pt >= cut3) n3++;
	}  // end loop over EM objects

	Bool_t ok = ( n1 >= 1 && n2 >= 2 && n3 >= 3) ;
	return ok;


}

Bool_t L1Menu2012::EGamma() {

	insert_ibin = 0;

	InsertInMenu("L1_SingleEG5", SingleEG(5.) );
	InsertInMenu("L1_SingleEG7", SingleEG(7.) );
	InsertInMenu("L1_SingleEG12", SingleEG(12.) );
	InsertInMenu("L1_SingleEG18er", SingleEG_Eta2p1(18.) );
	InsertInMenu("L1_SingleIsoEG18er", SingleIsoEG_Eta2p1(18.) );
	InsertInMenu("L1_SingleEG20", SingleEG(20.) );
	InsertInMenu("L1_SingleIsoEG20er", SingleIsoEG_Eta2p1(20.) );
	InsertInMenu("L1_SingleEG22", SingleEG(22.) );
	InsertInMenu("L1_SingleEG24", SingleEG(24.) );
	InsertInMenu("L1_SingleEG30", SingleEG(30.) );

// InsertInMenu("L1_DoubleEG_15_5", DoubleEG(15.,5.) );
	InsertInMenu("L1_DoubleEG_13_7", DoubleEG(13.,7.) );

	InsertInMenu("L1_TripleEG7", TripleEG(7.,7.,7.) );
	InsertInMenu("L1_TripleEG_12_7_5", TripleEG(12.,7.,5.) );

	Int_t NN = insert_ibin;

	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += insert_ibin;


	if (first) {

		NBITS_EGAMMA = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = (TString)insert_names[ibin];
			h_Egamma -> GetXaxis() -> SetBinLabel(ibin+1, l1name );
		}
		h_Egamma-> GetXaxis() -> SetBinLabel(NN+1,"EGAMMA");

		for (Int_t k=1; k <= kOFFSET -kOFFSET_old; k++) {
			h_All -> GetXaxis() -> SetBinLabel(k +kOFFSET_old , h_Egamma -> GetXaxis() -> GetBinLabel(k) );
		}
	}                      

	Bool_t res = false;      
	for (Int_t i=0; i < NN; i++) {
		res = res || insert_val[i] ;
		if (insert_val[i]) h_Egamma -> Fill(i);
	}      
	if (res) h_Egamma -> Fill(NN);

	return res;
}       

void L1Menu2012::Loop() {

	Int_t nevents = GetEntries();
//	nevents = 1000;

	Int_t NPASS = 0; 

	Int_t NJETS = 0;
	Int_t NEG = 0;
	Int_t NSUMS =0;
	Int_t NMUONS = 0;
	Int_t NCROSS = 0;

	Int_t nPAG =0;
	first = true;

	for (Long64_t i=0; i<nevents; i++)
	{     
	//load the i-th event
		Long64_t ientry = LoadTree(i); if (ientry < 0) break;
		GetEntry(i);

//      Fill the physics bits:
		FilL1Bits();

		if (first) MyInit();

		Bool_t raw = PhysicsBits[0];  // ZeroBias
		if (! raw) continue;


//  --- Reset the emulated "Trigger Bits"
		kOFFSET = 0;
		for (Int_t k=0; k < N128; k++) {
			TheTriggerBits[k] = false;
		}


		Bool_t jets = false;
		Bool_t eg = false;
		Bool_t sums = false;
		Bool_t muons = false;
		Bool_t cross = false;

		cross = Cross();
		eg = EGamma();
		muons = Muons();
		jets = Jets() ;
		sums = Sums();


		Bool_t pass  = jets || eg || sums || muons || cross  ;

		if (pass) NPASS ++;

		if (cross) NCROSS ++;
		if (muons) NMUONS ++;
		if (sums) NSUMS ++;
		if (eg) NEG ++;
		if (jets) NJETS ++;

		if (pass) h_Block -> Fill(5.);

		Bool_t dec[5];
		dec[0] = eg;
		dec[1] = jets;
		dec[2] = muons;
		dec[3] = sums;
		dec[4] = cross;
		for (Int_t l=0; l < 5; l++) {
			if (dec[l]) {
				h_Block -> Fill(l);
				for (Int_t k=0; k < 5; k++) {
					if (dec[k]) cor_Block -> Fill(l,k);
				}
			}

		}

		first = false;

	// -- now the pure rate stuff
	// -- kOFFSET now contains the number of triggers we have calculated

		Bool_t ddd[NPAGS];
		for (Int_t idd=0; idd < NPAGS; idd++) {
			ddd[idd] = false; 
		} 

		Float_t weightEvent = 1.;

		for (Int_t k=0; k < kOFFSET; k++) {
			if ( ! TheTriggerBits[k] ) continue;
			h_All -> Fill(k);

			TString name = h_All -> GetXaxis() -> GetBinLabel(k+1);
			std::string L1namest = (std::string)name;
			Bool_t IsTOP = setTOP.count(L1namest) > 0;
			Bool_t IsHIGGS = setHIGGS.count(L1namest) > 0;
			Bool_t IsBPH = setBPH.count(L1namest) > 0;
			Bool_t IsEXO = setEXO.count(L1namest) > 0;
			Bool_t IsSUSY = setSUSY.count(L1namest) > 0;
			Bool_t IsSMP = setSMP.count(L1namest) > 0;
			if (IsHIGGS) ddd[0] = true;
			if (IsSUSY) ddd[1] = true;
			if (IsEXO) ddd[2] = true;
			if (IsTOP) ddd[3] = true;
			if (IsSMP) ddd[4] = true;
			if (IsBPH) ddd[5] = true;

			Float_t ww = WeightsPAGs[L1namest];
			if (ww < weightEvent) weightEvent = ww;

		// did the event pass another trigger ?
			Bool_t nonpure = false;
			for (Int_t k2=0; k2 < kOFFSET; k2++) {
				if (k2 == k) continue;
				if ( TheTriggerBits[k2] ) nonpure = true;
			}
			Bool_t pure = !nonpure ;
			if (pure) h_Pure -> Fill(k);
		}

	// -- for the PAG rates :
		Bool_t PAG = false;
		for (Int_t idd=0; idd < NPAGS; idd++) {
			if (ddd[idd]) {
				Bool_t nonpure = false;
				PAG = true;
				for (Int_t jdd=0; jdd < NPAGS; jdd++) {
					if (ddd[jdd]) {
						cor_PAGS -> Fill(idd,jdd);
						if (jdd != idd) nonpure = true;
					}
				}   
				Bool_t pure = ! nonpure;
				if (pure) h_PAGS_pure -> Fill(idd);
				h_PAGS_shared -> Fill(idd,weightEvent);

			}  
		}
		if (PAG) nPAG ++;


	}  // end evt loop

	Float_t scal = 1./(23.3) ;       				// 1 LS = 23.3 sec
	scal = scal / theNumberOfUserdLumiSections ;
	scal = scal * 10.;      					// nanoDST is p'ed by 10
	scal = scal / 1000.  ;    					// rate in kHz

	scal = scal * theTargetLumi / theLumiForThisSetOfLumiSections   ;	// scale up from LumiForThisSetOfLumiSections to theTargetLumi

	scal = scal * theZeroBiasPrescale;									// because ZeroBias was pre-scaled

//	Float_t scalefor8TeV = 1.2;
    Float_t scalefor8TeV = 1.32;

//	Float_t extrarate = 3;
	Float_t extrarate = 5;
	
	h_Cross_8TeV = (TH1F*)h_Cross->Clone("h_Cross_8TeV");
	h_Cross_8TeV -> Scale(scal * scalefor8TeV);
	h_Cross -> Scale(scal);
	
	h_Jets_8TeV = (TH1F*)h_Jets->Clone("h_Jets_8TeV");
	h_Jets_8TeV -> Scale(scal * scalefor8TeV);
	h_Jets -> Scale(scal);
	
	h_Egamma_8TeV = (TH1F*)h_Egamma->Clone("h_Egamma_8TeV");
	h_Egamma_8TeV -> Scale(scal * scalefor8TeV);
	h_Egamma -> Scale(scal);
	
	h_Sums_8TeV = (TH1F*)h_Sums->Clone("h_Sums_8TeV");
	h_Sums_8TeV -> Scale(scal * scalefor8TeV);
	h_Sums -> Scale(scal);
	
	h_Muons_8TeV = (TH1F*)h_Muons->Clone("h_Muons_8TeV");
	h_Muons_8TeV -> Scale(scal * scalefor8TeV);
	h_Muons -> Scale(scal);

	h_All_8TeV = (TH1F*)h_All->Clone("h_All_8TeV");
	CorrectScale(h_All_8TeV, scal * scalefor8TeV);
	CorrectScale(h_All, scal);
	
	h_Pure_8TeV = (TH1F*)h_Pure->Clone("h_Pure_8TeV");
	h_Pure_8TeV  -> Scale(scal * scalefor8TeV);
	h_Pure  -> Scale(scal);

	cout << endl << " --------------------------------------------------------- " << endl << endl;
	cout << " Prescales for: " << theTargetLumi << ", LumiForThisSetOfLumiSections = " << theLumiForThisSetOfLumiSections << ", L1NtupleFileName = " << theL1NtupleFileName << endl;
	cout << endl << " --------------------------------------------------------- " << endl << endl;
	cout << " Rate that pass L1 " << NPASS * scal << " kHz  ( claimed by a PAG " << nPAG * scal << " kHz  i.e. " << 100.*(float)nPAG/(float)NPASS << "%. ) " << "scaled to 8 TeV (" << scalefor8TeV << ") and adding " << extrarate << " kHz = " << NPASS * scal * scalefor8TeV + extrarate << " kHz " << endl << endl;
	cout << " --------------------------------------------------------- " << endl;
	cout << " Rate that pass L1 jets: " << NJETS * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NJETS * scal * scalefor8TeV << " kHz"<< endl;
	cout << " Rate that pass L1 EG: " << NEG * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NEG * scal * scalefor8TeV << " kHz"<< endl;
	cout << " Rate that pass L1 Sums: " << NSUMS * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NSUMS * scal * scalefor8TeV << " kHz"<< endl;
	cout << " Rate that pass L1 Muons: " << NMUONS * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NMUONS * scal * scalefor8TeV << " kHz"<< endl;
	cout << " Rate that pass L1 Cross: " << NCROSS * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NCROSS * scal * scalefor8TeV << " kHz"<< endl;

	output << " Prescales for: " << theTargetLumi << ", LumiForThisSetOfLumiSections = " << theLumiForThisSetOfLumiSections << ", L1NtupleFileName = " << theL1NtupleFileName << endl;
	output << endl << " --------------------------------------------------------- " << endl << endl;
	output << " Rate that pass L1 " << NPASS * scal << " kHz  ( claimed by a PAG " << nPAG * scal << " kHz  i.e. " << 100.*(float)nPAG/(float)NPASS << "%. ) " << "scaled to 8 TeV (" << scalefor8TeV << ") and adding " << extrarate << " kHz = " << NPASS * scal * scalefor8TeV + extrarate << " kHz " << endl << endl;
	output << " --------------------------------------------------------- " << endl;
	output << " Rate that pass L1 jets: " << NJETS * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NJETS * scal * scalefor8TeV << " kHz"<< endl;
	output << " Rate that pass L1 EG: " << NEG * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NEG * scal * scalefor8TeV << " kHz"<< endl;
	output << " Rate that pass L1 Sums: " << NSUMS * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NSUMS * scal * scalefor8TeV << " kHz"<< endl;
	output << " Rate that pass L1 Muons: " << NMUONS * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NMUONS * scal * scalefor8TeV << " kHz"<< endl;
	output << " Rate that pass L1 Cross: " << NCROSS * scal << " kHz" << ", scaled to 8 TeV (" << scalefor8TeV << ") = " << NCROSS * scal * scalefor8TeV << " kHz"<< endl;

	for (Int_t i=1; i<= 5; i++) {
		Float_t nev = h_Block -> GetBinContent(i);
		for (Int_t j=1; j<= 5; j++) {
			Int_t ibin = cor_Block -> FindBin(i-1,j-1);
			Float_t val = cor_Block -> GetBinContent(ibin);
			val = val / nev;
			cor_Block -> SetBinContent(ibin,val);
		}
	}   

	h_Block -> Scale(scal);

	cor_PAGS -> Scale(scal);
	h_PAGS_pure -> Scale(scal);
	h_PAGS_shared -> Scale(scal);

	Int_t NBITS_ALL = NBITS_MUONS + NBITS_EGAMMA + NBITS_JETS + NBITS_SUMS + NBITS_CROSS;

	cout << endl << " --- TOTAL NUMBER OF BITS: " << endl;
	cout << "  USED : " << NBITS_ALL << endl;
	cout << "  MUONS : " << NBITS_MUONS << endl;
	cout << "  EGAMMA : " << NBITS_EGAMMA << endl;
	cout << "  JETS : " << NBITS_JETS << endl;
	cout << "  SUMS : " << NBITS_SUMS << endl;
	cout << "  CROSS : " << NBITS_CROSS << endl << endl;

	output << endl << " --- TOTAL NUMBER OF BITS: " << endl;
	output << "  USED : " << NBITS_ALL << endl;
	output << "  MUONS : " << NBITS_MUONS << endl;
	output << "  EGAMMA : " << NBITS_EGAMMA << endl;
	output << "  JETS : " << NBITS_JETS << endl;
	output << "  SUMS : " << NBITS_SUMS << endl;
	output << "  CROSS : " << NBITS_CROSS << endl << endl;
}

void RunL1(Bool_t drawplots=true,Bool_t writefiles=true,Float_t targetlumi=50,Int_t whichFileAndLumiToUse=1) {

	Int_t Nbin_max = 50;
	h_Cross = new TH1F("h_Cross","h_Cross",Nbin_max,-0.5,(float)Nbin_max-0.5);
	h_Sums = new TH1F("h_Sums","h_Sums",Nbin_max,-0.5,(float)Nbin_max-0.5);
	h_Jets = new TH1F("h_Jets","h_Jets",Nbin_max,-0.5,(float)Nbin_max-0.5);
	h_Egamma = new TH1F("h_Egamma","h_Egamma",Nbin_max,-0.5,(float)Nbin_max-0.5);
	h_Muons = new TH1F("h_Muons","h_Muons",Nbin_max,-0.5,(float)Nbin_max-0.5);

	h_Block = new TH1F("h_Block","h_Block",6,-0.5,5.5);
	cor_Block = new TH2F("cor_Block","cor_Block",5,-0.5,4.5,5,-0.5,4.5);

	cor_PAGS = new TH2F("cor_PAGS","cor_PAGS",NPAGS,-0.5,(float)NPAGS-0.5,NPAGS,-0.5,(float)NPAGS-0.5);
	h_PAGS_pure = new TH1F("h_PAGS_pure","h_PAGS_pure",NPAGS,-0.5,(float)NPAGS-0.5);
	h_PAGS_shared = new TH1F("h_PAGS_shared","h_PAGS_shared",NPAGS,-0.5,(float)NPAGS-0.5);

	h_All = new TH1F("h_All","h_All",N128,-0.5,N128-0.5);
	h_Pure = new TH1F("h_Pure","h_Pure",N128,-0.5,N128-0.5);

	// Using the 10-bunches High PU run, 179828. In this run ETM30 and HTT50 were enabled (they were not enabled in the 1-bunch run).
	Float_t NumberOfUserdLumiSections=0; 
	Float_t LumiForThisSetOfLumiSections=0;
	string L1NtupleFileName="";
	Float_t AveragePU=0;
	Float_t ZeroBiasPrescale=0;
	Bool_t L1JetCorrection=false;

	if (whichFileAndLumiToUse==1) {
	// -- Run 179828, LS 374 - 394, PU=28:
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 0.437;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1TreeL1Accept_179828_LS_374_394.root";
		AveragePU = 28;
		ZeroBiasPrescale = 3;
		L1JetCorrection=false;
	}
	else if (whichFileAndLumiToUse==2) {

	// -- Run 179828, LS 300 - 320, PU=29:
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 0.463;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1TreeL1Accept_179828_LS_300_320.root";
		AveragePU = 29;
		ZeroBiasPrescale = 3;
		L1JetCorrection=false;
	}
	else if (whichFileAndLumiToUse==3) {

	// -- Run 179828, LS 270 - 290, PU=30:
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 0.470;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1TreeL1Accept_179828_LS_270_290.root";
		AveragePU = 30;
		ZeroBiasPrescale = 3;
		L1JetCorrection=false;
	}
	else if (whichFileAndLumiToUse==4) {

	// -- Run 179828, LS 140 - 160, PU=33:
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 0.509;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1TreeL1Accept_179828_LS_140_160.root";
		AveragePU = 33;
		ZeroBiasPrescale = 3;
		L1JetCorrection=false;
	}
	else if (whichFileAndLumiToUse==5) {

	// -- Run 179828, LS 50 - 70, PU=34:
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 0.529;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1TreeL1Accept_179828_LS_50_70.root";
		AveragePU = 34;
		ZeroBiasPrescale = 3;
		L1JetCorrection=false;
	}
	else if (whichFileAndLumiToUse==6) {

	// -- Run 178803, LS 400 - 420, PU=18, (with bunch trains i.e. possible OOT PU): 
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 0.131;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1TreeL1Accept_178803_LS_400_420.root";
		AveragePU = 18;
		ZeroBiasPrescale = 29483;
		L1JetCorrection=false;
	}
	else {
		cout << endl << "ERROR: Please define a ntuple file which is in the allowed range! You did use: whichFileAndLumiToUse = " << whichFileAndLumiToUse << " This is not in the allowed range" << endl << endl;
	}

	ostringstream txtos;
	txtos << targetlumi << "_" << AveragePU << "_" << LumiForThisSetOfLumiSections << "_rates.txt";
	TString TXTOutPutFileName = txtos.str();
	ofstream TXTOutfile(TXTOutPutFileName);

	ostringstream csvos;
	csvos << targetlumi << "_" << AveragePU << "_" << LumiForThisSetOfLumiSections << "_rates.csv";
	TString CSVOutPutFileName = csvos.str();
	ofstream CSVOutfile(CSVOutPutFileName);

	cout << endl << "Target Luminosity = " << targetlumi << endl << endl;

	cout << endl << "Using: whichFileAndLumiToUse = " << whichFileAndLumiToUse << endl;
	cout << "  NumberOfUserdLumiSections        = " << NumberOfUserdLumiSections << endl;
	cout << "  LumiForThisSetOfLumiSections     = " << LumiForThisSetOfLumiSections << endl;
	cout << "  L1NtupleFileName                 = " << L1NtupleFileName << endl;
	cout << "  AveragePU                        = " << AveragePU << endl;
	cout << "  L1JetCorrections (for 2011 data) = " << L1JetCorrection << endl << endl;

	if (writefiles) { 
		cout << endl << "Writing CSV and txt files as well ..." << endl;
		cout << "  TXTOutPutFileName: " << TXTOutPutFileName << endl;
		cout << "  CVSOutPutFileName: " << CSVOutPutFileName << endl << endl;

		TXTOutfile << endl << "Target Luminosity = " << targetlumi << endl << endl;

		TXTOutfile << endl << "Using: whichFileAndLumiToUse = " << whichFileAndLumiToUse << endl;
		TXTOutfile << "  NumberOfUserdLumiSections    = " << NumberOfUserdLumiSections << endl;
		TXTOutfile << "  LumiForThisSetOfLumiSections = " << LumiForThisSetOfLumiSections << endl;
		TXTOutfile << "  L1NtupleFileName             = " << L1NtupleFileName << endl;
		TXTOutfile << "  AveragePU                        = " << AveragePU << endl;
		TXTOutfile << "  L1JetCorrections (for 2011 data) = " << L1JetCorrection << endl << endl;
	}

	L1Menu2012 a(targetlumi,NumberOfUserdLumiSections,LumiForThisSetOfLumiSections,L1NtupleFileName,AveragePU,ZeroBiasPrescale,L1JetCorrection);
	a.Open(L1NtupleFileName);
	a.Loop();

	if (drawplots) {

		TString YaxisName;
		if (targetlumi == 1.) YaxisName = "Rate (kHz) at 1e32 (PU = t.b.d)";
		if (targetlumi == 2.) YaxisName = "Rate (kHz) at 2e32 (PU = t.b.d)";
		if (targetlumi == 20.) YaxisName = "Rate (kHz) at 2e33 (PU = t.b.d.)";
		if (targetlumi == 50) YaxisName = "Rate (kHz) at 5e33 (PU = 28)";
		if (targetlumi == 50.001) YaxisName = "Rate (kHz) at 5e33 (PU = 28)";
		if (targetlumi == 60.) YaxisName = "Rate (kHz) at 6e33 (PU = 30)";
		if (targetlumi == 70.) YaxisName = "Rate (kHz) at 7e33 (PU = 33)";
		if (targetlumi == 70.001) YaxisName = "Rate (kHz) at 7e33 (PU = 33)";

		TCanvas* c1 = new TCanvas("c1","c1");
		c1 -> cd();
		gStyle -> SetOptStat(0);
		h_Cross -> SetLineColor(4);
		h_Cross -> GetXaxis() -> SetLabelSize(0.035);
		h_Cross -> SetYTitle(YaxisName);
		h_Cross -> Draw();

		TCanvas* c2 = new TCanvas("c2","c2");
		c2 -> cd();
		h_Sums -> SetLineColor(4);
		h_Sums -> GetXaxis() -> SetLabelSize(0.035);
		h_Sums -> SetYTitle(YaxisName);
		h_Sums -> Draw();

		TCanvas* c3 = new TCanvas("c3","c3");
		c3 -> cd();
		h_Egamma -> SetLineColor(4);
		h_Egamma -> GetXaxis() -> SetLabelSize(0.035);
		h_Egamma -> SetYTitle(YaxisName);
		h_Egamma -> Draw();

		TCanvas* c4 = new TCanvas("c4","c4");
		c4 -> cd();
		h_Jets -> SetLineColor(4);
		h_Jets -> GetXaxis() -> SetLabelSize(0.035);
		h_Jets -> SetYTitle(YaxisName);
		h_Jets -> Draw();

		TCanvas* c5 = new TCanvas("c5","c5");
		c5 -> cd();
		h_Muons -> SetLineColor(4);
		h_Muons -> GetXaxis() -> SetLabelSize(0.035);
		h_Muons -> SetYTitle(YaxisName);
		h_Muons -> Draw();


		TCanvas* c6 = new TCanvas("c6","c6");
		c6 -> cd();
		cor_Block -> GetXaxis() -> SetBinLabel(1,"EG");
		cor_Block -> GetXaxis() -> SetBinLabel(2,"Jets");
		cor_Block -> GetXaxis() -> SetBinLabel(3,"Muons");
		cor_Block -> GetXaxis() -> SetBinLabel(4,"Sums");
		cor_Block -> GetXaxis() -> SetBinLabel(5,"Cross");

		cor_Block -> GetYaxis() -> SetBinLabel(1,"EG");
		cor_Block -> GetYaxis() -> SetBinLabel(2,"Jets");
		cor_Block -> GetYaxis() -> SetBinLabel(3,"Muons");
		cor_Block -> GetYaxis() -> SetBinLabel(4,"Sums");
		cor_Block -> GetYaxis() -> SetBinLabel(5,"Cross");

		cor_Block -> Draw("colz");
		cor_Block -> Draw("same,text");

		TCanvas* c7 = new TCanvas("c7","c7");
		c7 -> cd();

		cor_PAGS -> GetXaxis() -> SetBinLabel(1,"HIGGS");
		cor_PAGS -> GetXaxis() -> SetBinLabel(2,"SUSY");
		cor_PAGS -> GetXaxis() -> SetBinLabel(3,"EXO");
		cor_PAGS -> GetXaxis() -> SetBinLabel(4,"TOP");
		cor_PAGS -> GetXaxis() -> SetBinLabel(5,"SMP");
		cor_PAGS -> GetXaxis() -> SetBinLabel(6,"BPH");

		cor_PAGS -> GetYaxis() -> SetBinLabel(1,"HIGGS");
		cor_PAGS -> GetYaxis() -> SetBinLabel(2,"SUSY");
		cor_PAGS -> GetYaxis() -> SetBinLabel(3,"EXO");
		cor_PAGS -> GetYaxis() -> SetBinLabel(4,"TOP");
		cor_PAGS -> GetYaxis() -> SetBinLabel(5,"SMP");
		cor_PAGS -> GetYaxis() -> SetBinLabel(6,"BPH");

		cor_PAGS -> Draw("colz");
		cor_PAGS -> Draw("same,text"); 

		TCanvas* c8 = new TCanvas("c8","c8");
		c8 -> cd();
		h_PAGS_pure -> GetXaxis() -> SetBinLabel(1,"HIGGS");
		h_PAGS_pure -> GetXaxis() -> SetBinLabel(2,"SUSY");
		h_PAGS_pure -> GetXaxis() -> SetBinLabel(3,"EXO");
		h_PAGS_pure -> GetXaxis() -> SetBinLabel(4,"TOP");
		h_PAGS_pure -> GetXaxis() -> SetBinLabel(5,"SMP");
		h_PAGS_pure -> GetXaxis() -> SetBinLabel(6,"BPH");
		h_PAGS_pure -> SetYTitle("Pure rate (kHz)");
		h_PAGS_pure -> Draw();

		TCanvas* c9 = new TCanvas("c9","c9");
		c9 -> cd();
		h_PAGS_shared -> GetXaxis() -> SetBinLabel(1,"HIGGS");
		h_PAGS_shared -> GetXaxis() -> SetBinLabel(2,"SUSY");
		h_PAGS_shared -> GetXaxis() -> SetBinLabel(3,"EXO");
		h_PAGS_shared -> GetXaxis() -> SetBinLabel(4,"TOP");
		h_PAGS_shared -> GetXaxis() -> SetBinLabel(5,"SMP");
		h_PAGS_shared -> GetXaxis() -> SetBinLabel(6,"BPH");
		h_PAGS_shared -> SetYTitle("Shared rate (kHz)");
		h_PAGS_shared -> Draw();

	}
	
	cout << "L1Bit" << "\t" << "L1SeedName" << "\t" << "pre-scale" << "\t" << "rate@7TeV" << "\t +/- \t" << "error_rate@7TeV" << "\t" << "pure@7TeV" << "\t | \t" << "rate@8TeV" << "\t +/- \t" << "error_rate@8TeV" << "\t" << "pure@8TeV"  << endl;
	TXTOutfile << "L1Bit" << "\t" << "L1SeedName" << "\t" << "pre-scale" << "\t" << "rate@7TeV" << "\t +/- \t" << "error_rate@7TeV" << "\t" << "pure@7TeV" << "\t | \t" << "rate@8TeV" << "\t +/- \t" << "error_rate@8TeV" << "\t" << "pure@8TeV" << endl;
	
	if (writefiles) { 
		CSVOutfile << "L1Bit" << ";" << "L1SeedName" << ";" << "AveragePU" << ';' << "pre-scale" << ";" << "rate@7TeV" << ";" << "error_rate@7TeV" << ";" << "pure@7TeV" << ";" << "rate@8TeV" << ";" << "error_rate@8TeV" << ";" << "pure@8TeV" << endl; 
	}

	for (Int_t k=1; k < kOFFSET+1; k++) {  // -- kOFFSET now contains the number of triggers we have calculated
		TString name = h_All -> GetXaxis() -> GetBinLabel(k);

		Float_t rate = h_All -> GetBinContent(k);
		Float_t rate_8TeV = h_All_8TeV -> GetBinContent(k);

		Float_t err_rate  = h_All -> GetBinError(k);
		Float_t err_rate_8TeV  = h_All_8TeV -> GetBinError(k);

		Float_t pure = h_Pure -> GetBinContent(k);
		Float_t pure_8TeV = h_Pure_8TeV -> GetBinContent(k);

		std::string L1namest = (std::string)name;
		map<string, int>::const_iterator it = a.Prescales.find(L1namest);
		Float_t pre;
		if (it == a.Prescales.end() ) {
			cout << " --- SET P = 1 FOR SEED :  " << L1namest << endl;
			if (writefiles) { TXTOutfile << " --- SET P = 1 FOR SEED :  " << L1namest << endl; }
			pre = 1;
		}
		else {
			pre = it -> second;
		}
		Bool_t bias = a.Biased[L1namest];

		if (bias) cout << a.L1Bit(L1namest) << "\t" << name << "\t" << pre << "\t" << rate << "\t +/- \t" << err_rate << "\t" << pure << "\t | \t" << rate_8TeV << "\t +/- \t" << err_rate_8TeV << "\t" << pure_8TeV << "\t" << " ***  BIAS  *** " << endl;
		else
			cout << a.L1Bit(L1namest) << "\t" << name << "\t" << pre << "\t" << rate << "\t +/- \t" << err_rate << "\t" << pure << "\t | \t" << rate_8TeV << "\t +/- \t" << err_rate_8TeV << "\t" << pure_8TeV  << endl;

		// CVS file filling
		if (writefiles) { 
			if (bias) { TXTOutfile << a.L1Bit(L1namest) << "\t" << name << "\t" << pre << "\t" << rate << "\t +/- \t" << err_rate << "\t" << pure << "\t | \t"<< rate_8TeV << "\t +/- \t" << err_rate_8TeV << "\t" << pure_8TeV << "\t" << " ***  BIAS  *** " << endl; }
			else { TXTOutfile << a.L1Bit(L1namest) << "\t" << name << "\t" << pre << "\t" << rate << "\t +/- \t" << err_rate << "\t" << pure << "\t | \t" << rate_8TeV << "\t +/- \t" << err_rate_8TeV << "\t" << pure_8TeV << endl; }

			CSVOutfile << a.L1Bit(L1namest) << ";" << name << ";" << AveragePU << ';' << pre << ";" << rate << ";" << err_rate << ";" << pure << ";" << rate_8TeV << ";" << err_rate_8TeV << ";" << pure_8TeV << endl; 
		}

	}

	if (writefiles) {
		TXTOutfile << endl << a.GetPrintout() << endl;
	}

	TXTOutfile.close();
	CSVOutfile.close();
}


