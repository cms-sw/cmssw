#include "L1Ntuple.h"
#include "hist.C"
#include "Style.C"

#include "TLegend.h"
#include "TMath.h"
#include "TText.h"
#include "TH2.h"
#include "TAxis.h"
#include "string.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <set>

// Run over a 2012 run, taken with the 2012 collision menu.
// Does not emulate the logic of the L1 bits; instead, take the online GT decision. The code calculates what the L1 rate would be for a different set of L1 prescales.

// -- Huge prescale value for seeds "for lower PU"
#define INFTY 10000

TH1F* h_Cross;
TH1F* h_Jets;
TH1F* h_Sums;
TH1F* h_Egamma;
TH1F* h_Muons;

TH1F* h_Block;
TH2F* cor_Block;

Int_t NPAGS = 6;
TH2F* cor_PAGS;
TH1F* h_PAGS_pure;
TH1F* h_PAGS_shared;

// For the pure rates
const Int_t N128 = 128;	// could be > 128 for "test seeds"
Int_t kOFFSET = 0;
Bool_t TheTriggerBits[N128] ;	// contains the emulated triggers for each event
TH1F* h_All;				// one bin for each trigger. Fill bin i if event fires trigger i.
TH1F* h_Pure;				// one bin for each trigger. Fill bin i if event fires trigger i and NO OTHER TRIGGER.

void MyScale(TH1F* h, Float_t scal) {

// -- to set the errors properly

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

class L1Menu2012 : public L1Ntuple {
	public :
 
	L1Menu2012(Float_t aTargetLumi, Float_t aNumberOfUserdLumiSections, Float_t aLumiForThisSetOfLumiSections, std::string aL1NtupleFileName,Float_t aAveragePU, Float_t aZeroBiasPrescale,Bool_t aRunOverZeroBias,Bool_t aDoExactRates) : 
	theTargetLumi(aTargetLumi), 
		theNumberOfUserdLumiSections(aNumberOfUserdLumiSections),
		theLumiForThisSetOfLumiSections(aLumiForThisSetOfLumiSections),
		theL1NtupleFileName(aL1NtupleFileName),
		theAveragePU(aAveragePU),
		theZeroBiasPrescale(aZeroBiasPrescale),
		theRunOverZeroBias(aRunOverZeroBias),
		theDoExactRates(aDoExactRates)
		{}

	~L1Menu2012() {}

	// The luminosity for which we want the rates, in units 1e32 (this sets also the right prescales for the 2 menus and some descriptions correctly).
	// 70. for 7e33, 50 for 5e33, etc. Use 70.001 for the "emergency columns" to the corresponding target luminosity.
	// For the moment we have pre-scales for 5e33,6e33,7e33 plus emergency pre-scales for 5e33 and 7e33
	Float_t theTargetLumi;

	// the setting below are/will be specific for each L1Ntuple file used
	Float_t theNumberOfUserdLumiSections;
	Float_t theLumiForThisSetOfLumiSections;
	std::string theL1NtupleFileName;
	Float_t theAveragePU;
	Float_t theZeroBiasPrescale;
	Bool_t theRunOverZeroBias;
	Bool_t theDoExactRates;

	std::stringstream output;
	std::string GetPrintout() { return output.str(); };

	void MyInit();
	void FillBits();

	std::map<std::string, int> Counts;
	std::map<std::string, int> OldPrescales;
	std::map<std::string, int> NewPrescales;

	std::map<std::string, Float_t> WeightsPAGs;

	void InsertInMenu(std::string L1name);
	Bool_t L1Bit(std::string L1name);
	Int_t  L1BitNumber(std::string L1name);
	Bool_t DoPrescale(Int_t evtn, Int_t pinitial, Int_t pfinal) ;

	Bool_t Cross();
	Bool_t Jets();
	Bool_t EGamma();
	Bool_t Muons();
	Bool_t Sums();

	void Loop();

	private :

	Bool_t PhysicsBits[128];
	Bool_t first;

	Int_t insert_ibin;
	Bool_t insert_val[100];
	std::string insert_names[100];

	Int_t NBITS_MUONS;
	Int_t NBITS_EGAMMA;
	Int_t NBITS_JETS;
	Int_t NBITS_SUMS;
	Int_t NBITS_CROSS;

	std::set<std::string> setTOP;
	std::set<std::string> setHIGGS;
	std::set<std::string> setEXO;
	std::set<std::string> setSMP;
	std::set<std::string> setBPH;
	std::set<std::string> setSUSY;

	std::map<std::string, int> BitMapping;
};

Bool_t L1Menu2012::DoPrescale(Int_t evtn, Int_t pinitial, Int_t pfinal) {

	if (theDoExactRates) {

	// Proper way to do it: new prescales are multiples of old prescales

		Int_t ratio = pfinal / pinitial ; 
		if ( evtn % ratio == 0) return true;
	}

	else {
		
	// Approximation: the new prescales do not have to be multiple of the old ones, but the correlations are not fully correct.

		Int_t evtprime = evtn / pinitial;
		Int_t r = evtprime % pfinal;
		if (r <= (pinitial-1) ) return true;
	}

	return false;
}

Bool_t L1Menu2012::L1Bit(std::string l1name) {

	std::map<std::string, int>::const_iterator it = BitMapping.find(l1name);
	if (it == BitMapping.end() ) {
		std::cout << " Wrong L1 name, not in BitMapping " << l1name << std::endl;
		return false;
	}

	Int_t ibit = BitMapping[l1name];
	Bool_t raw = PhysicsBits[ibit];

	return raw;
}


int L1Menu2012::L1BitNumber(string l1name) {

	map<string, int>::const_iterator it = BitMapping.find(l1name);
	if (it == BitMapping.end() ) {
		std::cout << " Wrong L1 name, not in BitMapping " << l1name << std::endl;
		return -1;
	}

	return BitMapping[l1name];
}

void L1Menu2012::InsertInMenu(std::string L1name) {

	Int_t prescale = 1;

	std::map<std::string, int>::const_iterator it = NewPrescales.find(L1name);
	if (it == NewPrescales.end() ) {
		std::cout << " --- BIT IS NOT EXISTING in NewPrescales !!! " << L1name << std::endl;
	}
	else {
		prescale = NewPrescales[L1name];
	}

	Bool_t pre_prescale = L1Bit(L1name);
	Bool_t post_prescale = false;

	if (pre_prescale) Counts[L1name] ++;
	Int_t n = Counts[L1name];

	Int_t p1 = OldPrescales[L1name];
	Int_t p2 = NewPrescales[L1name] ;

	if ( DoPrescale(n, p1, p2) ) post_prescale = pre_prescale;


	insert_names[insert_ibin] = L1name;
	insert_val[insert_ibin] = post_prescale ;

	insert_ibin ++;

}

void L1Menu2012::FillBits() {

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
	setBPH.insert("L1_DoubleMu3er_HighQ_WdEta22");
	setBPH.insert("L1_DoubleMu_5er_0er_HighQ_WdEta22");
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


// The prescales that were used for the run under test: to be obtained e.g. by dumping & parsing the L1 Run Settings Key

//	for (std::map<std::string, int>::iterator it=BitMapping.begin(); it != BitMapping.end(); it++) {
//		std::string name = it -> first;
//		OldPrescales[name] = 1;	
//	}
	
	// 5e33 column was used	
	OldPrescales["L1_ZeroBias"] = 9973;
	OldPrescales["L1_ZeroBias_Instance1"] = 211;
	OldPrescales["L1_BeamGas_Hf_BptxPlusPostQuiet"] = 1000;
	OldPrescales["L1_BeamGas_Hf_BptxMinusPostQuiet"] = 5000;
	OldPrescales["L1_InterBunch_Bptx"] = 1048573;
	OldPrescales["L1_BeamHalo"] = 1;
	OldPrescales["L1_TripleMu0"] = 1;
	OldPrescales["L1_Mu4_HTT125"] = 1;
	OldPrescales["L1_Mu3p5_EG12"] = 1;
	OldPrescales["L1_Mu12er_ETM20"] = 1;
	OldPrescales["L1_MuOpen_EG12"] = 1;
	OldPrescales["L1_Mu12_EG7"] = 1;
	OldPrescales["L1_SingleJet16"] = 50000;
	OldPrescales["L1_SingleJet36"] = 2400;
	OldPrescales["L1_SingleJet52"] = 200;
	OldPrescales["L1_SingleJet68"] = 50;
	OldPrescales["L1_SingleJet92"] = 10;
	OldPrescales["L1_SingleJet128"] = 1;
	OldPrescales["L1_DoubleEG6_HTT100"] = 1;
	OldPrescales["L1_DoubleEG6_HTT125"] = 1;
	OldPrescales["L1_Mu5_DoubleEG5"] = 1;
	OldPrescales["L1_DoubleMu3p5_EG5"] = 1;
	OldPrescales["L1_DoubleMu5_EG5"] = 1;
	OldPrescales["L1_DoubleMu0er_HighQ"] = 1;
	OldPrescales["L1_Mu5_DoubleEG6"] = 1;
	OldPrescales["L1_DoubleJetC44_ETM30"] = 1;
	OldPrescales["L1_Mu3_JetC16_WdEtaPhi2"] = 80;
	OldPrescales["L1_Mu3_JetC52_WdEtaPhi2"] = 1;
	OldPrescales["L1_SingleEG7"] = 400;
	OldPrescales["L1_SingleIsoEG20er"] = 1;
	OldPrescales["L1_EG22_ForJet24"] = 1;
	OldPrescales["L1_EG22_ForJet32"] = 1;
	OldPrescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 3;
	OldPrescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
	OldPrescales["L1_DoubleTauJet44er"] = 1;
	OldPrescales["L1_DoubleEG_13_7"] = 1;
	OldPrescales["L1_TripleEG_12_7_5"] = 1;
	OldPrescales["L1_HTT125"] = 262139;
	OldPrescales["L1_DoubleJetC52"] = 262139;
	OldPrescales["L1_SingleMu14er"] = 1;
	OldPrescales["L1_SingleIsoEG18er"] = 1;
	OldPrescales["L1_DoubleMu_10_Open"] = 1;
	OldPrescales["L1_DoubleMu_10_3p5"] = 1;
	OldPrescales["L1_ETT80"] = 262139;
	OldPrescales["L1_SingleEG5"] = 3000;
	OldPrescales["L1_SingleEG18er"] = 10;
	OldPrescales["L1_SingleEG22"] = 1;
	OldPrescales["L1_SingleEG12"] = 200;
	OldPrescales["L1_SingleEG24"] = 1;
	OldPrescales["L1_SingleEG20"] = 1;
	OldPrescales["L1_SingleEG30"] = 1;
	OldPrescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
	OldPrescales["L1_SingleMuOpen"] = 3500;
	OldPrescales["L1_SingleMu16"] = 100;
	OldPrescales["L1_SingleMu3"] = 2000;
	OldPrescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
	OldPrescales["L1_SingleMu7"] = 400;
	OldPrescales["L1_SingleMu20er"] = 1;
	OldPrescales["L1_SingleMu12"] = 200;
	OldPrescales["L1_SingleMu20"] = 100;
	OldPrescales["L1_SingleMu25er"] = 1;
	OldPrescales["L1_ETM100"] = 1;
	OldPrescales["L1_ETM36"] = 1;
	OldPrescales["L1_ETM30"] = 100;
	OldPrescales["L1_ETM50"] = 1;
	OldPrescales["L1_ETM70"] = 1;
	OldPrescales["L1_ETT300"] = 1;
	OldPrescales["L1_HTT100"] = 262139;
	OldPrescales["L1_HTT150"] = 1;
	OldPrescales["L1_HTT175"] = 1;
	OldPrescales["L1_HTT200"] = 1;
	OldPrescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 1;
	OldPrescales["L1_Mu10er_JetC32"] = 1;
	OldPrescales["L1_DoubleJetC64"] = 1;
	OldPrescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;
	OldPrescales["L1_SingleJetC32_NotBptxOR"] = 1;
	OldPrescales["L1_ETM40"] = 1;
	OldPrescales["L1_Mu0_HTT50"] = 262139;
	OldPrescales["L1_Mu0_HTT100"] = 1;
	OldPrescales["L1_DoubleEG5"] = 262139;
	OldPrescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
	OldPrescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
	OldPrescales["L1_SingleMu16er"] = 1;
	OldPrescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 1;
	OldPrescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
	OldPrescales["L1_SingleMu6_NotBptxOR"] = 1;
	OldPrescales["L1_Mu8_DoubleJetC20"] = 100;
	OldPrescales["L1_DoubleMu0"] = 150;
	OldPrescales["L1_EG8_DoubleJetC20"] = 650;
	OldPrescales["L1_DoubleMu5"] = 50;
	OldPrescales["L1_DoubleJetC56"] = 1;
	OldPrescales["L1_TripleMu0_HighQ"] = 1;
	OldPrescales["L1_TripleMu_5_5_0"] = 1;
	OldPrescales["L1_ETT140"] = 262139;
	OldPrescales["L1_DoubleJetC36"] = 160;
	OldPrescales["L1_DoubleJetC36_ETM30"] = 1;
	OldPrescales["L1_SingleJet36_FwdVeto5"] = 262139;
	OldPrescales["L1_TripleJet_64_44_24_VBF"] = 1;
	OldPrescales["L1_TripleJet_64_48_28_VBF"] = 1;
	OldPrescales["L1_TripleJet_68_48_32_VBF"] = 1;
	OldPrescales["L1_QuadJetC40"] = 1;
	OldPrescales["L1_QuadJetC36"] = 1;
	OldPrescales["L1_TripleJetC_52_28_28"] = 40;
	OldPrescales["L1_QuadJetC32"] = 262139;
	OldPrescales["L1_DoubleForJet16_EtaOpp"] = 262139;
	OldPrescales["L1_DoubleEG3_FwdVeto"] = 262139;
	OldPrescales["L1_SingleJet20_Central_NotBptxOR"] = 10;
	OldPrescales["L1_SingleJet16_FwdVeto5"] = 262139;
	OldPrescales["L1_SingleForJet16"] = 262139;
	OldPrescales["L1_DoubleJetC36_RomanPotsOR"] = 1;
	OldPrescales["L1_SingleMu20_RomanPotsOR"] = 1;
	OldPrescales["L1_SingleEG20_RomanPotsOR"] = 1;
	OldPrescales["L1_DoubleMu5_RomanPotsOR"] = 1;
	OldPrescales["L1_DoubleEG5_RomanPotsOR"] = 1;
	OldPrescales["L1_SingleJet52_RomanPotsOR"] = 1;
	OldPrescales["L1_SingleMu18er"] = 1;
	OldPrescales["L1_MuOpen_EG5"] = 262139;
	OldPrescales["L1_DoubleMu_12_5"] = 1;
	OldPrescales["L1_TripleEG7"] = 1;
	
// enter the new prescales below 

//	for (std::map<std::string, int>::iterator it=BitMapping.begin(); it != BitMapping.end(); it++) {
//		std::string name = it -> first;
//		NewPrescales[name] = OldPrescales[name] ;
//	}
	
	// current 7e33 proposal
	NewPrescales["L1_ZeroBias"] = 9973;
	NewPrescales["L1_ZeroBias_Instance1"] = 211;
	NewPrescales["L1_BeamGas_Hf_BptxPlusPostQuiet"] = 1000;
	NewPrescales["L1_BeamGas_Hf_BptxMinusPostQuiet"] = 5000;
	NewPrescales["L1_InterBunch_Bptx"] = 1048573;
	NewPrescales["L1_BeamHalo"] = 1;
	NewPrescales["L1_TripleMu0"] = 1;
	NewPrescales["L1_Mu4_HTT125"] = 1;
	NewPrescales["L1_Mu3p5_EG12"] = 1;
	NewPrescales["L1_Mu12er_ETM20"] = 1;
	NewPrescales["L1_MuOpen_EG12"] = 262139;
	NewPrescales["L1_Mu12_EG7"] = 1;
	NewPrescales["L1_SingleJet16"] = 50000;
	NewPrescales["L1_SingleJet36"] = 6000;
	NewPrescales["L1_SingleJet52"] = 500;
	NewPrescales["L1_SingleJet68"] = 100;
	NewPrescales["L1_SingleJet92"] = 20;
	NewPrescales["L1_SingleJet128"] = 1;
	NewPrescales["L1_DoubleEG6_HTT100"] = 262139;
	NewPrescales["L1_DoubleEG6_HTT125"] = 1;
	NewPrescales["L1_Mu5_DoubleEG5"] = 262139;
	NewPrescales["L1_DoubleMu3p5_EG5"] = 262139;
	NewPrescales["L1_DoubleMu5_EG5"] = 1;
	NewPrescales["L1_DoubleMu0er_HighQ"] = 1;
	NewPrescales["L1_Mu5_DoubleEG6"] = 1;
	NewPrescales["L1_DoubleJetC44_ETM30"] = 1;
	NewPrescales["L1_Mu3_JetC16_WdEtaPhi2"] = 300;
	NewPrescales["L1_Mu3_JetC52_WdEtaPhi2"] = 10;
	NewPrescales["L1_SingleEG7"] = 800;
	NewPrescales["L1_SingleIsoEG20er"] = 1;
	NewPrescales["L1_EG22_ForJet24"] = 262139;
	NewPrescales["L1_EG22_ForJet32"] = 1;
	NewPrescales["L1_DoubleJetC44_Eta1p74_WdEta4"] = 6;
	NewPrescales["L1_DoubleJetC56_Eta1p74_WdEta4"] = 1;
	NewPrescales["L1_DoubleTauJet44er"] = 1;
	NewPrescales["L1_DoubleEG_13_7"] = 1;
	NewPrescales["L1_TripleEG_12_7_5"] = 1;
	NewPrescales["L1_HTT125"] = 262139;
	NewPrescales["L1_DoubleJetC52"] = 262139;
	NewPrescales["L1_SingleMu14er"] = 1;
	NewPrescales["L1_SingleIsoEG18er"] = 1;
	NewPrescales["L1_DoubleMu_10_Open"] = 1;
	NewPrescales["L1_DoubleMu_10_3p5"] = 1;
	NewPrescales["L1_ETT80"] = 262139;
	NewPrescales["L1_SingleEG5"] = 4500;
	NewPrescales["L1_SingleEG18er"] = 80;
	NewPrescales["L1_SingleEG22"] = 262139;
	NewPrescales["L1_SingleEG12"] = 300;
	NewPrescales["L1_SingleEG24"] = 1;
	NewPrescales["L1_SingleEG20"] = 262139;
	NewPrescales["L1_SingleEG30"] = 1;
	NewPrescales["L1_DoubleMu3er_HighQ_WdEta22"] = 1;
	NewPrescales["L1_SingleMuOpen"] = 7000;
	NewPrescales["L1_SingleMu16"] = 150;
	NewPrescales["L1_SingleMu3"] = 4000;
	NewPrescales["L1_DoubleMu_5er_0er_HighQ_WdEta22"] = 1;
	NewPrescales["L1_SingleMu7"] = 600;
	NewPrescales["L1_SingleMu20er"] = 1;
	NewPrescales["L1_SingleMu12"] = 300;
	NewPrescales["L1_SingleMu20"] = 150;
	NewPrescales["L1_SingleMu25er"] = 1;
	NewPrescales["L1_ETM100"] = 1;
	NewPrescales["L1_ETM36"] = 262139;
	NewPrescales["L1_ETM30"] = 300;
	NewPrescales["L1_ETM50"] = 1;
	NewPrescales["L1_ETM70"] = 1;
	NewPrescales["L1_ETT300"] = 1;
	NewPrescales["L1_HTT100"] = 262139;
	NewPrescales["L1_HTT150"] = 262139;
	NewPrescales["L1_HTT175"] = 1;
	NewPrescales["L1_HTT200"] = 1;
	NewPrescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12"] = 40;
	NewPrescales["L1_Mu10er_JetC32"] = 262139;
	NewPrescales["L1_DoubleJetC64"] = 1;
	NewPrescales["L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12"] = 1;
	NewPrescales["L1_SingleJetC32_NotBptxOR"] = 1;
	NewPrescales["L1_ETM40"] = 1;
	NewPrescales["L1_Mu0_HTT50"] = 262139;
	NewPrescales["L1_Mu0_HTT100"] = 262139;
	NewPrescales["L1_DoubleEG5"] = 262139;
	NewPrescales["L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1"] = 1;
	NewPrescales["L1_EG18er_JetC_Cen36_Tau28_dPhi1"] = 262139;
	NewPrescales["L1_SingleMu16er"] = 1;
	NewPrescales["L1_EG18er_JetC_Cen28_Tau20_dPhi1"] = 262139;
	NewPrescales["L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1"] = 1;
	NewPrescales["L1_SingleMu6_NotBptxOR"] = 1;
	NewPrescales["L1_Mu8_DoubleJetC20"] = 200;
	NewPrescales["L1_DoubleMu0"] = 300;
	NewPrescales["L1_EG8_DoubleJetC20"] = 1000;
	NewPrescales["L1_DoubleMu5"] = 50;
	NewPrescales["L1_DoubleJetC56"] = 262139;
	NewPrescales["L1_TripleMu0_HighQ"] = 1;
	NewPrescales["L1_TripleMu_5_5_0"] = 1;
	NewPrescales["L1_ETT140"] = 262139;
	NewPrescales["L1_DoubleJetC36"] = 320;
	NewPrescales["L1_DoubleJetC36_ETM30"] = 262139;
	NewPrescales["L1_SingleJet36_FwdVeto5"] = 262139;
	NewPrescales["L1_TripleJet_64_44_24_VBF"] = 262139;
	NewPrescales["L1_TripleJet_64_48_28_VBF"] = 1;
	NewPrescales["L1_TripleJet_68_48_32_VBF"] = 1;
	NewPrescales["L1_QuadJetC40"] = 1;
	NewPrescales["L1_QuadJetC36"] = 262139;
	NewPrescales["L1_TripleJetC_52_28_28"] = 100;
	NewPrescales["L1_QuadJetC32"] = 262139;
	NewPrescales["L1_DoubleForJet16_EtaOpp"] = 262139;
	NewPrescales["L1_DoubleEG3_FwdVeto"] = 262139;
	NewPrescales["L1_SingleJet20_Central_NotBptxOR"] = 10;
	NewPrescales["L1_SingleJet16_FwdVeto5"] = 262139;
	NewPrescales["L1_SingleForJet16"] = 262139;
	NewPrescales["L1_DoubleJetC36_RomanPotsOR"] = 1;
	NewPrescales["L1_SingleMu20_RomanPotsOR"] = 1;
	NewPrescales["L1_SingleEG20_RomanPotsOR"] = 1;
	NewPrescales["L1_DoubleMu5_RomanPotsOR"] = 1;
	NewPrescales["L1_DoubleEG5_RomanPotsOR"] = 1;
	NewPrescales["L1_SingleJet52_RomanPotsOR"] = 1;
	NewPrescales["L1_SingleMu18er"] = 1;
	NewPrescales["L1_MuOpen_EG5"] = 262139;
	NewPrescales["L1_DoubleMu_12_5"] = 1;
	NewPrescales["L1_TripleEG7"] = 1;
	
// --- Check the ratio new / old prescale
	if (theDoExactRates) {

		for (std::map<std::string, int>::iterator it=BitMapping.begin(); it != BitMapping.end(); it++) {
			std::string name = it -> first;
			Int_t residual = NewPrescales[name] % OldPrescales[name] ;
			if (residual != 0) {
				std::cout << " !!!!!   NEW prescale should be a multiple of OLD.  Bit name : " << name << std::endl;
			}
		}

	}


// -- Each seed gets a "weight" according to how many PAGS are using it

	for (std::map<std::string, int>::iterator it=NewPrescales.begin(); it != NewPrescales.end(); it++) {
		std::string name = it -> first;
		Int_t UsedPernPAG = 0;
		if ( setTOP.count(name) > 0) UsedPernPAG ++;
		if ( setHIGGS.count(name) > 0) UsedPernPAG ++;
		if ( setSUSY.count(name) > 0) UsedPernPAG ++;
		if ( setEXO.count(name) > 0) UsedPernPAG ++;
		if ( setSMP.count(name) > 0) UsedPernPAG ++;
		if ( setBPH.count(name) > 0) UsedPernPAG ++;
		WeightsPAGs[name] = 1./(Float_t)UsedPernPAG;
	}


	for (std::map<std::string, int>::iterator it=NewPrescales.begin(); it != NewPrescales.end(); it++) {
		std::string name = it -> first;
		Counts[name] = 0;
	}



}

Bool_t L1Menu2012::Muons() {

	insert_ibin = 0;
	InsertInMenu("L1_SingleMuOpen");
	InsertInMenu("L1_SingleMu3");
	InsertInMenu("L1_SingleMu7");
	InsertInMenu("L1_SingleMu12");
	InsertInMenu("L1_SingleMu16");
	InsertInMenu("L1_SingleMu20");

	InsertInMenu("L1_SingleMu14er");
	InsertInMenu("L1_SingleMu16er");
	InsertInMenu("L1_SingleMu18er");
	InsertInMenu("L1_SingleMu20er");
	InsertInMenu("L1_SingleMu25er");

	InsertInMenu("L1_DoubleMu0");
	InsertInMenu("L1_DoubleMu0er_HighQ");
	InsertInMenu("L1_DoubleMu3er_HighQ_WdEta22");
	InsertInMenu("L1_DoubleMu_5er_0er_HighQ_WdEta22");

	InsertInMenu("L1_DoubleMu5");
	InsertInMenu("L1_DoubleMu_12_5");
	InsertInMenu("L1_DoubleMu_10_Open");
	InsertInMenu("L1_DoubleMu_10_3p5");

	InsertInMenu("L1_TripleMu0");
	InsertInMenu("L1_TripleMu0_HighQ");
	InsertInMenu("L1_TripleMu_5_5_0");


	Int_t NN = insert_ibin;
	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += insert_ibin;


	if (first) {

		NBITS_MUONS = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = insert_names[ibin];
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

Bool_t L1Menu2012::Cross() {

	insert_ibin = 0;

	InsertInMenu("L1_Mu0_HTT50");
	InsertInMenu("L1_Mu0_HTT100");
	InsertInMenu("L1_Mu4_HTT125");

	InsertInMenu("L1_Mu12er_ETM20");
	InsertInMenu("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_20_12");
	InsertInMenu("L1_Mu10er_JetC32");
	InsertInMenu("L1_Mu10er_JetC12_WdEtaPhi1_DoubleJetC_32_12");

	InsertInMenu("L1_Mu3_JetC16_WdEtaPhi2");
	InsertInMenu("L1_Mu3_JetC52_WdEtaPhi2");
	InsertInMenu("L1_Mu8_DoubleJetC20");

	InsertInMenu("L1_EG22_ForJet24");
	InsertInMenu("L1_EG22_ForJet32");
	InsertInMenu("L1_DoubleEG6_HTT100");
	InsertInMenu("L1_DoubleEG6_HTT125");

	InsertInMenu("L1_EG18er_JetC_Cen28_Tau20_dPhi1");
	InsertInMenu("L1_IsoEG18er_JetC_Cen32_Tau24_dPhi1");
	InsertInMenu("L1_IsoEG18er_JetC_Cen36_Tau28_dPhi1");
	InsertInMenu("L1_EG18er_JetC_Cen36_Tau28_dPhi1");

	InsertInMenu("L1_EG8_DoubleJetC20");

	InsertInMenu("L1_Mu12_EG7");
	InsertInMenu("L1_MuOpen_EG12");
	InsertInMenu("L1_Mu3p5_EG12");

	InsertInMenu("L1_DoubleMu3p5_EG5");
	InsertInMenu("L1_DoubleMu5_EG5");

	InsertInMenu("L1_Mu5_DoubleEG5");
	InsertInMenu("L1_Mu5_DoubleEG6");

	InsertInMenu("L1_DoubleJetC36_ETM30");
	InsertInMenu("L1_DoubleJetC44_ETM30");

	Int_t NN = insert_ibin;
	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += NN;

	if (first) {

		NBITS_CROSS = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = insert_names[ibin];
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

Bool_t L1Menu2012::Jets() {

	insert_ibin = 0;

	InsertInMenu("L1_SingleJet16");
	InsertInMenu("L1_SingleJet36");
	InsertInMenu("L1_SingleJet52");
	InsertInMenu("L1_SingleJet68");
	InsertInMenu("L1_SingleJet92");
	InsertInMenu("L1_SingleJet128");

	InsertInMenu("L1_DoubleJetC36");
	InsertInMenu("L1_DoubleJetC44_Eta1p74_WdEta4");
	InsertInMenu("L1_DoubleJetC52");
	InsertInMenu("L1_DoubleJetC56_Eta1p74_WdEta4");
	InsertInMenu("L1_DoubleJetC56");
	InsertInMenu("L1_DoubleJetC64");

	InsertInMenu("L1_TripleJet_64_44_24_VBF");
	InsertInMenu("L1_TripleJet_64_48_28_VBF");
	InsertInMenu("L1_TripleJet_68_48_32_VBF");
	InsertInMenu("L1_TripleJetC_52_28_28");

	InsertInMenu("L1_QuadJetC32");
	InsertInMenu("L1_QuadJetC36");
	InsertInMenu("L1_QuadJetC40");

	InsertInMenu("L1_DoubleTauJet44er");

	Int_t NN = insert_ibin;

	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += insert_ibin;


	if (first) {

		NBITS_JETS = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = insert_names[ibin];
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

Bool_t L1Menu2012::Sums() {

	insert_ibin = 0;

	InsertInMenu("L1_ETM30");
	InsertInMenu("L1_ETM36");
	InsertInMenu("L1_ETM40");
	InsertInMenu("L1_ETM50");
	InsertInMenu("L1_ETM70");
	InsertInMenu("L1_ETM100");

	InsertInMenu("L1_HTT100");
	InsertInMenu("L1_HTT125");
	InsertInMenu("L1_HTT150");
	InsertInMenu("L1_HTT175");
	InsertInMenu("L1_HTT200");

	InsertInMenu("L1_ETT300");

	Int_t NN = insert_ibin;

	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += insert_ibin;


	if (first) {

		NBITS_SUMS = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = insert_names[ibin]; 
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

Bool_t L1Menu2012::EGamma() {

	insert_ibin = 0;

	InsertInMenu("L1_SingleEG5");
	InsertInMenu("L1_SingleEG7");
	InsertInMenu("L1_SingleEG12");
	InsertInMenu("L1_SingleEG18er");
	InsertInMenu("L1_SingleIsoEG18er");
	InsertInMenu("L1_SingleEG20");
	InsertInMenu("L1_SingleIsoEG20er");
	InsertInMenu("L1_SingleEG22");
	InsertInMenu("L1_SingleEG24");
	InsertInMenu("L1_SingleEG30");

	InsertInMenu("L1_DoubleEG_13_7");

	InsertInMenu("L1_TripleEG7");
	InsertInMenu("L1_TripleEG_12_7_5");

	Int_t NN = insert_ibin;

	Int_t kOFFSET_old = kOFFSET;
	for (Int_t k=0; k < NN; k++) {
		TheTriggerBits[k + kOFFSET_old] = insert_val[k];
	}
	kOFFSET += insert_ibin;


	if (first) {

		NBITS_EGAMMA = NN;

		for (Int_t ibin=0; ibin < insert_ibin; ibin++) {
			TString l1name = insert_names[ibin];
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
	std::cout << " -- Total number of entries : " << nevents << std::endl;

	// nevents = nevents / 10 ;

	Int_t NPASS = 0; 

	Int_t NJETS = 0;
	Int_t NEG = 0;
	Int_t NSUMS =0;
	Int_t NMUONS = 0;
	Int_t NCROSS = 0;

	Int_t nPAG =0;
	first = true;

	Int_t verbose = 10000;
	for (Long64_t i=0; i<nevents; i++)
	{     
		if (i % verbose == 0) {
			std::cout << "  ... iEvent " << i << std::endl;
			verbose = verbose * 10;
		}

	//load the i-th event
		Long64_t ientry = LoadTree(i); if (ientry < 0) break;
		GetEntry(i);

//      Fill the physics bits:
		FillBits();

		if (first) MyInit();

		Bool_t raw = PhysicsBits[0];  // ZeroBias
		Bool_t notraw = ! raw;
		if (theRunOverZeroBias &&  notraw) continue;


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

			std::string name = h_All -> GetXaxis() -> GetBinLabel(k+1);
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

	Float_t scal = 1./(23.3) ;       									// 1 LS = 23.3 sec
	scal = scal / theNumberOfUserdLumiSections ;
	scal = scal * 10.;      											// nanoDST is p'ed by 10
	scal = scal / 1000.  ;    											// rate in kHz

	scal = scal * theTargetLumi / theLumiForThisSetOfLumiSections   ;	// scale up from LumiForThisSetOfLumiSections to theTargetLumi

	if (theRunOverZeroBias) { scal = scal * theZeroBiasPrescale; }		// because ZeroBias was pre-scaled


	std::cout << std::endl;
	std::cout << " --------------------------------------------------------- " << std::endl;
	std::cout << " Rate that pass L1 " << NPASS * scal << " kHz  " << std::endl;
	std::cout << "        ( claimed by a PAG " << nPAG * scal << " kHz  i.e. " << 100.*(Float_t)nPAG/(Float_t)NPASS << "%. ) " << std::endl;
	std::cout << "    scaled to 8 TeV (1.2) and adding 3 kHz " << NPASS * scal * 1.2 + 3.0 << " kHz " << std::endl;

	h_Cross -> Scale(scal);
	h_Jets -> Scale(scal);
	h_Egamma -> Scale(scal);
	h_Sums -> Scale(scal);
	h_Muons -> Scale(scal);

// h_All -> Scale(scal);
	MyScale(h_All, scal);

	h_Pure  -> Scale(scal);

	std::cout << " --------------------------------------------------------- " << std::endl;
	std::cout << " Rate that pass L1 jets " << NJETS * scal << " kHz  " << std::endl;
	std::cout << " Rate that pass L1 EG " << NEG * scal << " kHz  " << std::endl;
	std::cout << " Rate that pass L1 Sums " << NSUMS * scal << " kHz  " << std::endl;
	std::cout << " Rate that pass L1 Muons " << NMUONS * scal << " kHz  " << std::endl;
	std::cout << " Rate that pass L1 Cross " << NCROSS * scal << " kHz  " << std::endl;

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


	std::cout << std::endl;
	Int_t NBITS_ALL = NBITS_MUONS + NBITS_EGAMMA + NBITS_JETS + NBITS_SUMS + NBITS_CROSS;

	std::cout << " --- TOTAL NUMBER OF BITS USED : " << NBITS_ALL << std::endl;
	std::cout << "                          MUONS : " << NBITS_MUONS << std::endl;
	std::cout << "                          EGAMMA : " << NBITS_EGAMMA << std::endl;
	std::cout << "                          JETS : " << NBITS_JETS << std::endl;
	std::cout << "                          SUMS : " << NBITS_SUMS << std::endl;
	std::cout << "                          CROSS : " << NBITS_CROSS << std::endl;


}

void RunL1(Bool_t drawplots=true,Bool_t writefiles=true,Float_t targetlumi=50,Int_t whichFileAndLumiToUse=1) {

	Int_t Nbin_max = 50;
	h_Cross = new TH1F("h_Cross","h_Cross",Nbin_max,-0.5,(Float_t)Nbin_max-0.5);
	h_Sums = new TH1F("h_Sums","h_Sums",Nbin_max,-0.5,(Float_t)Nbin_max-0.5);
	h_Jets = new TH1F("h_Jets","h_Jets",Nbin_max,-0.5,(Float_t)Nbin_max-0.5);
	h_Egamma = new TH1F("h_Egamma","h_Egamma",Nbin_max,-0.5,(Float_t)Nbin_max-0.5);
	h_Muons = new TH1F("h_Muons","h_Muons",Nbin_max,-0.5,(Float_t)Nbin_max-0.5);

	h_Block = new TH1F("h_Block","h_Block",6,-0.5,5.5);
	cor_Block = new TH2F("cor_Block","cor_Block",5,-0.5,4.5,5,-0.5,4.5);

	cor_PAGS = new TH2F("cor_PAGS","cor_PAGS",NPAGS,-0.5,(Float_t)NPAGS-0.5,NPAGS,-0.5,(Float_t)NPAGS-0.5);
	h_PAGS_pure = new TH1F("h_PAGS_pure","h_PAGS_pure",NPAGS,-0.5,(Float_t)NPAGS-0.5);
	h_PAGS_shared = new TH1F("h_PAGS_shared","h_PAGS_shared",NPAGS,-0.5,(Float_t)NPAGS-0.5);

	h_All = new TH1F("h_All","h_All",N128,-0.5,N128-0.5);
	h_Pure = new TH1F("h_Pure","h_Pure",N128,-0.5,N128-0.5);

	// Using the 10-bunches High PU run, 179828. In this run ETM30 and HTT50 were enabled (they were not enabled in the 1-bunch run).
	Float_t NumberOfUserdLumiSections=0; 
	Float_t LumiForThisSetOfLumiSections=0;
	std::string L1NtupleFileName="";
	Float_t AveragePU=0;
	Float_t ZeroBiasPrescale=0;
	Bool_t RunOverZeroBias=false;
	Bool_t DoExactRates=true;	// true means new prescales must be a multiple of the old prescales. Else, new prescales can be anything, but the correlations are not fully correct.

	if (whichFileAndLumiToUse==1) {
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 52.547*0.9;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1Ntuple_191718_40-60_5e33.root";
		AveragePU = 24;
		ZeroBiasPrescale = 3;
		RunOverZeroBias=false;
		DoExactRates=true;
	} else if (whichFileAndLumiToUse==2) {
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 49.2155*0.9;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1Ntuple_191718_113-133_5e33.root";
		AveragePU = 21;
		ZeroBiasPrescale = 3;
		RunOverZeroBias=false;
		DoExactRates=true;
	} else if (whichFileAndLumiToUse==3) {
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 45.646*0.9;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1Ntuple_191718_178-198_5e33.root";
		AveragePU = 19;
		ZeroBiasPrescale = 3;
		RunOverZeroBias=false;
		DoExactRates=true;
	} else if (whichFileAndLumiToUse==4) {
		std::cout << std::endl << "WARNING: This file is probably broken. The rates calculated are too low ..." << std::endl;
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 42.4557*0.9;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1Ntuple_191721_25-45_5e33.root";
		AveragePU = 13;
		ZeroBiasPrescale = 3;
		RunOverZeroBias=false;
		DoExactRates=true;
	} else if (whichFileAndLumiToUse==5) {
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 41.679*0.9;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1Ntuple_191271_187-207_5e33.root";
		AveragePU = 12;
		ZeroBiasPrescale = 3;
		RunOverZeroBias=false;
		DoExactRates=true;
	} else if (whichFileAndLumiToUse==6) {
		NumberOfUserdLumiSections = 21; 
		LumiForThisSetOfLumiSections = 60.5165*0.9;
		L1NtupleFileName = "~heistera/scratch1/L1Ntuples/L1Ntuple_191830_50-70_5e33.root";
		AveragePU = 25.5;
		ZeroBiasPrescale = 3;
		RunOverZeroBias=false;
		DoExactRates=true;
	}
	else {
		std::cout << std::endl << "ERROR: Please define a ntuple file which is in the allowed range! You did use: whichFileAndLumiToUse = " << whichFileAndLumiToUse << " This is not in the allowed range" << std::endl << std::endl;
	}

	std::ostringstream txtos;
	txtos << targetlumi << "_" << AveragePU << "_" << LumiForThisSetOfLumiSections << "_rates.txt";
	TString TXTOutPutFileName = txtos.str();
	std::ofstream TXTOutfile(TXTOutPutFileName);

	std::ostringstream csvos;
	csvos << targetlumi << "_" << AveragePU << "_" << LumiForThisSetOfLumiSections << "_rates.csv";
	TString CSVOutPutFileName = csvos.str();
	std::ofstream CSVOutfile(CSVOutPutFileName);

	std::cout << std::endl << "Target Luminosity = " << targetlumi << std::endl << std::endl;

	std::cout << std::endl << "Using: whichFileAndLumiToUse = " << whichFileAndLumiToUse << std::endl;
	std::cout << "  NumberOfUserdLumiSections        = " << NumberOfUserdLumiSections << std::endl;
	std::cout << "  LumiForThisSetOfLumiSections     = " << LumiForThisSetOfLumiSections << std::endl;
	std::cout << "  L1NtupleFileName                 = " << L1NtupleFileName << std::endl;
	std::cout << "  AveragePU                        = " << AveragePU << std::endl;
	std::cout << "  RunOverZeroBias                  = " << RunOverZeroBias << std::endl << std::endl;

	if (writefiles) { 
		std::cout << std::endl << "Writing CSV and txt files as well ..." << std::endl;
		std::cout << "  TXTOutPutFileName: " << TXTOutPutFileName << std::endl;
		std::cout << "  CVSOutPutFileName: " << CSVOutPutFileName << std::endl << std::endl;

		TXTOutfile << std::endl << "Target Luminosity = " << targetlumi << std::endl << std::endl;

		TXTOutfile << std::endl << "Using: whichFileAndLumiToUse = " << whichFileAndLumiToUse << std::endl;
		TXTOutfile << "  NumberOfUserdLumiSections    = " << NumberOfUserdLumiSections << std::endl;
		TXTOutfile << "  LumiForThisSetOfLumiSections = " << LumiForThisSetOfLumiSections << std::endl;
		TXTOutfile << "  L1NtupleFileName             = " << L1NtupleFileName << std::endl;
		TXTOutfile << "  AveragePU                    = " << AveragePU << std::endl;
		TXTOutfile << "  RunOverZeroBias              = " << RunOverZeroBias << std::endl << std::endl;
	}

	L1Menu2012 a(targetlumi,NumberOfUserdLumiSections,LumiForThisSetOfLumiSections,L1NtupleFileName,AveragePU,ZeroBiasPrescale,RunOverZeroBias,DoExactRates);
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

	std::cout << "L1Bit" << "\t" << "L1SeedName" << "\t" << "pre-scale" << "\t" << "rate@7TeV" << "\t +/- \t" << "error_rate@7TeV" << "\t" << "pure@7TeV" << "\t | \t" << "rate@8TeV" << "\t +/- \t" << "error_rate@8TeV" << "\t" << "pure@8TeV"  << std::endl;
	
	if (writefiles) { 
		CSVOutfile << "L1Bit" << ";" << "L1SeedName" << ";" << "AveragePU" << ';' << "pre-scale" << ";" << "rate@7TeV" << ";" << "error_rate@7TeV" << ";" << "pure@7TeV" << ";" << "rate@8TeV" << ";" << "error_rate@8TeV" << ";" << "pure@8TeV" << std::endl; 
		TXTOutfile << "L1Bit" << "\t" << "L1SeedName" << "\t" << "pre-scale" << "\t" << "rate@7TeV" << "\t +/- \t" << "error_rate@7TeV" << "\t" << "pure@7TeV" << "\t | \t" << "rate@8TeV" << "\t +/- \t" << "error_rate@8TeV" << "\t" << "pure@8TeV" << std::endl;
	}

	for (Int_t k=1; k < kOFFSET+1; k++) {
		std::string name = h_All -> GetXaxis() -> GetBinLabel(k);
		Float_t rate = h_All -> GetBinContent(k);
		Float_t err_rate  = h_All -> GetBinError(k);
		Float_t pure = h_Pure -> GetBinContent(k);
		std::string L1namest = (std::string)name;
		std::map<std::string, int>::const_iterator it = a.NewPrescales.find(L1namest);
		Float_t pre;
		if (it == a.NewPrescales.end() ) {
			std::cout << " --- SET P = 1 FOR SEED :  " << L1namest << std::endl;
			if (writefiles) { TXTOutfile << " --- SET P = 1 FOR SEED :  " << L1namest << std::endl; }
			pre = 1;
		}
		else {
			pre = it -> second;
		}
		std::cout << a.L1BitNumber(L1namest) << "\t" << name << "\t" << pre << "\t" << rate << "\t +/- \t" << err_rate << "\t" << pure << std::endl;

		if (writefiles) { 
			TXTOutfile << a.L1BitNumber(L1namest) << "\t" << name << "\t" << pre << "\t" << rate << "\t +/- \t" << err_rate << "\t" << pure << std::endl;
			CSVOutfile << a.L1BitNumber(L1namest) << ";" << name << ";" << AveragePU << ';' << pre << ";" << rate << ";" << err_rate << ";" << pure << std::endl; 
		}
	}

	if (writefiles) {
		TXTOutfile << std::endl << a.GetPrintout() << std::endl;
	}
	
	TXTOutfile.close();
	CSVOutfile.close();
}