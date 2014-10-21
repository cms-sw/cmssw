#include <TFile.h>
#include "TVector3.h"                   
#include <cmath>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/PhysicsHWW/interface/EGammaMvaEleEstimator.h"
#include "DQM/PhysicsHWW/interface/electronSelections.h"
#include "DQM/PhysicsHWW/interface/trackSelections.h"

using namespace std;
using namespace HWWFunctions;

double electron_d0PV_wwV1_local(HWW& hww, unsigned int index) { 
    if ( hww.vtxs_sumpt().empty() ) return 9999.;
    int iMax = 0; // try the first vertex
	double dxyPV=0;
	if(hww.els_gsftrkidx().at(index)>=0)
	{
    	dxyPV = hww.els_d0().at(index)-
        	hww.vtxs_position().at(iMax).x()*sin(hww.gsftrks_p4().at(hww.els_gsftrkidx().at(index)).phi())+
        	hww.vtxs_position().at(iMax).y()*cos(hww.gsftrks_p4().at(hww.els_gsftrkidx().at(index)).phi());
	}
	else 
	{
    	dxyPV = hww.els_d0().at(index)-
        	hww.vtxs_position().at(iMax).x()*sin(hww.els_trk_p4().at(index).phi())+
        	hww.vtxs_position().at(iMax).y()*cos(hww.els_trk_p4().at(index).phi());
	}

    return dxyPV;
}


//--------------------------------------------------------------------------------------------------
EGammaMvaEleEstimator::EGammaMvaEleEstimator() :
fMethodname("BDTG method"),
fisInitialized(kFALSE),
fPrintMVADebug(kFALSE),
fMVAType(kTrig),
fUseBinnedVersion(kTRUE),
fNMVABins(0)
{
  // Constructor.  
}

//--------------------------------------------------------------------------------------------------
EGammaMvaEleEstimator::~EGammaMvaEleEstimator()
{
  for (unsigned int i=0;i<fTMVAReader.size(); ++i) {
    if (fTMVAReader[i]) delete fTMVAReader[i];
  }
}

//--------------------------------------------------------------------------------------------------
void EGammaMvaEleEstimator::initialize( std::string methodName,
                                       	std::string weightsfile,
                                       	EGammaMvaEleEstimator::MVAType type)
{
  
  std::vector<std::string> tempWeightFileVector;
  tempWeightFileVector.push_back(weightsfile);
  initialize(methodName,type,kFALSE,tempWeightFileVector);
}


//--------------------------------------------------------------------------------------------------
void EGammaMvaEleEstimator::initialize( std::string methodName,
                                       	EGammaMvaEleEstimator::MVAType type,
                                       	Bool_t useBinnedVersion,
				       					std::vector<std::string> weightsfiles
  ) {

  //clean up first
  for (unsigned int i=0;i<fTMVAReader.size(); ++i) {
    if (fTMVAReader[i]) delete fTMVAReader[i];
  }
  fTMVAReader.clear();

  //initialize
  fisInitialized = kTRUE;
  fMVAType = type;
  fMethodname = methodName;
  fUseBinnedVersion = useBinnedVersion;

  //Define expected number of bins
  UInt_t ExpectedNBins = 0;
  if (!fUseBinnedVersion) {
    ExpectedNBins = 1;
  } else if (type == kTrig) {
    ExpectedNBins = 6;
  } else if (type == kNonTrig) {
    ExpectedNBins = 6;
  } else if (type == kIsoRings) {
    ExpectedNBins = 4;
  }
  fNMVABins = ExpectedNBins;
  
  //Check number of weight files given
  if (fNMVABins != weightsfiles.size() ) {
     edm::LogError("InvalidInput") << "Error: Expected Number of bins = " << fNMVABins << " does not equal to weightsfiles.size() = " 
              << weightsfiles.size(); 
  }

  //Loop over all bins
  for (unsigned int i=0;i<fNMVABins; ++i) {
  
    //TMVA::Reader *tmpTMVAReader = new TMVA::Reader( "!Color:!Silent:Error" );  
    TMVA::Reader *tmpTMVAReader = new TMVA::Reader( "!Color:Silent:Error" );  
    //tmpTMVAReader->SetVerbose(kTRUE);
    tmpTMVAReader->SetVerbose(kFALSE);
  
    if (type == kTrig) {
      // Pure tracking variables
      tmpTMVAReader->AddVariable("fbrem",           &fMVAVar_fbrem);
      tmpTMVAReader->AddVariable("kfchi2",          &fMVAVar_kfchi2);
      tmpTMVAReader->AddVariable("kfhits",          &fMVAVar_kfhits);
      tmpTMVAReader->AddVariable("gsfchi2",         &fMVAVar_gsfchi2);

      // Geometrical matchings
      tmpTMVAReader->AddVariable("deta",            &fMVAVar_deta);
      tmpTMVAReader->AddVariable("dphi",            &fMVAVar_dphi);
      tmpTMVAReader->AddVariable("detacalo",        &fMVAVar_detacalo);
      // tmpTMVAReader->AddVariable("dphicalo",        &fMVAVar_dphicalo);   // Pruned but save in your ntuple. 
    
      // Pure ECAL -> shower shapes
      tmpTMVAReader->AddVariable("see",             &fMVAVar_see);
      tmpTMVAReader->AddVariable("spp",             &fMVAVar_spp);
      tmpTMVAReader->AddVariable("etawidth",        &fMVAVar_etawidth);
      tmpTMVAReader->AddVariable("phiwidth",        &fMVAVar_phiwidth);
      tmpTMVAReader->AddVariable("e1x5e5x5",        &fMVAVar_e1x5e5x5);
      tmpTMVAReader->AddVariable("R9",              &fMVAVar_R9);
      // tmpTMVAReader->AddVariable("nbrems",          &fMVAVar_nbrems); // Pruned but save in your ntuple. 

      // Energy matching
      tmpTMVAReader->AddVariable("HoE",             &fMVAVar_HoE);
      tmpTMVAReader->AddVariable("EoP",             &fMVAVar_EoP); 
      tmpTMVAReader->AddVariable("IoEmIoP",         &fMVAVar_IoEmIoP);
      tmpTMVAReader->AddVariable("eleEoPout",       &fMVAVar_eleEoPout);
      //  tmpTMVAReader->AddVariable("EoPout",          &fMVAVar_EoPout); // Pruned but save in your ntuple.    
      if(i == 2 || i == 5) 
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);
      
      if(!fUseBinnedVersion)
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);

      // IP
      tmpTMVAReader->AddVariable("d0",              &fMVAVar_d0);
      tmpTMVAReader->AddVariable("ip3d",            &fMVAVar_ip3d);
    
      tmpTMVAReader->AddSpectator("eta",            &fMVAVar_eta);
      tmpTMVAReader->AddSpectator("pt",             &fMVAVar_pt);
    }
  
    if (type == kNonTrig) {
      // Pure tracking variables
      tmpTMVAReader->AddVariable("fbrem",           &fMVAVar_fbrem);
      tmpTMVAReader->AddVariable("kfchi2",          &fMVAVar_kfchi2);
      tmpTMVAReader->AddVariable("kfhits",          &fMVAVar_kfhits);
      tmpTMVAReader->AddVariable("gsfchi2",         &fMVAVar_gsfchi2);

      // Geometrical matchings
      tmpTMVAReader->AddVariable("deta",            &fMVAVar_deta);
      tmpTMVAReader->AddVariable("dphi",            &fMVAVar_dphi);
      tmpTMVAReader->AddVariable("detacalo",        &fMVAVar_detacalo);
      // tmpTMVAReader->AddVariable("dphicalo",        &fMVAVar_dphicalo);   // Pruned but save in your ntuple. 
    
      // Pure ECAL -> shower shapes
      tmpTMVAReader->AddVariable("see",             &fMVAVar_see);
      tmpTMVAReader->AddVariable("spp",             &fMVAVar_spp);
      tmpTMVAReader->AddVariable("etawidth",        &fMVAVar_etawidth);
      tmpTMVAReader->AddVariable("phiwidth",        &fMVAVar_phiwidth);
      tmpTMVAReader->AddVariable("e1x5e5x5",        &fMVAVar_e1x5e5x5);
      tmpTMVAReader->AddVariable("R9",              &fMVAVar_R9);
      // tmpTMVAReader->AddVariable("nbrems",          &fMVAVar_nbrems); // Pruned but save in your ntuple. 

      // Energy matching
      tmpTMVAReader->AddVariable("HoE",             &fMVAVar_HoE);
      tmpTMVAReader->AddVariable("EoP",             &fMVAVar_EoP); 
      tmpTMVAReader->AddVariable("IoEmIoP",         &fMVAVar_IoEmIoP);
      tmpTMVAReader->AddVariable("eleEoPout",       &fMVAVar_eleEoPout);
      //  tmpTMVAReader->AddVariable("EoPout",          &fMVAVar_EoPout); // Pruned but save in your ntuple. 
      if(i == 2 || i == 5) 
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);
    
      if(!fUseBinnedVersion)
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);

      tmpTMVAReader->AddSpectator("eta",            &fMVAVar_eta);
      tmpTMVAReader->AddSpectator("pt",             &fMVAVar_pt);
    }

    if (type == kIsoRings) {
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p0To0p1",         &fMVAVar_ChargedIso_DR0p0To0p1        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p1To0p2",         &fMVAVar_ChargedIso_DR0p1To0p2        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p2To0p3",         &fMVAVar_ChargedIso_DR0p2To0p3        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p3To0p4",         &fMVAVar_ChargedIso_DR0p3To0p4        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p4To0p5",         &fMVAVar_ChargedIso_DR0p4To0p5        );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p0To0p1",           &fMVAVar_GammaIso_DR0p0To0p1          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p1To0p2",           &fMVAVar_GammaIso_DR0p1To0p2          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p2To0p3",           &fMVAVar_GammaIso_DR0p2To0p3          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p3To0p4",           &fMVAVar_GammaIso_DR0p3To0p4          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p4To0p5",           &fMVAVar_GammaIso_DR0p4To0p5          );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p0To0p1",   &fMVAVar_NeutralHadronIso_DR0p0To0p1  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p1To0p2",   &fMVAVar_NeutralHadronIso_DR0p1To0p2  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p2To0p3",   &fMVAVar_NeutralHadronIso_DR0p2To0p3  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p3To0p4",   &fMVAVar_NeutralHadronIso_DR0p3To0p4  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p4To0p5",   &fMVAVar_NeutralHadronIso_DR0p4To0p5  );
      tmpTMVAReader->AddSpectator("eta",            &fMVAVar_eta);
      tmpTMVAReader->AddSpectator("pt",             &fMVAVar_pt);
    }
  
    tmpTMVAReader->BookMVA(fMethodname , weightsfiles[i]);
    fTMVAReader.push_back(tmpTMVAReader);
  }

}


//--------------------------------------------------------------------------------------------------
UInt_t EGammaMvaEleEstimator::GetMVABin( double eta, double pt) const {
  
    //Default is to return the first bin
    unsigned int bin = 0;

    if (fMVAType == EGammaMvaEleEstimator::kIsoRings) {
      if (pt < 10 && fabs(eta) < 1.479) bin = 0;
      if (pt < 10 && fabs(eta) >= 1.479) bin = 1;
      if (pt >= 10 && fabs(eta) < 1.479) bin = 2;
      if (pt >= 10 && fabs(eta) >= 1.479) bin = 3;
    }

    if (fMVAType == EGammaMvaEleEstimator::kNonTrig ) {
      bin = 0;
      if (pt < 10 && fabs(eta) < 0.8) bin = 0;
      if (pt < 10 && fabs(eta) >= 0.8 && fabs(eta) < 1.479 ) bin = 1;
      if (pt < 10 && fabs(eta) >= 1.479) bin = 2;
      if (pt >= 10 && fabs(eta) < 0.8) bin = 3;
      if (pt >= 10 && fabs(eta) >= 0.8 && fabs(eta) < 1.479 ) bin = 4;
      if (pt >= 10 && fabs(eta) >= 1.479) bin = 5;
    }


    if (fMVAType == EGammaMvaEleEstimator::kTrig) {
      bin = 0;
      if (pt < 20 && fabs(eta) < 0.8) bin = 0;
      if (pt < 20 && fabs(eta) >= 0.8 && fabs(eta) < 1.479 ) bin = 1;
      if (pt < 20 && fabs(eta) >= 1.479) bin = 2;
      if (pt >= 20 && fabs(eta) < 0.8) bin = 3;
      if (pt >= 20 && fabs(eta) >= 0.8 && fabs(eta) < 1.479 ) bin = 4;
      if (pt >= 20 && fabs(eta) >= 1.479) bin = 5;
    }

 

    return bin;
}

Double_t EGammaMvaEleEstimator::mvaValue(HWW& hww, Int_t ele, Bool_t printDebug) {

	Double_t mvavalue = -999.;

	Double_t fbrem 				=	hww.els_fbrem().at(ele); 
	Double_t kfchi2				=	hww.els_trkidx().at(ele)>=0 ? hww.trks_chi2().at(hww.els_trkidx().at(ele))/hww.trks_ndof().at(hww.els_trkidx().at(ele)) : 0.;
	Int_t    kfhits				= 	hww.els_trkidx().at(ele)>=0 ? hww.trks_nlayers().at(hww.els_trkidx().at(ele)) : -1;
	Double_t gsfchi2			= 	hww.els_chi2().at(ele) / hww.els_ndof().at(ele);
	Double_t deta				=	hww.els_dEtaIn().at(ele);
	Double_t dphi				=	hww.els_dPhiIn().at(ele); 
	Double_t detacalo			= 	hww.els_dEtaOut().at(ele);
	Double_t see				= 	hww.els_sigmaIEtaIEta().at(ele);
	Double_t spp				=	hww.els_sigmaIPhiIPhi().at(ele); // FIXME : check the case where it's 0 
	Double_t etawidth			=	hww.els_etaSCwidth().at(ele);
	Double_t phiwidth			= 	hww.els_phiSCwidth().at(ele);
	Double_t e1x5e5x5			=	hww.els_e5x5().at(ele)!=0. ? 1. - hww.els_e1x5().at(ele)/hww.els_e5x5().at(ele) : -1; 
	Double_t R9					= 	hww.els_e3x3().at(ele) / hww.els_eSCRaw().at(ele);
	Double_t HoE				=	hww.els_hOverE().at(ele);
	Double_t EoP				=	hww.els_eOverPIn().at(ele);
	//Double_t IoEmIoP			=	1./hww.els_eSC().at(ele) - 1./hww.els_p4().at(ele).P(); 
	Double_t IoEmIoP			=	1./hww.els_ecalEnergy().at(ele) - 1./hww.els_p4().at(ele).P(); // this is consistent with CMSSW 
	Double_t eleEoPout			=	hww.els_eOverPOut().at(ele);
	Double_t PreShowerOverRaw	=	hww.els_eSCPresh().at(ele) / hww.els_eSCRaw().at(ele);
	Double_t d0					=	electron_d0PV_wwV1_local(hww, ele);
	const double gsfsign = ( (gsftrks_d0_pv(hww, hww.els_gsftrkidx().at(ele),0).first)   >=0 ) ? 1. : -1.;
	Double_t ip3d				=	hww.els_ip3d().at(ele)*gsfsign; 
	Double_t eta				= 	hww.els_etaSC().at(ele);
	Double_t pt					= 	hww.els_p4().at(ele).pt();


	mvavalue =  EGammaMvaEleEstimator::mvaValue(
					 fbrem,
					 kfchi2,
					 kfhits,
					 gsfchi2,
					 deta,
					 dphi,
					 detacalo,
					// dphicalo,
					 see,
					 spp,
					 etawidth,
					 phiwidth,
					 e1x5e5x5,
					 R9,
					//Int_t    nbrems,
					 HoE,
					 EoP,
					 IoEmIoP,
					 eleEoPout,
					 PreShowerOverRaw,
					// EoPout,
					 d0,
					 ip3d,
					 eta,
					 pt,
					 printDebug); 


	return  mvavalue;
}

//--------------------------------------------------------------------------------------------------
Double_t EGammaMvaEleEstimator::mvaValue(Double_t fbrem, 
					Double_t kfchi2,
					Int_t    kfhits,
					Double_t gsfchi2,
					Double_t deta,
					Double_t dphi,
					Double_t detacalo,
					//Double_t dphicalo,
					Double_t see,
					Double_t spp,
					Double_t etawidth,
					Double_t phiwidth,
					Double_t e1x5e5x5,
					Double_t R9,
					//Int_t    nbrems,
					Double_t HoE,
					Double_t EoP,
					Double_t IoEmIoP,
					Double_t eleEoPout,
					Double_t PreShowerOverRaw,
					//Double_t EoPout,
					Double_t d0,
					Double_t ip3d,
					Double_t eta,
					Double_t pt,
					Bool_t printDebug) {
  
  if (!fisInitialized) { 
    edm::LogError("NotInitialized") << "Error: EGammaMvaEleEstimator not properly initialized."; 
    return -9999;
  }

  fMVAVar_fbrem           = fbrem; 
  fMVAVar_kfchi2          = kfchi2;
  fMVAVar_kfhits          = float(kfhits);   // BTD does not support int variables
  fMVAVar_gsfchi2         = gsfchi2;

  fMVAVar_deta            = deta;
  fMVAVar_dphi            = dphi;
  fMVAVar_detacalo        = detacalo;
  // fMVAVar_dphicalo        = dphicalo;


  fMVAVar_see             = see;
  fMVAVar_spp             = spp;
  fMVAVar_etawidth        = etawidth;
  fMVAVar_phiwidth        = phiwidth;
  fMVAVar_e1x5e5x5        = e1x5e5x5;
  fMVAVar_R9              = R9;
  //fMVAVar_nbrems          = float(nbrems);   // BTD does not support int variables


  fMVAVar_HoE             = HoE;
  fMVAVar_EoP             = EoP;
  fMVAVar_IoEmIoP         = IoEmIoP;
  fMVAVar_eleEoPout       = eleEoPout;
  fMVAVar_PreShowerOverRaw= PreShowerOverRaw;
  //fMVAVar_EoPout          = EoPout; 

  fMVAVar_d0              = d0;
  fMVAVar_ip3d            = ip3d;
  fMVAVar_eta             = eta;
  fMVAVar_pt              = pt;


  bindVariables();
  Double_t mva = -9999;  
  if (fUseBinnedVersion) {
    mva = fTMVAReader[GetMVABin(fMVAVar_eta,fMVAVar_pt)]->EvaluateMVA(fMethodname);
  } else {
    mva = fTMVAReader[0]->EvaluateMVA(fMethodname);
  }

  if(printDebug) {
	  LogDebug("EGammaMvaEleEstimator") << " bin "              << GetMVABin(fMVAVar_eta,fMVAVar_pt)
	                                    << " fbrem "            <<  fMVAVar_fbrem  
                                      << " kfchi2 "           << fMVAVar_kfchi2  
	                                    << " kfhits "           << fMVAVar_kfhits  
	                                    << " gsfchi2 "          << fMVAVar_gsfchi2  
	                                    << " deta "             <<  fMVAVar_deta  
	                                    << " dphi "             << fMVAVar_dphi  
                                      << " detacalo "         << fMVAVar_detacalo  
	                                    << " see "              << fMVAVar_see  
	                                    << " spp "              << fMVAVar_spp  
	                                    << " etawidth "         << fMVAVar_etawidth  
	                                    << " phiwidth "         << fMVAVar_phiwidth  
	                                    << " e1x5e5x5 "         << fMVAVar_e1x5e5x5  
	                                    << " R9 "               << fMVAVar_R9  
	                                    << " HoE "              << fMVAVar_HoE  
	                                    << " EoP "              << fMVAVar_EoP  
	                                    << " IoEmIoP "          << fMVAVar_IoEmIoP  
	                                    << " eleEoPout "        << fMVAVar_eleEoPout  
	                                    << " PreShowerOverRaw " << fMVAVar_PreShowerOverRaw  
	                                    << " d0 "               << fMVAVar_d0  
	                                    << " ip3d "             << fMVAVar_ip3d  
	                                    << " eta "              << fMVAVar_eta  
	                                    << " pt "               << fMVAVar_pt
                                      << " ### MVA "          << mva;
  }


  return mva;
}
//--------------------------------------------------------------------------------------------------
Double_t EGammaMvaEleEstimator::mvaValue(Double_t fbrem, 
					Double_t kfchi2,
					Int_t    kfhits,
					Double_t gsfchi2,
					Double_t deta,
					Double_t dphi,
					Double_t detacalo,
					//Double_t dphicalo,
					Double_t see,
					Double_t spp,
					Double_t etawidth,
					Double_t phiwidth,
					Double_t e1x5e5x5,
					Double_t R9,
					//Int_t    nbrems,
					Double_t HoE,
					Double_t EoP,
					Double_t IoEmIoP,
					Double_t eleEoPout,
					Double_t PreShowerOverRaw,
					//Double_t EoPout,
					Double_t eta,
					Double_t pt,
					Bool_t printDebug) {
  
  if (!fisInitialized) { 
    edm::LogError("NotInitialized") << "Error: EGammaMvaEleEstimator not properly initialized."; 
    return -9999;
  }

  fMVAVar_fbrem           = fbrem; 
  fMVAVar_kfchi2          = kfchi2;
  fMVAVar_kfhits          = float(kfhits);   // BTD does not support int variables
  fMVAVar_gsfchi2         = gsfchi2;

  fMVAVar_deta            = deta;
  fMVAVar_dphi            = dphi;
  fMVAVar_detacalo        = detacalo;


  fMVAVar_see             = see;
  fMVAVar_spp             = spp;
  fMVAVar_etawidth        = etawidth;
  fMVAVar_phiwidth        = phiwidth;
  fMVAVar_e1x5e5x5        = e1x5e5x5;
  fMVAVar_R9              = R9;


  fMVAVar_HoE             = HoE;
  fMVAVar_EoP             = EoP;
  fMVAVar_IoEmIoP         = IoEmIoP;
  fMVAVar_eleEoPout       = eleEoPout;
  fMVAVar_PreShowerOverRaw= PreShowerOverRaw;

  fMVAVar_eta             = eta;
  fMVAVar_pt              = pt;


  bindVariables();
  Double_t mva = -9999;  
  if (fUseBinnedVersion) {
    mva = fTMVAReader[GetMVABin(fMVAVar_eta,fMVAVar_pt)]->EvaluateMVA(fMethodname);
  } else {
    mva = fTMVAReader[0]->EvaluateMVA(fMethodname);
  }



  if(printDebug) {
	  LogDebug("EGammaMvaEleEstimator") << " bin "              << GetMVABin(fMVAVar_eta,fMVAVar_pt)
	                                    << " fbrem "            <<  fMVAVar_fbrem  
                                      << " kfchi2 "           << fMVAVar_kfchi2  
	                                    << " kfhits "           << fMVAVar_kfhits  
	                                    << " gsfchi2 "          << fMVAVar_gsfchi2  
	                                    << " deta "             <<  fMVAVar_deta  
	                                    << " dphi "             << fMVAVar_dphi  
                                      << " detacalo "         << fMVAVar_detacalo  
	                                    << " see "              << fMVAVar_see  
	                                    << " spp "              << fMVAVar_spp  
	                                    << " etawidth "         << fMVAVar_etawidth  
	                                    << " phiwidth "         << fMVAVar_phiwidth  
	                                    << " e1x5e5x5 "         << fMVAVar_e1x5e5x5  
	                                    << " R9 "               << fMVAVar_R9  
	                                    << " HoE "              << fMVAVar_HoE  
	                                    << " EoP "              << fMVAVar_EoP  
	                                    << " IoEmIoP "          << fMVAVar_IoEmIoP  
	                                    << " eleEoPout "        << fMVAVar_eleEoPout  
	                                    << " PreShowerOverRaw " << fMVAVar_PreShowerOverRaw  
	                                    << " d0 "               << fMVAVar_d0  
	                                    << " ip3d "             << fMVAVar_ip3d  
	                                    << " eta "              << fMVAVar_eta  
	                                    << " pt "               << fMVAVar_pt
                                      << " ### MVA "          << mva;
  }


  return mva;
}

void EGammaMvaEleEstimator::bindVariables() {

  // this binding is needed for variables that sometime diverge. 


  if(fMVAVar_fbrem < -1.)
    fMVAVar_fbrem = -1.;	
  
  fMVAVar_deta = fabs(fMVAVar_deta);
  if(fMVAVar_deta > 0.06)
    fMVAVar_deta = 0.06;
  
  fMVAVar_dphi = fabs(fMVAVar_dphi);
  if(fMVAVar_dphi > 0.6)
    fMVAVar_dphi = 0.6;
  
  if(fMVAVar_EoP > 20.)
    fMVAVar_EoP = 20.;
  
  if(fMVAVar_eleEoPout > 20.)
    fMVAVar_eleEoPout = 20.;
  
  fMVAVar_detacalo = fabs(fMVAVar_detacalo);
  if(fMVAVar_detacalo > 0.2)
    fMVAVar_detacalo = 0.2;
  
  if(fMVAVar_e1x5e5x5 < -1.)
    fMVAVar_e1x5e5x5 = -1;
  
  if(fMVAVar_e1x5e5x5 > 2.)
    fMVAVar_e1x5e5x5 = 2.; 
  
  if(fMVAVar_R9 > 5)
    fMVAVar_R9 = 5;
  
  if(fMVAVar_gsfchi2 > 200.)
    fMVAVar_gsfchi2 = 200;
  
  if(fMVAVar_kfchi2 > 10.)
    fMVAVar_kfchi2 = 10.;
  
  // Needed for a bug in CMSSW_420, fixed in more recent CMSSW versions
  if(std::isnan(fMVAVar_spp))
    fMVAVar_spp = 0.;	
  
  
  return;
}
