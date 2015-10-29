#include "CommonTools/PileupAlgos/interface/PuppiAlgo.h"
#include "CommonTools/PileupAlgos/interface/PuppiContainer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "fastjet/internal/base.hh"
#include "Math/QuantFuncMathCore.h"
#include "Math/SpecFuncMathCore.h"
#include "Math/ProbFunc.h"
#include "TMath.h"


PuppiAlgo::PuppiAlgo(edm::ParameterSet &iConfig) {
    fEtaMin             = iConfig.getParameter<std::vector< double > >("etaMin");
    fEtaMax             = iConfig.getParameter<std::vector< double > >("etaMax");
    fPtMin              = iConfig.getParameter<std::vector< double > >("ptMin");
    fNeutralPtMin       = iConfig.getParameter<std::vector< double > >("MinNeutralPt");      // Weighted Neutral Pt Cut
    fNeutralPtSlope     = iConfig.getParameter<std::vector< double > >("MinNeutralPtSlope"); // Slope vs #pv
    fRMSEtaSF           = iConfig.getParameter<std::vector< double > >("RMSEtaSF");
    fMedEtaSF           = iConfig.getParameter<std::vector< double > >("MedEtaSF");
    fEtaMaxExtrap       = iConfig.getParameter<double>("EtaMaxExtrap");
    
    std::vector<edm::ParameterSet> lAlgos = iConfig.getParameter<std::vector<edm::ParameterSet> >("puppiAlgos");
    fNAlgos = lAlgos.size();
    //Uber Configurable Puppi
    std::vector<double> tmprms;
    std::vector<double> tmpmed;

    for(unsigned int i0 = 0; i0 < lAlgos.size(); i0++)  {
        int    pAlgoId      = lAlgos[i0].getParameter<int >  ("algoId");
        bool   pCharged     = lAlgos[i0].getParameter<bool>  ("useCharged");
        bool   pWeight0     = lAlgos[i0].getParameter<bool>  ("applyLowPUCorr");
        int    pComb        = lAlgos[i0].getParameter<int>   ("combOpt");           // 0=> add in chi2/1=>Multiply p-values
        double pConeSize    = lAlgos[i0].getParameter<double>("cone");              // Min Pt when computing pt and rms
        double pRMSPtMin    = lAlgos[i0].getParameter<double>("rmsPtMin");          // Min Pt when computing pt and rms
        double pRMSSF       = lAlgos[i0].getParameter<double>("rmsScaleFactor");    // Additional Tuning parameter for Jokers
        fAlgoId        .push_back(pAlgoId);
        fCharged       .push_back(pCharged);
        fAdjust        .push_back(pWeight0);
        fCombId        .push_back(pComb);
        fConeSize      .push_back(pConeSize);
        fRMSPtMin      .push_back(pRMSPtMin);
        fRMSScaleFactor.push_back(pRMSSF);
        double pRMS  = 0;
        double pMed  = 0;
        double pMean = 0;
        int    pNCount = 0;
        fRMS   .push_back(pRMS);
        fMedian.push_back(pMed);
        fMean  .push_back(pMean);
        fNCount.push_back(pNCount);

        tmprms.clear();
        tmpmed.clear();
        for (unsigned int j0 = 0; j0 < fEtaMin.size(); j0++){
            tmprms.push_back(pRMS);
            tmpmed.push_back(pMed);
        }
        fRMS_perEta.push_back(tmprms);
        fMedian_perEta.push_back(tmpmed);
    }

    cur_PtMin = -99.;
    cur_NeutralPtMin = -99.;
    cur_NeutralPtSlope = -99.;
    cur_RMS = -99.;
    cur_Med = -99.;
}
PuppiAlgo::~PuppiAlgo() {
    fPups  .clear();
    fPupsPV.clear();
}
void PuppiAlgo::reset() {
    fPups  .clear();
    fPupsPV.clear();
    for(unsigned int i0 = 0; i0 < fNAlgos; i0++) {
        fMedian[i0] =  0;
        fRMS   [i0] =  0;
        fMean  [i0] =  0;
        fNCount[i0] =  0;
    }
}

void PuppiAlgo::fixAlgoEtaBin(int i_eta) {
  cur_PtMin = fPtMin[i_eta]; 
  cur_NeutralPtMin = fNeutralPtMin[i_eta];
  cur_NeutralPtSlope = fNeutralPtSlope[i_eta];
  cur_RMS = fRMS_perEta[0][i_eta]; // 0 is number of algos within this eta bin
  cur_Med = fMedian_perEta[0][i_eta]; // 0 is number of algos within this eta bin
}

void PuppiAlgo::add(const fastjet::PseudoJet &iParticle,const double &iVal,const unsigned int iAlgo) {
    if(iParticle.pt() < fRMSPtMin[iAlgo]) return;
    // Change from SRR : Previously used fastjet::PseudoJet::user_index to decide the particle type.
    // In CMSSW we use the user_index to specify the index in the input collection, so I invented
    // a new mechanism using the fastjet UserInfo functionality. Of course, it's still just an integer
    // but that interface could be changed (or augmented) if desired / needed.
    int puppi_register = std::numeric_limits<int>::lowest();
    if ( iParticle.has_user_info() ) {
        PuppiContainer::PuppiUserInfo const * pInfo = dynamic_cast<PuppiContainer::PuppiUserInfo const *>( iParticle.user_info_ptr() );
        if ( pInfo != 0 ) {
            puppi_register = pInfo->puppi_register();
        }
    }
    if ( puppi_register == std::numeric_limits<int>::lowest() ) {
        throw cms::Exception("PuppiRegisterNotSet") << "The puppi register is not set. This must be set before use.\n";
    }

    //// original code
    // if(fCharged[iAlgo] && std::abs(puppi_register)  < 1) return;
    // if(fCharged[iAlgo] && (std::abs(puppi_register) >=1 && std::abs(puppi_register) <=2)) fPupsPV.push_back(iVal);
    //if(fCharged[iAlgo] && std::abs(puppi_register) < 3) return;
    //// if used fCharged and not CHPU, just return
    // fPups.push_back(iVal); //original
    // fNCount[iAlgo]++;

    // added by Nhan -- for all eta regions, compute mean/RMS from the central charged PU
    //std::cout << "std::abs(puppi_register) = " << std::abs(puppi_register) << std::endl;
    if ((std::abs(iParticle.eta()) < fEtaMaxExtrap) && (std::abs(puppi_register) >= 3)){
        fPups.push_back(iVal);
        // fPupsPV.push_back(iVal);        
        fNCount[iAlgo]++;
    }
    // for the low PU case, correction.  for checking that the PU-only median will be below the PV particles
    if(std::abs(iParticle.eta()) < fEtaMaxExtrap && (std::abs(puppi_register) >=1 && std::abs(puppi_register) <=2)) fPupsPV.push_back(iVal);

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//NHAN'S VERSION
void PuppiAlgo::computeMedRMS(const unsigned int &iAlgo,const double &iPVFrac) {

    //std::cout << "fNCount[iAlgo] = " << fNCount[iAlgo] << std::endl;
    if(iAlgo >= fNAlgos   ) return;
    if(fNCount[iAlgo] == 0) return;

    // sort alphas
    int lNBefore = 0;
    for(unsigned int i0 = 0; i0 < iAlgo; i0++) lNBefore += fNCount[i0];
    std::sort(fPups.begin()+lNBefore,fPups.begin()+lNBefore+fNCount[iAlgo]);
    
    // in case you have alphas == 0
    int lNum0 = 0;
    for(int i0 = lNBefore; i0 < lNBefore+fNCount[iAlgo]; i0++) {
        if(fPups[i0] == 0) lNum0 = i0-lNBefore;
    }

    // comput median, removed lCorr for now
    int lNHalfway = lNBefore + lNum0 + int( double( fNCount[iAlgo]-lNum0 )*0.50);
    fMedian[iAlgo] = fPups[lNHalfway];
    double lMed = fMedian[iAlgo];  //Just to make the readability easier
    
    int lNRMS = 0;
    for(int i0 = lNBefore; i0 < lNBefore+fNCount[iAlgo]; i0++) {
        fMean[iAlgo] += fPups[i0];
        if(fPups[i0] == 0) continue;
        // if(!fCharged[iAlgo] && fAdjust[iAlgo] && fPups[i0] > lMed) continue;
        if(fAdjust[iAlgo] && fPups[i0] > lMed) continue;
        lNRMS++;
        fRMS [iAlgo] += (fPups[i0]-lMed)*(fPups[i0]-lMed);
    }
    fMean[iAlgo]/=fNCount[iAlgo];
    if(lNRMS > 0) fRMS [iAlgo]/=lNRMS;
    if(fRMS[iAlgo] == 0) fRMS[iAlgo] = 1e-5;
    // here is the raw RMS
    fRMS [iAlgo] = sqrt(fRMS[iAlgo]);

    // some ways to do corrections to fRMS and fMedian
    fRMS [iAlgo] *= fRMSScaleFactor[iAlgo];

    if(fAdjust[iAlgo]){ 
        //Adjust the p-value to correspond to the median
        std::sort(fPupsPV.begin(),fPupsPV.end());
        int lNPV = 0; 
        for(unsigned int i0 = 0; i0 < fPupsPV.size(); i0++) if(fPupsPV[i0] <= lMed ) lNPV++;
        double lAdjust = double(lNPV)/double(lNPV+0.5*fNCount[iAlgo]);
        if(lAdjust > 0) {
          fMedian[iAlgo] -= sqrt(ROOT::Math::chisquared_quantile(lAdjust,1.)*fRMS[iAlgo]);
          fRMS[iAlgo]    -= sqrt(ROOT::Math::chisquared_quantile(lAdjust,1.)*fRMS[iAlgo]);
        }        
    }

    // fRMS_perEta[iAlgo]    *= cur_RMSEtaSF;
    // fMedian_perEta[iAlgo] *= cur_MedEtaSF;

    for (unsigned int j0 = 0; j0 < fEtaMin.size(); j0++){
        fRMS_perEta[iAlgo][j0] = fRMS[iAlgo]*fRMSEtaSF[j0];
        fMedian_perEta[iAlgo][j0] = fMedian[iAlgo]*fMedEtaSF[j0];
    }    

}
////////////////////////////////////////////////////////////////////////////////

//This code is probably a bit confusing
double PuppiAlgo::compute(std::vector<double> const &iVals,double iChi2) const {

    if(fAlgoId[0] == -1) return 1;
    double lVal  = 0.;
    double lPVal = 1.;
    int    lNDOF = 0;
    for(unsigned int i0 = 0; i0 < fNAlgos; i0++) {
        if(fNCount[i0] == 0) return 1.;   //in the NoPU case return 1.
        if(fCombId[i0] == 1 && i0 > 0) {  //Compute the previous p-value so that p-values can be multiplieed
            double pPVal = ROOT::Math::chisquared_cdf(lVal,lNDOF);
            lPVal *= pPVal;
            lNDOF = 0;
            lVal  = 0;
        }
        double pVal = iVals[i0];
        //Special Check for any algo with log(0)
        if(fAlgoId[i0] == 0 && iVals[i0] == 0) pVal = cur_Med;
        if(fAlgoId[i0] == 3 && iVals[i0] == 0) pVal = cur_Med;
        if(fAlgoId[i0] == 5 && iVals[i0] == 0) pVal = cur_Med;
        lVal += (pVal-cur_Med)*(fabs(pVal-cur_Med))/cur_RMS/cur_RMS;
        lNDOF++;
        if(i0 == 0 && iChi2 != 0) lNDOF++;      //Add external Chi2 to first element
        if(i0 == 0 && iChi2 != 0) lVal+=iChi2;  //Add external Chi2 to first element
    }
    //Top it off with the last calc
    lPVal *= ROOT::Math::chisquared_cdf(lVal,lNDOF);
    return lPVal;

}
