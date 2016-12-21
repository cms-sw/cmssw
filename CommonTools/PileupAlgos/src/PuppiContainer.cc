#include "CommonTools/PileupAlgos/interface/PuppiContainer.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "fastjet/internal/base.hh"
#include "fastjet/FunctionOfPseudoJet.hh"
#include "Math/ProbFunc.h"
#include "TMath.h"
#include <iostream>
#include <math.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

using namespace std;
using namespace fastjet;

PuppiContainer::PuppiContainer(const edm::ParameterSet &iConfig) {
    fPuppiDiagnostics = iConfig.getParameter<bool>("puppiDiagnostics");
    fApplyCHS        = iConfig.getParameter<bool>("applyCHS");
    fInvert          = iConfig.getParameter<bool>("invertPuppi");    
    fUseExp          = iConfig.getParameter<bool>("useExp");
    fPuppiWeightCut  = iConfig.getParameter<double>("MinPuppiWeight");
    std::vector<edm::ParameterSet> lAlgos = iConfig.getParameter<std::vector<edm::ParameterSet> >("algos");
    fNAlgos = lAlgos.size();
    for(unsigned int i0 = 0; i0 < lAlgos.size(); i0++) {
      fPuppiAlgo.emplace_back(lAlgos[i0]);
    }
}

void PuppiContainer::initialize(const std::vector<RecoObj> &iRecoObjects) {
    //Clear everything
    fRecoParticles.resize(0);
    fPFParticlesNodes.resize(0);
    fPFParticlesRap.resize(0);
    fPFParticlesPhi.resize(0);
    fPFParticles  .resize(0);
    fPFParticlesTree.clear();
    fChargedPVNodes.resize(0);
    fChargedPVRap.resize(0);
    fChargedPVPhi.resize(0);
    fChargedPV    .resize(0);
    fChargedPVTree.clear();
    fPupParticles .resize(0);
    fWeights      .resize(0);
    fVals.resize(0);
    fRawAlphas.resize(0);
    fAlphaMed     .resize(0);
    fAlphaRMS     .resize(0);
    //fChargedNoPV.resize(0);
    //Link to the RecoObjects
    fPVFrac = 0.;
    fNPV    = 1.;
    fRecoParticles = iRecoObjects;

    std::array<float,2> minpos({ {0.0f,0.0f} }), maxpos({ {0.0f,0.0f} });
    std::array<float,2> chminpos({ {0.0f,0.0f} }), chmaxpos({ {0.0f,0.0f} });

    for (unsigned int i = 0; i < fRecoParticles.size(); ++i){
        fastjet::PseudoJet curPseudoJet;
        const auto& fRecoParticle = fRecoParticles[i];
        // float nom = sqrt((fRecoParticle.m)*(fRecoParticle.m) + (fRecoParticle.pt)*(fRecoParticle.pt)*(cosh(fRecoParticle.eta))*(cosh(fRecoParticle.eta))) + (fRecoParticle.pt)*sinh(fRecoParticle.eta);//hacked
        // float denom = sqrt((fRecoParticle.m)*(fRecoParticle.m) + (fRecoParticle.pt)*(fRecoParticle.pt));//hacked
        // float rapidity = log(nom/denom);//hacked
        if (edm::isFinite(fRecoParticle.rapidity)){
            curPseudoJet.reset_PtYPhiM(fRecoParticle.pt,fRecoParticle.rapidity,fRecoParticle.phi,fRecoParticle.m);//hacked
        } else {        
            curPseudoJet.reset_PtYPhiM(0, 99., 0, 0);//skipping may have been a better choice     
        }                   
        //curPseudoJet.reset_PtYPhiM(fRecoParticle.pt,fRecoParticle.eta,fRecoParticle.phi,fRecoParticle.m);
        int puppi_register = 0;
        if(fRecoParticle.id == 0 or fRecoParticle.charge == 0)  puppi_register = 0; // zero is neutral hadron
        if(fRecoParticle.id == 1 and fRecoParticle.charge != 0) puppi_register = fRecoParticle.charge; // from PV use the
        if(fRecoParticle.id == 2 and fRecoParticle.charge != 0) puppi_register = fRecoParticle.charge+5; // from NPV use the charge as key +5 as key
        curPseudoJet.set_user_info( new PuppiUserInfo( puppi_register ) );
        // fill vector of pseudojets for internal references
	const double rap = curPseudoJet.rapidity();
	double phi = curPseudoJet.phi();
	if( phi > M_PI ) phi -= 2*M_PI; // convert to atan2 notation [-pi,pi]
        fPFParticlesNodes.emplace_back(i,(float)rap,(float)phi);
	fPFParticlesNodes.emplace_back(i,(float)rap,(float)(phi+2*M_PI));
	fPFParticlesNodes.emplace_back(i,(float)rap,(float)(phi-2*M_PI));
	fPFParticles.push_back(curPseudoJet);
	fPFParticlesRap.push_back(rap);
	fPFParticlesPhi.push_back(phi);
	
	if( i == 0 ) {
	  minpos[0] = rap; minpos[1] = phi-2*M_PI;
	  maxpos[0] = rap; maxpos[1] = phi+2*M_PI;
	} else {
	  minpos[0] = std::min((float)rap,minpos[0]);
	  minpos[1] = std::min((float)(phi-2*M_PI),minpos[1]);
	  maxpos[0] = std::max((float)rap,maxpos[0]);
	  maxpos[1] = std::max((float)(phi+2*M_PI),maxpos[1]);	  
	}
	
        //Take Charged particles associated to PV
        if(std::abs(fRecoParticle.id) == 1) { 
	  if( fChargedPV.size() == 0 ) {
	    chminpos[0] = rap; chminpos[1] = phi-2*M_PI;
	    chmaxpos[0] = rap; chmaxpos[1] = phi+2*M_PI;
	  } else {
	    chminpos[0] = std::min((float)rap,chminpos[0]);
	    chminpos[1] = std::min((float)(phi-2*M_PI),chminpos[1]);
	    chmaxpos[0] = std::max((float)rap,chmaxpos[0]);
	    chmaxpos[1] = std::max((float)(phi+2*M_PI),chmaxpos[1]);	  
	  }
	  fChargedPVNodes.emplace_back(fChargedPV.size(),(float)rap,(float)phi);
	  fChargedPVNodes.emplace_back(fChargedPV.size(),(float)rap,(float)(phi+2*M_PI));
	  fChargedPVNodes.emplace_back(fChargedPV.size(),(float)rap,(float)(phi-2*M_PI));
	  fChargedPV.push_back(curPseudoJet);
	  fChargedPVRap.push_back(rap);
	  fChargedPVPhi.push_back(phi);
	}
        if(std::abs(fRecoParticle.id) >= 1 ) fPVFrac+=1.;
        //if((fRecoParticle.id == 0) && (inParticles[i].id == 2))  _genParticles.push_back( curPseudoJet);
        //if(fRecoParticle.id <= 2 && !(inParticles[i].pt < fNeutralMinE && fRecoParticle.id < 2)) _pfchsParticles.push_back(curPseudoJet);
        //if(fRecoParticle.id == 3) _chargedNoPV.push_back(curPseudoJet);
        // if(fNPV < fRecoParticle.vtxId) fNPV = fRecoParticle.vtxId;
    }
    if (fPVFrac != 0) fPVFrac = double(fChargedPV.size())/fPVFrac;
    else fPVFrac = 0;

    KDTreeBox bounds(minpos[0],maxpos[0],
                     minpos[1],maxpos[1]);
    KDTreeBox chbounds(chminpos[0],chmaxpos[0],
		       chminpos[1],chmaxpos[1]);

    fPFParticlesTree.build(fPFParticlesNodes,bounds);
    fChargedPVTree.build(fChargedPVNodes,chbounds);

}
PuppiContainer::~PuppiContainer(){}

double PuppiContainer::goodVar(PseudoJet const &iPart,std::vector<PseudoJet> const &iParts, int iOpt,const double iRCone) {
    double lPup = 0;
    lPup = var_within_R(iOpt,iParts,iPart,iRCone);
    return lPup;
}
double PuppiContainer::var_within_R(int iId, const vector<PseudoJet> & particles, const PseudoJet& centre, const double R){
    if(iId == -1) return 1;

    //this is a circle in rapidity-phi
    //it would make more sense to have var definition consistent
    //fastjet::Selector sel = fastjet::SelectorCircle(R);
    //sel.set_reference(centre);
    //the original code used Selector infrastructure: it is too heavy here
    //logic of SelectorCircle is preserved below

    const double Rprime = R + 0.001; // make sure there is no roundoff in kd-tree search
    const double R2 = R*R;

    std::vector<KDNode> found;
    const double centreRap = centre.rapidity();
    double centrePhi = centre.phi();
    if( centrePhi > M_PI ) centrePhi -= 2*M_PI; // convert to atan2 notation [-pi,pi]
    KDTreeBox bounds((float)(centreRap - Rprime), (float)(centreRap + Rprime),
		     (float)(centrePhi - Rprime), (float)(centrePhi + Rprime));

    if( &particles == &fPFParticles ) {
      fPFParticlesTree.search(bounds,found);
    } else if( &particles == &fChargedPV ) {
      fChargedPVTree.search(bounds,found);
    } else {
      throw cms::Exception("InvalidParticleCollection")
	<< "var_within_R passed a collection that is not fPFParticles or fChargedPV!";
    }
    
    double var = 0;
    for( auto const& node : found ) {
      double pt(std::numeric_limits<double>::max()), dr2(std::numeric_limits<double>::max());
      if( &particles == &fPFParticles ) {
	dr2 = reco::deltaR2(centreRap,centrePhi,fPFParticlesRap[node.data],fPFParticlesPhi[node.data]); 
      } else if( &particles == &fChargedPV ) {
	dr2 = reco::deltaR2(centreRap,centrePhi,fChargedPVRap[node.data],fChargedPVPhi[node.data]); 
      }
      
      if( dr2 >= R2 || dr2  <  0.0001 ) continue;
      
      switch(iId) {
      case 0:
	var += (pt / dr2);
	break;
      case 1:
      case 4:
	var += pt;
	break;
      case 2:
      case 3:
	var += (1.0/dr2);
	break;
      case 5:
	var += (pt * pt / dr2);
	break;
      default:
	break;
      }
    }
        
    if(iId == 1) var += centre.pt(); //Sum in a cone
    else if(iId == 0 && var != 0) var = log(var);
    else if(iId == 3 && var != 0) var = log(var);
    else if(iId == 5 && var != 0) var = log(var);
    return var;
}
//In fact takes the median not the average
void PuppiContainer::getRMSAvg(int iOpt,std::vector<fastjet::PseudoJet> const &iConstits,std::vector<fastjet::PseudoJet> const &iParticles,std::vector<fastjet::PseudoJet> const &iChargedParticles) {
    for(unsigned int i0 = 0; i0 < iConstits.size(); i0++ ) {
        double pVal = -1;
        //Calculate the Puppi Algo to use
        int  pPupId   = getPuppiId(iConstits[i0].pt(),iConstits[i0].eta());
        if(pPupId == -1 || fPuppiAlgo[pPupId].numAlgos() <= iOpt){
            fVals.push_back(-1);
            continue;
        }
        //Get the Puppi Sub Algo (given iteration)
        int  pAlgo    = fPuppiAlgo[pPupId].algoId   (iOpt);
        bool pCharged = fPuppiAlgo[pPupId].isCharged(iOpt);
        double pCone  = fPuppiAlgo[pPupId].coneSize (iOpt);
        //Compute the Puppi Metric
        if(!pCharged) pVal = goodVar(iConstits[i0],iParticles       ,pAlgo,pCone);
        if( pCharged) pVal = goodVar(iConstits[i0],iChargedParticles,pAlgo,pCone);
        fVals.push_back(pVal);
        //if(std::isnan(pVal) || std::isinf(pVal)) cerr << "====> Value is Nan " << pVal << " == " << iConstits[i0].pt() << " -- " << iConstits[i0].eta() << endl;
        if( ! edm::isFinite(pVal)) {
            LogDebug( "NotFound" )  << "====> Value is Nan " << pVal << " == " << iConstits[i0].pt() << " -- " << iConstits[i0].eta() << endl;
            continue;
        }
        
        // // fPuppiAlgo[pPupId].add(iConstits[i0],pVal,iOpt);
        //code added by Nhan, now instead for every algorithm give it all the particles
        for(int i1 = 0; i1 < fNAlgos; i1++){
            pAlgo    = fPuppiAlgo[i1].algoId   (iOpt);
            pCharged = fPuppiAlgo[i1].isCharged(iOpt);
            pCone    = fPuppiAlgo[i1].coneSize (iOpt);
            double curVal = -1; 
            if(!pCharged) curVal = goodVar(iConstits[i0],iParticles       ,pAlgo,pCone);
            if( pCharged) curVal = goodVar(iConstits[i0],iChargedParticles,pAlgo,pCone);
            //std::cout << "i1 = " << i1 << ", curVal = " << curVal << ", eta = " << iConstits[i0].eta() << ", pupID = " << pPupId << std::endl;
            fPuppiAlgo[i1].add(iConstits[i0],curVal,iOpt);
        }

    }
    for(int i0 = 0; i0 < fNAlgos; i0++) fPuppiAlgo[i0].computeMedRMS(iOpt,fPVFrac);
}
//In fact takes the median not the average
void PuppiContainer::getRawAlphas(int iOpt,std::vector<fastjet::PseudoJet> const &iConstits,std::vector<fastjet::PseudoJet> const &iParticles,std::vector<fastjet::PseudoJet> const &iChargedParticles) {
    for(int j0 = 0; j0 < fNAlgos; j0++){
        for(unsigned int i0 = 0; i0 < iConstits.size(); i0++ ) {
            double pVal = -1;
            //Get the Puppi Sub Algo (given iteration)
            int  pAlgo    = fPuppiAlgo[j0].algoId   (iOpt);
            bool pCharged = fPuppiAlgo[j0].isCharged(iOpt);
            double pCone  = fPuppiAlgo[j0].coneSize (iOpt);
            //Compute the Puppi Metric
            if(!pCharged) pVal = goodVar(iConstits[i0],iParticles       ,pAlgo,pCone);
            if( pCharged) pVal = goodVar(iConstits[i0],iChargedParticles,pAlgo,pCone);
            fRawAlphas.push_back(pVal);
            if( ! edm::isFinite(pVal)) {
                LogDebug( "NotFound" )  << "====> Value is Nan " << pVal << " == " << iConstits[i0].pt() << " -- " << iConstits[i0].eta() << endl;
                continue;
            }
        }
    }
}
int    PuppiContainer::getPuppiId( float iPt, float iEta) {
    int lId = -1;
    for(int i0 = 0; i0 < fNAlgos; i0++) {
        int nEtaBinsPerAlgo = fPuppiAlgo[i0].etaBins();
        for (int i1 = 0; i1 < nEtaBinsPerAlgo; i1++){
            if ( (std::abs(iEta) > fPuppiAlgo[i0].etaMin(i1)) && (std::abs(iEta) < fPuppiAlgo[i0].etaMax(i1)) ){ 
                fPuppiAlgo[i0].fixAlgoEtaBin( i1 );
                if(iPt > fPuppiAlgo[i0].ptMin()){
                    lId = i0; 
                    break;
                }
            }
        }
    }
    //if(lId == -1) std::cerr << "Error : Full fiducial range is not defined " << std::endl;
    return lId;
}
double PuppiContainer::getChi2FromdZ(double iDZ) {
    //We need to obtain prob of PU + (1-Prob of LV)
    // Prob(LV) = Gaus(dZ,sigma) where sigma = 1.5mm  (its really more like 1mm)
    //double lProbLV = ROOT::Math::normal_cdf_c(std::abs(iDZ),0.2)*2.; //*2 is to do it double sided
    //Take iDZ to be corrected by sigma already
    double lProbLV = ROOT::Math::normal_cdf_c(std::abs(iDZ),1.)*2.; //*2 is to do it double sided
    double lProbPU = 1-lProbLV;
    if(lProbPU <= 0) lProbPU = 1e-16;   //Quick Trick to through out infs
    if(lProbPU >= 0) lProbPU = 1-1e-16; //Ditto
    double lChi2PU = TMath::ChisquareQuantile(lProbPU,1);
    lChi2PU*=lChi2PU;
    return lChi2PU;
}
std::vector<double> const & PuppiContainer::puppiWeights() {
    fPupParticles .resize(0);
    fWeights      .resize(0);
    fVals         .resize(0);
    for(int i0 = 0; i0 < fNAlgos; i0++) fPuppiAlgo[i0].reset();
    
    int lNMaxAlgo = 1;
    for(int i0 = 0; i0 < fNAlgos; i0++) lNMaxAlgo = std::max(fPuppiAlgo[i0].numAlgos(),lNMaxAlgo);
    //Run through all compute mean and RMS
    int lNParticles    = fRecoParticles.size();
    for(int i0 = 0; i0 < lNMaxAlgo; i0++) {
        getRMSAvg(i0,fPFParticles,fPFParticles,fChargedPV);
    }
    if (fPuppiDiagnostics) getRawAlphas(0,fPFParticles,fPFParticles,fChargedPV);

    std::vector<double> pVals;
    for(int i0 = 0; i0 < lNParticles; i0++) {
        //Refresh
        pVals.clear();
        double pWeight = 1;
        //Get the Puppi Id and if ill defined move on
        int  pPupId   = getPuppiId(fRecoParticles[i0].pt,fRecoParticles[i0].eta);
        if(pPupId == -1) {
            fWeights .push_back(pWeight);
            fAlphaMed.push_back(-10);
            fAlphaRMS.push_back(-10);            
            continue;
        }
        // fill the p-values
        double pChi2   = 0;
        if(fUseExp){
            //Compute an Experimental Puppi Weight with delta Z info (very simple example)
            pChi2 = getChi2FromdZ(fRecoParticles[i0].dZ);
            //Now make sure Neutrals are not set
            if(fRecoParticles[i0].pfType > 3) pChi2 = 0;
        }
        //Fill and compute the PuppiWeight
        int lNAlgos = fPuppiAlgo[pPupId].numAlgos();
        for(int i1 = 0; i1 < lNAlgos; i1++) pVals.push_back(fVals[lNParticles*i1+i0]);

        pWeight = fPuppiAlgo[pPupId].compute(pVals,pChi2);
        //Apply the CHS weights
        if(fRecoParticles[i0].id == 1 && fApplyCHS ) pWeight = 1;
        if(fRecoParticles[i0].id == 2 && fApplyCHS ) pWeight = 0;
        //Basic Weight Checks
        if( ! edm::isFinite(pWeight)) {
            pWeight = 0.0;
            LogDebug("PuppiWeightError") << "====> Weight is nan : " << pWeight << " : pt " << fRecoParticles[i0].pt << " -- eta : " << fRecoParticles[i0].eta << " -- Value" << fVals[i0] << " -- id :  " << fRecoParticles[i0].id << " --  NAlgos: " << lNAlgos << std::endl;
        }
        //Basic Cuts
        if(pWeight                         < fPuppiWeightCut) pWeight = 0;  //==> Elminate the low Weight stuff
        if(pWeight*fPFParticles[i0].pt()   < fPuppiAlgo[pPupId].neutralPt(fNPV) && fRecoParticles[i0].id == 0 ) pWeight = 0;  //threshold cut on the neutral Pt
        if(fInvert) pWeight = 1.-pWeight;
        //std::cout << "fRecoParticles[i0].pt = " <<  fRecoParticles[i0].pt << ", fRecoParticles[i0].charge = " << fRecoParticles[i0].charge << ", fRecoParticles[i0].id = " << fRecoParticles[i0].id << ", weight = " << pWeight << std::endl;

        fWeights .push_back(pWeight);
        fAlphaMed.push_back(fPuppiAlgo[pPupId].median());
        fAlphaRMS.push_back(fPuppiAlgo[pPupId].rms());        
        //Now get rid of the thrown out weights for the particle collection

        // leave these lines in, in case want to move eventually to having no 1-to-1 correspondence between puppi and pf cands
        // if( std::abs(pWeight) < std::numeric_limits<double>::denorm_min() ) continue; // this line seems not to work like it's supposed to...
        // if(std::abs(pWeight) <= 0. ) continue; 
        
        //Produce
        PseudoJet curjet( pWeight*fPFParticles[i0].px(), pWeight*fPFParticles[i0].py(), pWeight*fPFParticles[i0].pz(), pWeight*fPFParticles[i0].e() );
        curjet.set_user_index(i0);
        fPupParticles.push_back(curjet);
    }
    return fWeights;
}


