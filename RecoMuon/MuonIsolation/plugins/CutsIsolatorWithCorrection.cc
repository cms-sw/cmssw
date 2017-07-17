#include "CutsIsolatorWithCorrection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace muonisolation;

CutsIsolatorWithCorrection::CutsIsolatorWithCorrection(const edm::ParameterSet & par,
						       edm::ConsumesCollector && iC ):
  theCuts(par.getParameter<std::vector<double> > ("EtaBounds"),
	  par.getParameter<std::vector<double> > ("ConeSizes"),
	  par.getParameter<std::vector<double> > ("Thresholds")),
  theCutsRel(par.getParameter<std::vector<double> > ("EtaBoundsRel"),
	  par.getParameter<std::vector<double> > ("ConeSizesRel"),
	  par.getParameter<std::vector<double> > ("ThresholdsRel")),
  theCutAbsIso(par.getParameter<bool>("CutAbsoluteIso")),
  theCutRelativeIso(par.getParameter<bool>("CutRelativeIso")),
  theUseRhoCorrection(par.getParameter<bool>("UseRhoCorrection")),
  theRhoToken(iC.consumes<double>(par.getParameter<edm::InputTag>("RhoSrc"))),
  theRhoMax(par.getParameter<double>("RhoMax")),
  theRhoScaleBarrel(par.getParameter<double>("RhoScaleBarrel")),
  theRhoScaleEndcap(par.getParameter<double>("RhoScaleEndcap")),
  theEffAreaSFBarrel(par.getParameter<double>("EffAreaSFBarrel")),
  theEffAreaSFEndcap(par.getParameter<double>("EffAreaSFEndcap")),
  theReturnAbsoluteSum(par.getParameter<bool>("ReturnAbsoluteSum")),
  theReturnRelativeSum(par.getParameter<bool>("ReturnRelativeSum")),
  theAndOrCuts(par.getParameter<bool>("AndOrCuts"))
{
  if (! ( theCutAbsIso || theCutRelativeIso ) ) throw cms::Exception("BadConfiguration")
    << "Something has to be cut: set either CutAbsoluteIso or CutRelativeIso to true";
}

double CutsIsolatorWithCorrection::depSum(const DepositContainer& deposits, double dr, double corr) const {
  double dephlt = -corr;
  unsigned int nDeps = deposits.size();
  //  edm::LogWarning("CutsIsolatorWithCorrection::depSumIn")
  //    << "add nDeposit "<< nDeps<< " \t dr "<<dr<<" \t corr "<<corr;
  for(unsigned int iDep = 0; iDep < nDeps; ++iDep ){
    double lDep = deposits[iDep].dep->depositWithin(dr);
    dephlt += lDep;
    //    edm::LogWarning("CutsIsolatorWithCorrection::depSumIDep")
    //      <<"dep "<<iDep<<" \t added "<<lDep<<" \t sumnow "<<dephlt;
  }

  return dephlt;
}

MuIsoBaseIsolator::Result CutsIsolatorWithCorrection::result(const DepositContainer& deposits, const reco::Track& tk, const edm::Event* ev ) const {
  Result answer(ISOL_BOOL_TYPE);
  
  bool absDecision = false;
  bool relDecision = false;

  double rho = 0.0;
  double effAreaSF = 1.0;

  static const double pi = 3.14159265358979323846;

  //  edm::LogWarning("CutsIsolatorWithCorrection::resultIn")
  //    <<"Start tk.pt "<<tk.pt()<<" \t tk.eta "<<tk.eta()<<" \t tk.phi "<<tk.phi();



  if (theUseRhoCorrection){
    edm::Handle<double> rhoHandle; 
    ev->getByToken(theRhoToken, rhoHandle); 
    rho = *(rhoHandle.product());
    if (rho < 0.0) rho = 0.0;
    double rhoScale = fabs(tk.eta()) > 1.442 ? theRhoScaleEndcap : theRhoScaleBarrel;
    effAreaSF = fabs(tk.eta()) > 1.442 ? theEffAreaSFEndcap : theEffAreaSFBarrel;
    //    edm::LogWarning("CutsIsolatorWithCorrection::resultInRho")
    //      << "got rho "<<rho<<" vs max "<<theRhoMax<<" will scale by "<<rhoScale;
    if (rho > theRhoMax){
      rho = theRhoMax;
    }
    rho = rho*rhoScale;
    //    edm::LogWarning("CutsIsolatorWithCorrection::resultOutRho")<<" final rho "<<rho;
  }

  if (theCutAbsIso){
    muonisolation::Cuts::CutSpec cuts_here = theCuts(tk.eta());
    double conesize = cuts_here.conesize;
    double dephlt = depSum(deposits, conesize, rho*conesize*conesize*pi*effAreaSF);
    if (theReturnAbsoluteSum ) answer.valFloat = (float)dephlt;
    if (dephlt<cuts_here.threshold) {
      absDecision = true;
    } else {
      absDecision = false;
    }
    //    edm::LogWarning("CutsIsolatorWithCorrection::resultOutAbsIso")
    //      <<"compared dephlt "<<dephlt<<" \t with "<<cuts_here.threshold;
  } else absDecision = true;
  
  if (theCutRelativeIso){
    muonisolation::Cuts::CutSpec cuts_here = theCutsRel(tk.eta());
    double conesize = cuts_here.conesize;
    double dephlt = depSum(deposits, conesize, rho*conesize*conesize*pi*effAreaSF)/tk.pt();
    if (theReturnRelativeSum ) answer.valFloat = (float)dephlt;
    if (dephlt<cuts_here.threshold) {
      relDecision = true;
    } else {
      relDecision = false;
    }
    //    edm::LogWarning("CutsIsolatorWithCorrection::resultOutRelIso")
    //      <<"compared dephlt "<<dephlt<<" \t with "<<cuts_here.threshold;
  } else relDecision = true;
  

  if (theAndOrCuts){
    answer.valBool = absDecision && relDecision;
  } else {
    answer.valBool = absDecision || relDecision;
  }

  //  edm::LogWarning("CutsIsolatorWithCorrection::result")
  //    <<"isAbsIsolated "<<absDecision<<" \t isRelIsolated  "<<relDecision
  //    <<" \t combined with AND "<<theAndOrCuts
  //    <<" = "  << answer.valBool;
  
  return answer;
}
