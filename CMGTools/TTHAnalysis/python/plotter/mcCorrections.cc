#include <smearer.h>
#include <cmath>
using std::abs;
using std::exp;
using std::log;

double scaleIpVarsMC(double ipvar, int pdgId, double pt, double eta, int mcMatchId, int mcMatchAny) {
    if (abs(pdgId) == 13) {
        if (mcMatchId > 0 || mcMatchAny <= 1) {
            return ipvar * (abs(eta) < 1.5 ? 1.04 : 1.10);
        } else {
            return ipvar * 0.95;
        }
    } else {
        if (mcMatchId > 0 || mcMatchAny <= 1) {
            return ipvar * (abs(eta) < 1.479 ? 1.02 : 1.07);
        } else {
            return ipvar * 0.95;
        }
    }
}
double scaleSip3dMC(double sip3d, int pdgId, double pt, double eta, int mcMatchId, int mcMatchAny) {
    if (abs(pdgId) == 11 && (mcMatchId > 0 || mcMatchAny <= 1) && abs(eta) >= 1.479) {
        return logSmearMC(sip3d, 0.10, 0.2);
    }
    return scaleIpVarsMC(sip3d,pdgId,pt,eta,mcMatchId,mcMatchAny);
}
double scaleDzMC(double dz, int pdgId, double pt, double eta, int mcMatchId, int mcMatchAny) {
    if (abs(pdgId) == 11 && (mcMatchId > 0 || mcMatchAny <= 1) && abs(eta) >= 1.479) {
        return logSmearMC(dz, 0.20, 0.3);
    }
    return scaleIpVarsMC(dz,pdgId,pt,eta,mcMatchId,mcMatchAny);
}
double scaleDxyMC(double dxy, int pdgId, double pt, double eta, int mcMatchId, int mcMatchAny) {
    if (abs(pdgId) == 11 && (mcMatchId > 0 || mcMatchAny <= 1) && abs(eta) >= 1.479) {
        return logSmearMC(dxy, 0.07, 0.3);
    }
    return scaleIpVarsMC(dxy,pdgId,pt,eta,mcMatchId,mcMatchAny);
}

double correctJetPtRatioMC(double jetPtRatio, int pdgId, double pt, double eta, int mcMatchId, int mcMatchAny) {
    if (mcMatchAny >= 2) {
        if (pt < 15 && jetPtRatio == 1) {
            if (gSmearer_->Rndm() < 0.2) {
                if (abs(eta) < 1.5) {
                    return gSmearer_->Gaus(0.35,0.10);
                } else {
                    return gSmearer_->Gaus(0.47,0.20);
                }
            }
        } 
        if (abs(eta) < 1.5) {
            return (jetPtRatio == 1.0 ? 1.0 : jetPtRatio * 0.95);
        } else {
            // pull closer to central value
            return (jetPtRatio == 1.0 ? 1.0 : 0.95*jetPtRatio + 0.05*0.45);
        }
    }
    if (abs(eta) < 1.5) {
        if (jetPtRatio == 1) return 1;
        return jetPtRatio * 0.98;
    } else {
        if (jetPtRatio == 1) return 1;
        return 0.95*(jetPtRatio*0.972) + 0.05*0.8374; // pull closer to central value
    }
}

double correctJetDRMC(double jetDR, int pdgId, double pt, double eta, int mcMatchId, int mcMatchAny) {
    if (jetDR == 0.) return 0.;
    if (abs(pdgId) == 13) {
        if (mcMatchAny >= 2) return jetDR;
        if (pt > 15) { // muons, high pt
            if (abs(eta) < 1.5) {
                return jetDR * 1.01;
            } else {
                return log(1+jetDR*(eta*eta)*0.95)/(eta*eta); // pull closer to zero
            } 
        } else { // muons, low pt
            if (abs(eta) < 1.5) {
                return jetDR * 1.01;
            } else if (abs(eta) < 1.9) {
                return log(1+jetDR*0.5)/0.5; 
            } else {
                return log(1+jetDR*2*0.95)/2; 
            }
        }
    } else { // electrons
        if (mcMatchAny >= 2) return jetDR;
        //if (pt > 0) { // electrons, any pt for now
            if (abs(eta) < 1.5) {
                return jetDR * 1.01;
            } else if (abs(eta) < 2.0) {
                return log(1+jetDR*0.5)/0.5; 
            } else {
                return log(1+jetDR*5*0.95)/5; 
            } 
        //}  
    }
}

void mcCorrections() {}
