#ifndef COMMONTOOLS_PUPPI_PUPPICONTAINER_H_
#define COMMONTOOLS_PUPPI_PUPPICONTAINER_H_

#include "CommonTools/PileupModules/interface/PuppiAlgo.h"
#include "CommonTools/PileupModules/interface/RecoObj.h"
#include "fastjet/internal/base.hh"
#include "fastjet/PseudoJet.hh"

//FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

//......................
class PuppiContainer{
public:
    PuppiContainer(const edm::ParameterSet &iConfig);
    ~PuppiContainer(); 
    void initialize(const std::vector<RecoObj> &iRecoObjects);
    std::vector<fastjet::PseudoJet> const & pfParticles() const { return fPFParticles; }    
    std::vector<fastjet::PseudoJet> const & pvParticles() const { return fChargedPV; }        
    std::vector<double> const & puppiWeights() ;
    std::vector<fastjet::PseudoJet> const & puppiParticles() const { return fPupParticles;}

protected:
    double  goodVar      (fastjet::PseudoJet const &iPart,std::vector<fastjet::PseudoJet> const &iParts, int iOpt,double iRCone);
    void    getRMSAvg    (int iOpt,std::vector<fastjet::PseudoJet> const &iConstits,std::vector<fastjet::PseudoJet> const &iParticles,std::vector<fastjet::PseudoJet> const &iChargeParticles);
    double  getChi2FromdZ(double iDZ);
    int     getPuppiId   ( float iPt, float iEta);
    double  var_within_R (int iId, const std::vector<fastjet::PseudoJet> & particles, const fastjet::PseudoJet& centre, double R);  
    
    std::vector<RecoObj>   fRecoParticles;
    std::vector<fastjet::PseudoJet> fPFParticles;
    std::vector<fastjet::PseudoJet> fChargedPV;
    std::vector<fastjet::PseudoJet> fPupParticles;
    std::vector<double>    fWeights;
    std::vector<double>    fVals;
    bool   fApplyCHS;
    bool   fUseExp;
    double fNeutralMinPt;
    double fNeutralSlope;
    double fPuppiWeightCut;
    int    fNAlgos;
    int    fNPV;
    double fPVFrac;
    std::vector<PuppiAlgo> fPuppiAlgo;
};
#endif

