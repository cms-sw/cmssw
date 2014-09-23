#ifndef COMMONTOOLS_PUPPI_PUPPICONTAINER_H_
#define COMMONTOOLS_PUPPI_PUPPICONTAINER_H_

#include "CommonTools/Puppi/interface/PuppiAlgo.h"
#include "CommonTools/Puppi/interface/NoTrees.hh"
#include "CommonTools/Puppi/interface/RecoObj.hh"
#include "fastjet/internal/base.hh"
#include "fastjet/PseudoJet.hh"

using namespace std;
using namespace fastjet;

//FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

//......................
class PuppiContainer{
public:
    PuppiContainer(const edm::ParameterSet &iConfig);
    ~PuppiContainer(); 
    void initialize(const std::vector<RecoObj> &iRecoObjects);
    std::vector<fastjet::PseudoJet> pfParticles(){ return fPFParticles; }    
    std::vector<fastjet::PseudoJet> pvParticles(){ return fChargedPV; }        
    const std::vector<double> puppiWeights();
    std::vector<fastjet::PseudoJet> puppiParticles() { return fPupParticles;}

protected:
    double  goodVar      (PseudoJet &iPart,std::vector<PseudoJet> &iParts, int iOpt,double iRCone);
    void    getRMSAvg    (int iOpt,std::vector<fastjet::PseudoJet> &iConstits,std::vector<fastjet::PseudoJet> &iParticles,std::vector<fastjet::PseudoJet> &iChargeParticles);
    double  getChi2FromdZ(double iDZ);
    int     getPuppiId   (const float &iPt,const float &iEta);
    double  var_within_R (int iId, const vector<PseudoJet> & particles, const PseudoJet& centre, double R);  
    
    std::vector<RecoObj>   fRecoParticles;
    std::vector<PseudoJet> fPFParticles;
    std::vector<PseudoJet> fChargedPV;
    std::vector<PseudoJet> fPupParticles;
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

