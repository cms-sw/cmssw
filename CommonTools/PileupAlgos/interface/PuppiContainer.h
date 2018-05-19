#ifndef COMMONTOOLS_PUPPI_PUPPICONTAINER_H_
#define COMMONTOOLS_PUPPI_PUPPICONTAINER_H_

#include "CommonTools/PileupAlgos/interface/PuppiAlgo.h"
#include "CommonTools/PileupAlgos/interface/RecoObj.h"
#include "CommonTools/PileupAlgos/interface/PuppiCandidate.h"

class PuppiContainer{
public:
    PuppiContainer(const edm::ParameterSet &iConfig);
    ~PuppiContainer(); 
    void initialize(const std::vector<RecoObj> &iRecoObjects);
    void setNPV(int iNPV){ fNPV = iNPV; }

    std::vector<PuppiCandidate> const & pfParticles() const { return fPFParticles; }
    std::vector<PuppiCandidate> const & pvParticles() const { return fChargedPV; }
    std::vector<double> const & puppiWeights();
    const std::vector<double> & puppiRawAlphas(){ return fRawAlphas; }
    const std::vector<double> & puppiAlphas(){ return fVals; }
    // const std::vector<double> puppiAlpha   () {return fAlpha;}
    const std::vector<double> & puppiAlphasMed() {return fAlphaMed;}
    const std::vector<double> & puppiAlphasRMS() {return fAlphaRMS;}

    int puppiNAlgos(){ return fNAlgos; }
    std::vector<PuppiCandidate> const & puppiParticles() const { return fPupParticles;}

protected:
    double  goodVar      (PuppiCandidate const &iPart,std::vector<PuppiCandidate> const &iParts, int iOpt,const double iRCone);
    void    getRMSAvg    (int iOpt,std::vector<PuppiCandidate> const &iConstits,std::vector<PuppiCandidate> const &iParticles,std::vector<PuppiCandidate> const &iChargeParticles);
    void    getRawAlphas    (int iOpt,std::vector<PuppiCandidate> const &iConstits,std::vector<PuppiCandidate> const &iParticles,std::vector<PuppiCandidate> const &iChargeParticles);
    double  getChi2FromdZ(double iDZ);
    int     getPuppiId   ( float iPt, float iEta);
    double  var_within_R (int iId, const std::vector<PuppiCandidate> & particles, const PuppiCandidate& centre, const double R);
    
    bool      fPuppiDiagnostics;
    std::vector<RecoObj>   fRecoParticles;
    std::vector<PuppiCandidate> fPFParticles;
    std::vector<PuppiCandidate> fChargedPV;
    std::vector<PuppiCandidate> fPupParticles;
    std::vector<double>    fWeights;
    std::vector<double>    fVals;
    std::vector<double>    fRawAlphas;
    std::vector<double>    fAlphaMed;
    std::vector<double>    fAlphaRMS;

    bool   fApplyCHS;
    bool   fInvert;
    bool   fUseExp;
    double fNeutralMinPt;
    double fNeutralSlope;
    double fPuppiWeightCut;
    double fPtMax;
    int    fNAlgos;
    int    fNPV;
    double fPVFrac;
    std::vector<PuppiAlgo> fPuppiAlgo;
};
#endif

