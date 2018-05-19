#ifndef COMMONTOOLS_PUPPI_PUPPICONTAINER_H_
#define COMMONTOOLS_PUPPI_PUPPICONTAINER_H_

#include "CommonTools/PileupAlgos/interface/PuppiAlgo.h"
#include "CommonTools/PileupAlgos/interface/RecoObj.h"
#include "fastjet/internal/base.hh"
#include "fastjet/PseudoJet.hh"

//FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

const double pseudojet_invalid_eta = -1e200;

//......................
class PuppiContainer{
public:
  

  // Helper class designed to store Puppi information inside of fastjet pseudojets.
  // In CMSSW we use the user_index to refer to the index of the input collection, 
  // but Puppi uses it to decide between NHs, PV CHs, and PU CHs. Instead,
  // make that a register. 
  class PuppiUserInfo : public fastjet::PseudoJet::UserInfoBase {
   public : 
     PuppiUserInfo( int puppi_register = -1) : puppi_register_(puppi_register) {}
     ~PuppiUserInfo() override{}
  
     void set_puppi_register(int i) { puppi_register_ = i; }
  
     inline int puppi_register() const { return puppi_register_; }
  
   protected : 
     int puppi_register_;     /// Used by puppi algorithm to decide neutrals vs PV vs PU
  };


  // Extension of fastjet::PseudoJet that caches eta
  class PseudoJet : public fastjet::PseudoJet {
	public:
      using fastjet::PseudoJet::PseudoJet;
      double pseudorapidity() const { _ensure_valid_eta(); return _eta; }
      double eta() const { return pseudorapidity(); }
      void _ensure_valid_eta() const { if(_eta==pseudojet_invalid_eta) _eta = fastjet::PseudoJet::pseudorapidity(); }
    private:
      mutable double _eta = pseudojet_invalid_eta;
  };


    PuppiContainer(const edm::ParameterSet &iConfig);
    ~PuppiContainer(); 
    void initialize(const std::vector<RecoObj> &iRecoObjects);
    void setNPV(int iNPV){ fNPV = iNPV; }

    std::vector<PseudoJet> const & pfParticles() const { return fPFParticles; }
    std::vector<PseudoJet> const & pvParticles() const { return fChargedPV; }
    std::vector<double> const & puppiWeights();
    const std::vector<double> & puppiRawAlphas(){ return fRawAlphas; }
    const std::vector<double> & puppiAlphas(){ return fVals; }
    // const std::vector<double> puppiAlpha   () {return fAlpha;}
    const std::vector<double> & puppiAlphasMed() {return fAlphaMed;}
    const std::vector<double> & puppiAlphasRMS() {return fAlphaRMS;}

    int puppiNAlgos(){ return fNAlgos; }
    std::vector<PseudoJet> const & puppiParticles() const { return fPupParticles;}

protected:
    double  goodVar      (PseudoJet const &iPart,std::vector<PseudoJet> const &iParts, int iOpt,const double iRCone);
    void    getRMSAvg    (int iOpt,std::vector<PseudoJet> const &iConstits,std::vector<PseudoJet> const &iParticles,std::vector<PseudoJet> const &iChargeParticles);
    void    getRawAlphas    (int iOpt,std::vector<PseudoJet> const &iConstits,std::vector<PseudoJet> const &iParticles,std::vector<PseudoJet> const &iChargeParticles);
    double  getChi2FromdZ(double iDZ);
    int     getPuppiId   ( float iPt, float iEta);
    double  var_within_R (int iId, const std::vector<PseudoJet> & particles, const PseudoJet& centre, const double R);
    
    bool      fPuppiDiagnostics;
    std::vector<RecoObj>   fRecoParticles;
    std::vector<PseudoJet> fPFParticles;
    std::vector<PseudoJet> fChargedPV;
    std::vector<PseudoJet> fPupParticles;
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

