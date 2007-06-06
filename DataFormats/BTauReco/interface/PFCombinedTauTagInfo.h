#ifndef BTauReco_PFTauTagCombination_h
#define BTauReco_PFTauTagCombination_h

/* class PFCombinedTauTagInfo
 *  Extended object for the Particle Flow Tau Combination algorithm, 
 *  created: Apr 21 2007,
 *  revised: May 08 2007,
 *  author: Ludovic Houchu.
 */

#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/PFCombinedTauTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/PFIsolatedTauTagInfo.h"

#include "DataFormats/JetReco/interface/GenericJet.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <limits>
#include <math.h>

using namespace edm;
using namespace std;

namespace reco { 
  class PFCombinedTauTagInfo : public BaseTagInfo{
  public:
    PFCombinedTauTagInfo() {
      candidate_selectedByPFChargedHadrCands_=false;
      electronTagged_=false;
      muonTagged_=false;
      PFChargedHadrCands_.clear();
      selectedPFChargedHadrCands_.clear();
      signalPFChargedHadrCands_.clear();
      leadPFChargedHadrCandsignedSipt_=NAN;
      leadPFChargedHadrCandsignedSip3D_=NAN;
      signedSflightpath_=NAN;
      PFChargedHadrCandsEtJetEtRatio_=NAN;
      PFGammaCandsE_=NAN;
      isolPFGammaCandsE_=NAN;
      PFGammaCandsN_=numeric_limits<int>::quiet_NaN();
      PFGammaCandsRadius_=NAN;
      PFGammaCandsEJetalternatERatio_=NAN;
      isolPFGammaCandsEJetalternatERatio_=NAN;
      PFGammaCandsERatio_=NAN;
      alternatLorentzVect_.SetPx(NAN);
      alternatLorentzVect_.SetPy(NAN);
      alternatLorentzVect_.SetPz(NAN);
      alternatLorentzVect_.SetE(NAN);
      ECALEtleadPFChargedHadrCandPtRatio_=NAN;
      HCALEtleadPFChargedHadrCandPtRatio_=NAN;
    }
    virtual ~PFCombinedTauTagInfo() {};
    
    //the reference to the GenericJet;
    const GenericJetRef& genericjetRef()const{return GenericJetRef_;}
    void setgenericjetRef(const GenericJetRef x){GenericJetRef_=x;}

    //the reference to the PFIsolatedTauTagInfo;
    const PFIsolatedTauTagInfoRef& isolatedtautaginfoRef()const{return PFIsolatedTauTagInfoRef_;}
    void setisolatedtautaginfoRef(const PFIsolatedTauTagInfoRef x) {PFIsolatedTauTagInfoRef_=x;}
    
    //the PF charged hadron candidates contained in the PF Jet;
    const PFCandidateRefVector& PFChargedHadrCands()const{return PFChargedHadrCands_;}
    void setPFChargedHadrCands(const PFCandidateRefVector& x) {PFChargedHadrCands_=x;}
    
    //the PF charged hadron candidates considered in the isolation strip and signal cone selections;
    const PFCandidateRefVector& selectedPFChargedHadrCands()const{return selectedPFChargedHadrCands_;}
    void setselectedPFChargedHadrCands(const PFCandidateRefVector& x) {selectedPFChargedHadrCands_=x;}
    
    //the PF charged hadron candidates inside signal cone;
    const PFCandidateRefVector& signalPFChargedHadrCands()const{return signalPFChargedHadrCands_;}
    void setsignalPFChargedHadrCands(const PFCandidateRefVector& x) {signalPFChargedHadrCands_=x;}
    
    virtual PFCombinedTauTagInfo* clone() const{return new PFCombinedTauTagInfo(*this );}
    
    // float discriminator() returns 0.        if candidate did not pass PF charged hadron candidates selection,   
    //                               1.        if candidate passed PF charged hadron candidates selection and did not contain PF gamma candidate(s),   
    //                               0<=  <=1  if candidate passed PF charged hadron candidates selection, contained PF gamma candidate(s) and went through the likelihood ratio mechanism,   
    //                               NaN       the values of the likelihood functions PDFs are 0 (test the result of discriminator() with bool isnan(.));   
   
    //default discriminator, computed with the parameters taken from the RecoTauTag/PFCombinedTauTag/data/ .cfi files
    float discriminator()const{return discriminator_;}
    void setdiscriminator(double x){discriminator_=x;}
    
    bool selectedByPFChargedHadrCands()const{return(candidate_selectedByPFChargedHadrCands_);}
    void setselectedByPFChargedHadrCands(bool x){candidate_selectedByPFChargedHadrCands_=x;}

   bool electronTagged()const{return(electronTagged_);} // true : passed PF charged hadron candidates sel., contains 1 signal charged hadron candidate, e-identified through (ECALEtleadPFChargedHadrCandPtRatio(),HCALEtleadPFChargedHadrCandPtRatio()) space;
   void setelectronTagged(bool x){electronTagged_=x;} 

   bool muonTagged()const{return(muonTagged_);} // true : passed PF charged hadron candidates sel., contains 1 signal charged hadron candidate, mu-identified through (ECALEtleadPFChargedHadrCandPtRatio(),HCALEtleadPFChargedHadrCandPtRatio()) space;
   void setmuonTagged(bool x){muonTagged_=x;}

   double leadPFChargedHadrCandsignedSipt()const{return (leadPFChargedHadrCandsignedSipt_);}  // NaN : failure;
   void setleadPFChargedHadrCandsignedSipt(double x){leadPFChargedHadrCandsignedSipt_=x;}

   double leadPFChargedHadrCandsignedSip3D()const{return(leadPFChargedHadrCandsignedSip3D_);}  // NaN : failure;
   void setleadPFChargedHadrCandsignedSip3D(double x){leadPFChargedHadrCandsignedSip3D_=x;}

   double signedSflightpath()const{return (signedSflightpath_);}  // NaN : failure, did not build a SV.;
   void setsignedSflightpath(double x){signedSflightpath_=x;}
   
   // Et_PFchargedhadrcands/Etjet;
   double PFChargedHadrCandsEtJetEtRatio()const{return(PFChargedHadrCandsEtJetEtRatio_);} 
   void setPFChargedHadrCandsEtJetEtRatio(double x){PFChargedHadrCandsEtJetEtRatio_=x;}

   // PF gamma candidates E sum;
   double PFGammaCandsE()const{return(PFGammaCandsE_);} 
   void setPFGammaCandsE(double x){PFGammaCandsE_=x;}

   // isol. band PF gamma candidates E sum;
   double isolPFGammaCandsE()const{return(isolPFGammaCandsE_);} 
   void setisolPFGammaCandsE(double x){isolPFGammaCandsE_=x;}

   int PFGammaCandsN()const{return(PFGammaCandsN_);}
   void setPFGammaCandsN(int x){PFGammaCandsN_=x;}

   //mean DR_PFgammacands-lead.PFcand.;
   double PFGammaCandsRadius()const{return(PFGammaCandsRadius_);} // NaN : PFGammaCandsN()=0;
   void setPFGammaCandsRadius(double x){PFGammaCandsRadius_=x;}

   // E_PFgammacands / (E_PFgammacands + E_PFchargedhadrcands);
   double PFGammaCandsEJetalternatERatio()const{return(PFGammaCandsEJetalternatERatio_);} 
   void setPFGammaCandsEJetalternatERatio(double x){PFGammaCandsEJetalternatERatio_=x;} 

   // E_PFgammacands,isol.band / (E_PFgammacands + E_PFchargedhadrcands);
   double isolPFGammaCandsEJetalternatERatio()const{return(isolPFGammaCandsEJetalternatERatio_);} 
   void setisolPFGammaCandsEJetalternatERatio(double x){isolPFGammaCandsEJetalternatERatio_=x;}

   // E_PFgammacands,isol.band / E_PFgammacands;
   double PFGammaCandsERatio()const{return(PFGammaCandsERatio_);} // NaN : PFGammaCandsN()=0;
   void setPFGammaCandsERatio(double x){PFGammaCandsERatio_=x;}

   math::XYZTLorentzVector alternatLorentzVect()const{return(alternatLorentzVect_);} // rec. charged hadr. candidates + rec. gamma candidates combined;   
   void setalternatLorentzVect(math::XYZTLorentzVector x){alternatLorentzVect_=x;}

   // EtECAL*/Pt_lead.PFcand.        *using ECAL cell hits inside a DR cone around lead. charged hadr. candidate ECAL impact point direction;
   double ECALEtleadPFChargedHadrCandPtRatio()const{return(ECALEtleadPFChargedHadrCandPtRatio_);} // NaN : failure when trying to find the lead. charged hadr. candidate contact on ECAL surface point; 
   void setECALEtleadPFChargedHadrCandPtRatio(double x){ECALEtleadPFChargedHadrCandPtRatio_=x;}

   // EtHCAL**/Pt_lead.PFcand.;      **using HCAL tower hits inside a DR cone around lead. charged hadr. candidate ECAL impact point direction; 
   double HCALEtleadPFChargedHadrCandPtRatio()const{return(HCALEtleadPFChargedHadrCandPtRatio_);} // NaN : failure when trying to find the lead. charged hadr. candidate contact on ECAL surface point; 
   void setHCALEtleadPFChargedHadrCandPtRatio(double x){HCALEtleadPFChargedHadrCandPtRatio_=x;}
 private:
   GenericJetRef GenericJetRef_;
   PFIsolatedTauTagInfoRef PFIsolatedTauTagInfoRef_;
   PFCandidateRefVector PFChargedHadrCands_;
   PFCandidateRefVector selectedPFChargedHadrCands_;
   PFCandidateRefVector signalPFChargedHadrCands_;
   double discriminator_;
   bool candidate_selectedByPFChargedHadrCands_;
   bool electronTagged_;
   bool muonTagged_;
   double leadPFChargedHadrCandsignedSipt_;
   double leadPFChargedHadrCandsignedSip3D_;
   double signedSflightpath_;
   double PFChargedHadrCandsEtJetEtRatio_;
   double PFGammaCandsE_;
   double isolPFGammaCandsE_;
   int PFGammaCandsN_;
   double PFGammaCandsRadius_;
   double PFGammaCandsEJetalternatERatio_;
   double isolPFGammaCandsEJetalternatERatio_;
   double PFGammaCandsERatio_;
   math::XYZTLorentzVector alternatLorentzVect_;
   double ECALEtleadPFChargedHadrCandPtRatio_;
   double HCALEtleadPFChargedHadrCandPtRatio_;
 };
}
#endif
