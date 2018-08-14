#ifndef DataFormats_BTauReco_PFCombinedTauTagInfo_h
#define DataFormats_BTauReco_PFCombinedTauTagInfo_h

/* class PFCombinedTauTagInfo
 *  Extended object for the Particle Flow Tau Combination algorithm, 
 *  created: Apr 21 2007,
 *  revised: Jun 23 2007,
 *  author: Ludovic Houchu.
 */

#include <limits>
#include <math.h>

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/PFIsolatedTauTagInfo.h"




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
      isolPFChargedHadrCands_.clear();
      leadPFChargedHadrCandsignedSipt_=NAN;
      leadPFChargedHadrCandsignedSip3D_=NAN;
      signedSflightpath_=NAN;
      PFChargedHadrCandsEtJetEtRatio_=NAN;
      PFNeutrHadrCandsE_=NAN;
      PFNeutrHadrCandsN_=std::numeric_limits<int>::quiet_NaN();
      PFNeutrHadrCandsRadius_=NAN;
      PFGammaCandsE_=NAN;
      isolPFGammaCandsE_=NAN;
      PFGammaCandsN_=std::numeric_limits<int>::quiet_NaN();
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
    
    //the reference to the PFJet;
    const PFJetRef& pfjetRef()const{return PFJetRef_;}
    void setpfjetRef(const PFJetRef x){PFJetRef_=x;}

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
    
    //the PF charged hadron candidates inside isolation band;
    const PFCandidateRefVector& isolPFChargedHadrCands()const{return isolPFChargedHadrCands_;}
    void setisolPFChargedHadrCands(const PFCandidateRefVector& x) {isolPFChargedHadrCands_=x;}
    
    virtual PFCombinedTauTagInfo* clone() const{return new PFCombinedTauTagInfo(*this );}
    
    // float JetTag::discriminator() returns 0.        if candidate did not pass PF charged hadron candidates selection,   
    //                                       1.        if candidate passed PF charged hadron candidates selection and did not contain PF gamma candidate(s),   
    //                                       0<=  <=1  if candidate passed PF charged hadron candidates selection, contained PF gamma candidate(s) and went through the likelihood ratio mechanism,   
    //                                       NaN       the values of the likelihood functions PDFs are 0 (test the result of discriminator() with bool isnan(.));   
    //computed with the parameters taken from the RecoTauTag/PFCombinedTauTag/data/ .cfi files
    
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

   // PF neutral hadron candidates E sum;
   double PFNeutrHadrCandsE()const{return(PFNeutrHadrCandsE_);} 
   void setPFNeutrHadrCandsE(double x){PFNeutrHadrCandsE_=x;}

   int PFNeutrHadrCandsN()const{return(PFNeutrHadrCandsN_);}
   void setPFNeutrHadrCandsN(int x){PFNeutrHadrCandsN_=x;}

   //mean DR_PFNeutrHadrcands-lead.PFcand.;
   double PFNeutrHadrCandsRadius()const{return(PFNeutrHadrCandsRadius_);} // NaN : PFNeutrHadrCandsN()=0;
   void setPFNeutrHadrCandsRadius(double x){PFNeutrHadrCandsRadius_=x;}

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
   PFJetRef PFJetRef_;
   PFIsolatedTauTagInfoRef PFIsolatedTauTagInfoRef_;
   PFCandidateRefVector PFChargedHadrCands_;
   PFCandidateRefVector selectedPFChargedHadrCands_;
   PFCandidateRefVector signalPFChargedHadrCands_;
   PFCandidateRefVector isolPFChargedHadrCands_;
   bool candidate_selectedByPFChargedHadrCands_;
   bool electronTagged_;
   bool muonTagged_;
   double leadPFChargedHadrCandsignedSipt_;
   double leadPFChargedHadrCandsignedSip3D_;
   double signedSflightpath_;
   double PFChargedHadrCandsEtJetEtRatio_;
   double PFNeutrHadrCandsE_;
   int PFNeutrHadrCandsN_;
   double PFNeutrHadrCandsRadius_;
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

  DECLARE_EDM_REFS( PFCombinedTauTagInfo )

}

#endif // DataFormats_BTauReco_PFCombinedTauTagInfo_h
