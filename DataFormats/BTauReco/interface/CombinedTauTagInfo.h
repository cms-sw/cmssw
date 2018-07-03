#ifndef DataFormats_BTauReco_CombinedTauTagInfo_h
#define DataFormats_BTauReco_CombinedTauTagInfo_h

/* class CombinedTauTagInfo
 *  Extended object for the Tau Combination algorithm, 
 *  created: Dec 18 2006,
 *  revised: Jul 02 2007
 *  author: Ludovic Houchu.
 */

#include <limits>
#include <cmath>
#include "CLHEP/Vector/LorentzVector.h"
#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

namespace reco {

  class CombinedTauTagInfo : public JTATagInfo {
  public:
    CombinedTauTagInfo(){}
    CombinedTauTagInfo(const JetTracksAssociationRef& jtaRef) : JTATagInfo(jtaRef) {
      thecandidate_passed_trackerselection=false;
      thecandidate_is_GoodTauCandidate=false;
      thecandidate_is_infact_GoodElectronCandidate=false;
      thecandidate_is_infact_GoodMuonCandidate=false;
      thecandidate_needs_LikelihoodRatio_discrimination=false;
      filtered_Tks_.clear();
      signal_Tks_.clear();
      isol_Tks_.clear();
      theleadTk_signedipt_significance=NAN;
      theleadTk_signedip3D_significance=NAN;
      thesignedflightpath_significance=NAN;
      theTksEt_o_JetEt=NAN;
      theneutralE=NAN;
      theisolneutralE=NAN;
      theisolneutralEtsum=NAN;
      theneutralECALClus_number=std::numeric_limits<int>::quiet_NaN();
      theneutralECALClus_radius=NAN;
      theneutralE_o_TksEneutralE=NAN;
      theisolneutralE_o_TksEneutralE=NAN;
      theneutralE_ratio=NAN;
      thealternatrecJet_HepLV.setPx(NAN);
      thealternatrecJet_HepLV.setPy(NAN);
      thealternatrecJet_HepLV.setPz(NAN);
      thealternatrecJet_HepLV.setE(NAN);
      theECALEt_o_leadTkPt=NAN;
      theHCALEt_o_leadTkPt=NAN;
    }
    ~CombinedTauTagInfo() override {};
    
    // float discriminator() returns 0.        if candidate did not pass tracker selection,   
    //                               1.        if candidate passed tracker selection and did not contain neutral ECAL clus.,   
    //                               0<=  <=1  if candidate passed tracker selection, contained neutral ECAL clus. and went through the likelihood ratio mechanism,   
    //                               NaN       the values of the likelihood functions PDFs are 0 (test the result of discriminator() with bool isnan(.));   
    
    //the reference to the IsolatedTauTagInfo
    const IsolatedTauTagInfoRef & isolatedtautaginfoRef() const { return IsolatedTauTagInfoRef_; }
    void setisolatedtautaginfoRef(const IsolatedTauTagInfoRef & x) { IsolatedTauTagInfoRef_ = x; }
   
    //get the tracks from the JetTag
    const TrackRefVector& allTks() const { return m_jetTracksAssociation->second; }
   
    //the tracks considered in the isolation strip and signal cone selections
    const TrackRefVector& selectedTks() const { return filtered_Tks_; }
    void setselectedTks(const TrackRefVector& x) { filtered_Tks_=x; }
    
    //the tracks in the signal cone
    const TrackRefVector& signalTks() const { return signal_Tks_; }
    void setsignalTks(const TrackRefVector& x) { signal_Tks_=x; }
   
    int signalTks_qsum()const{              // NaN : (int)(signal_Tks_.size())=0;   
      int signal_Tks_qsum_=std::numeric_limits<int>::quiet_NaN();   
      if((int)(signal_Tks_.size())!=0){   
	signal_Tks_qsum_=0;   
	for(TrackRefVector::const_iterator iTk=signal_Tks_.begin();iTk!=signal_Tks_.end();iTk++){   
	  signal_Tks_qsum_+=(**iTk).charge();   
	}   
      }   
      return signal_Tks_qsum_;   
    } 
    
    //the tracks in the isolation band
    const TrackRefVector& isolTks() const { return isol_Tks_; }
    void setisolTks(const TrackRefVector& x) { isol_Tks_=x; }
     
    CombinedTauTagInfo* clone() const override{return new CombinedTauTagInfo(*this );}
    
    bool passed_trackerselection()const{return(thecandidate_passed_trackerselection);}
    void setpassed_trackerselection(bool x){thecandidate_passed_trackerselection=x;}

   bool is_GoodTauCandidate()const{return(thecandidate_is_GoodTauCandidate);} // true : passed tracker sel. and no neutral activity inside jet;
   void setis_GoodTauCandidate(bool x){thecandidate_is_GoodTauCandidate=x;}

   bool infact_GoodElectronCandidate()const{return(thecandidate_is_infact_GoodElectronCandidate);} // true : passed tracker sel., contains 1 signal tk, e-identified through (ECALEt_o_leadTkPt(),HCALEt_o_leadTkPt()) space;
   void setinfact_GoodElectronCandidate(bool x){thecandidate_is_infact_GoodElectronCandidate=x;} 

   bool infact_GoodMuonCandidate()const{return(thecandidate_is_infact_GoodMuonCandidate);} // true : passed tracker sel., contains 1 signal tk, mu-identified through (ECALEt_o_leadTkPt(),HCALEt_o_leadTkPt()) space;
   void setinfact_GoodMuonCandidate(bool x){thecandidate_is_infact_GoodMuonCandidate=x;}

   bool needs_LikelihoodRatio_discrimination()const{return(thecandidate_needs_LikelihoodRatio_discrimination);} // true : passed tracker sel. and neutral activity inside jet;
   void setneeds_LikelihoodRatio_discrimination(bool x){thecandidate_needs_LikelihoodRatio_discrimination=x;}

   double leadTk_signedipt_significance()const{return (theleadTk_signedipt_significance);}  // NaN : failure;
   void setleadTk_signedipt_significance(double x){theleadTk_signedipt_significance=x;}

   double leadTk_signedip3D_significance()const{return(theleadTk_signedip3D_significance);}  // NaN : failure;
   void setleadTk_signedip3D_significance(double x){theleadTk_signedip3D_significance=x;}

   double signedflightpath_significance()const{return (thesignedflightpath_significance);}  // NaN : failure, did not build a SV.;
   void setsignedflightpath_significance(double x){thesignedflightpath_significance=x;}
   
   // Ettks/Etjet;
   double TksEt_o_JetEt()const{return(theTksEt_o_JetEt);} 
   void setTksEt_o_JetEt(double x){theTksEt_o_JetEt=x;}

   // Eneutr.clus.;
   double neutralE()const{return(theneutralE);} 
   void setneutralE(double x){theneutralE=x;}

   // Eneutr.clus.,isol.band;
   double isolneutralE()const{return(theisolneutralE);} 
   void setisolneutralE(double x){theisolneutralE=x;}

   // sum of Etneutr.clus.,isol.band;
   double isolneutralEtsum()const{return(theisolneutralEtsum);} 
   void setisolneutralEtsum(double x){theisolneutralEtsum=x;}

   int neutralECALClus_number()const{return(theneutralECALClus_number);}
   void setneutralECALClus_number(int x){theneutralECALClus_number=x;}

   //mean DRneutr.clus.-lead.tk
   double neutralECALClus_radius()const{return(theneutralECALClus_radius);} // NaN : neutralECALClus_number()=0;
   void setneutralECALClus_radius(double x){theneutralECALClus_radius=x;}

   // Eneutr.clus. / (Eneutr.clus. + Etks) , Etks built with tks impulsion and charged pi mass hypothesis; 
   double neutralE_o_TksEneutralE()const{return(theneutralE_o_TksEneutralE);} 
   void setneutralE_o_TksEneutralE(double x){theneutralE_o_TksEneutralE=x;} 

   // Eneutr.clus.,isol.band / (Eneutr.clus. + Etks);
   double isolneutralE_o_TksEneutralE()const{return(theisolneutralE_o_TksEneutralE);} 
   void setisolneutralE_o_TksEneutralE(double x){theisolneutralE_o_TksEneutralE=x;}

   // Eneutr.clus.,isol.band / Eneutr.clus.;
   double neutralE_ratio()const{return(theneutralE_ratio);} // NaN : neutralECALClus_number()=0;
   void setneutralE_ratio(double x){theneutralE_ratio=x;}

   CLHEP::HepLorentzVector alternatrecJet_HepLV()const{return(thealternatrecJet_HepLV);} // rec. pi+/- candidates + neutral ECAL clus. combined;   
   void setalternatrecJet_HepLV(CLHEP::HepLorentzVector x){thealternatrecJet_HepLV=x;}

   // EtECAL*/Ptlead.tk        *using ECAL cell hits inside a DR cone around lead tk ECAL impact point direction;
   double ECALEt_o_leadTkPt()const{return(theECALEt_o_leadTkPt);} // NaN : failure when trying to find the lead. tk contact on ECAL surface point; 
   void setECALEt_o_leadTkPt(double x){theECALEt_o_leadTkPt=x;}

   // EtHCAL**/Ptlead.tk;      **using HCAL tower hits inside a DR cone around lead tk ECAL impact point direction; 
   double HCALEt_o_leadTkPt()const{return(theHCALEt_o_leadTkPt);} // NaN : failure when trying to find the lead. tk contact on ECAL surface point; 
   void setHCALEt_o_leadTkPt(double x){theHCALEt_o_leadTkPt=x;}
 private:
   IsolatedTauTagInfoRef IsolatedTauTagInfoRef_;
   TrackRefVector filtered_Tks_;
   TrackRefVector signal_Tks_;
   TrackRefVector isol_Tks_;
   bool thecandidate_passed_trackerselection;
   bool thecandidate_is_GoodTauCandidate;
   bool thecandidate_is_infact_GoodElectronCandidate;
   bool thecandidate_is_infact_GoodMuonCandidate;
   bool thecandidate_needs_LikelihoodRatio_discrimination;
   double theleadTk_signedipt_significance;
   double theleadTk_signedip3D_significance;
   double thesignedflightpath_significance;
   double theTksEt_o_JetEt;
   double theneutralE;
   double theisolneutralE;
   double theisolneutralEtsum;
   int theneutralECALClus_number;
   double theneutralECALClus_radius;
   double theneutralE_o_TksEneutralE;
   double theisolneutralE_o_TksEneutralE;
   double theneutralE_ratio;
   CLHEP::HepLorentzVector thealternatrecJet_HepLV;
   double theECALEt_o_leadTkPt;
   double theHCALEt_o_leadTkPt;
 };

 DECLARE_EDM_REFS( CombinedTauTagInfo )

}

#endif // DataFormsts_BTauReco_CombinedTauTagInfo_h
