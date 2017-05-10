/* class PFRecoTauDiscriminationAgainstElectron
 * created : May 02 2008,
 * revised : ,
 * Authorss : Chi Nhan Nguyen (Texas A&M)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

using namespace reco;

class PFRecoTauDiscriminationAgainstElectron : public PFTauDiscriminationProducerBase  {
   public:
      explicit PFRecoTauDiscriminationAgainstElectron(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig) {

         emFraction_maxValue_              = iConfig.getParameter<double>("EmFraction_maxValue");
         applyCut_emFraction_              = iConfig.getParameter<bool>("ApplyCut_EmFraction");
         hcalTotOverPLead_minValue_        = iConfig.getParameter<double>("HcalTotOverPLead_minValue");
         applyCut_hcalTotOverPLead_        = iConfig.getParameter<bool>("ApplyCut_HcalTotOverPLead");
         hcalMaxOverPLead_minValue_        = iConfig.getParameter<double>("HcalMaxOverPLead_minValue");
         applyCut_hcalMaxOverPLead_        = iConfig.getParameter<bool>("ApplyCut_HcalMaxOverPLead");
         hcal3x3OverPLead_minValue_        = iConfig.getParameter<double>("Hcal3x3OverPLead_minValue");

         applyCut_hcal3x3OverPLead_        = iConfig.getParameter<bool>("ApplyCut_Hcal3x3OverPLead");
         EOverPLead_minValue_              = iConfig.getParameter<double>("EOverPLead_minValue");
         EOverPLead_maxValue_              = iConfig.getParameter<double>("EOverPLead_maxValue");
         applyCut_EOverPLead_              = iConfig.getParameter<bool>("ApplyCut_EOverPLead");
         bremsRecoveryEOverPLead_minValue_ = iConfig.getParameter<double>("BremsRecoveryEOverPLead_minValue");
         bremsRecoveryEOverPLead_maxValue_ = iConfig.getParameter<double>("BremsRecoveryEOverPLead_maxValue");

         applyCut_bremsRecoveryEOverPLead_ = iConfig.getParameter<bool>("ApplyCut_BremsRecoveryEOverPLead");

         applyCut_electronPreID_           = iConfig.getParameter<bool>("ApplyCut_ElectronPreID");

         applyCut_electronPreID_2D_        = iConfig.getParameter<bool>("ApplyCut_ElectronPreID_2D");

         elecPreID0_EOverPLead_maxValue_   = iConfig.getParameter<double>("ElecPreID0_EOverPLead_maxValue");
         elecPreID0_HOverPLead_minValue_   = iConfig.getParameter<double>("ElecPreID0_HOverPLead_minValue");
         elecPreID1_EOverPLead_maxValue_   = iConfig.getParameter<double>("ElecPreID1_EOverPLead_maxValue");
         elecPreID1_HOverPLead_minValue_   = iConfig.getParameter<double>("ElecPreID1_HOverPLead_minValue");


         applyCut_PFElectronMVA_           = iConfig.getParameter<bool>("ApplyCut_PFElectronMVA");
         pfelectronMVA_maxValue_           = iConfig.getParameter<double>("PFElectronMVA_maxValue");



         applyCut_ecalCrack_               = iConfig.getParameter<bool>("ApplyCut_EcalCrackCut");

         applyCut_bremCombined_            = iConfig.getParameter<bool>("ApplyCut_BremCombined");
         bremCombined_fraction_            = iConfig.getParameter<double>("BremCombined_Fraction");
         bremCombined_maxHOP_              = iConfig.getParameter<double>("BremCombined_HOP");
         bremCombined_minMass_             = iConfig.getParameter<double>("BremCombined_Mass");
         bremCombined_stripSize_           = iConfig.getParameter<double>("BremCombined_StripSize");

      }

      double discriminate(const PFTauRef& pfTau);

      ~PFRecoTauDiscriminationAgainstElectron(){}

   private:
      bool isInEcalCrack(double) const;
      edm::InputTag PFTauProducer_;
      bool applyCut_emFraction_;
      double emFraction_maxValue_;
      bool applyCut_hcalTotOverPLead_;
      double hcalTotOverPLead_minValue_;
      bool applyCut_hcalMaxOverPLead_;
      double hcalMaxOverPLead_minValue_;
      bool applyCut_hcal3x3OverPLead_;
      double hcal3x3OverPLead_minValue_;

      bool applyCut_EOverPLead_;
      double EOverPLead_minValue_;
      double EOverPLead_maxValue_;
      bool applyCut_bremsRecoveryEOverPLead_;
      double bremsRecoveryEOverPLead_minValue_;
      double bremsRecoveryEOverPLead_maxValue_;

      bool applyCut_electronPreID_;

      bool applyCut_electronPreID_2D_;
      double elecPreID0_EOverPLead_maxValue_;
      double elecPreID0_HOverPLead_minValue_;
      double elecPreID1_EOverPLead_maxValue_;
      double elecPreID1_HOverPLead_minValue_;

      bool applyCut_PFElectronMVA_;
      double pfelectronMVA_maxValue_;
      bool applyCut_ecalCrack_;

      bool   applyCut_bremCombined_;
      double bremCombined_fraction_;
      double bremCombined_maxHOP_;
      double bremCombined_minMass_;
      double bremCombined_stripSize_;



};

double PFRecoTauDiscriminationAgainstElectron::discriminate(const PFTauRef& thePFTauRef)
{



    // ensure tau has at least one charged object

    if( (*thePFTauRef).leadPFChargedHadrCand().isNull() )
    {
       return 0.;
    } else
    {
       // Check if track goes to Ecal crack
       TrackRef myleadTk;
       myleadTk=(*thePFTauRef).leadPFChargedHadrCand()->trackRef();
       math::XYZPointF myleadTkEcalPos = (*thePFTauRef).leadPFChargedHadrCand()->positionAtECALEntrance();
       if(myleadTk.isNonnull())
       {
          if (applyCut_ecalCrack_ && isInEcalCrack(myleadTkEcalPos.eta()))
          {
             return 0.;
          }
       }
    }

    bool decision = false;
    bool emfPass = true, htotPass = true, hmaxPass = true;
    bool h3x3Pass = true, estripPass = true, erecovPass = true;
    bool epreidPass = true, epreid2DPass = true;
    bool mvaPass = true, bremCombinedPass = true;

    if (applyCut_emFraction_) {
      if ((*thePFTauRef).emFraction() > emFraction_maxValue_) {
	emfPass = false;
      }
    }
    if (applyCut_hcalTotOverPLead_) {
      if ((*thePFTauRef).hcalTotOverPLead() < hcalTotOverPLead_minValue_) {
	htotPass = false;
      }
    }
    if (applyCut_hcalMaxOverPLead_) {
      if ((*thePFTauRef).hcalMaxOverPLead() < hcalMaxOverPLead_minValue_) {
	hmaxPass = false;
      }
    }
    if (applyCut_hcal3x3OverPLead_) {
      if ((*thePFTauRef).hcal3x3OverPLead() < hcal3x3OverPLead_minValue_) {
	h3x3Pass = false;
      }
    }
    if (applyCut_EOverPLead_) {
      if ((*thePFTauRef).ecalStripSumEOverPLead() > EOverPLead_minValue_ &&
	  (*thePFTauRef).ecalStripSumEOverPLead() < EOverPLead_maxValue_) {
	estripPass = false;
      } else {
	estripPass = true;
      }
    }
    if (applyCut_bremsRecoveryEOverPLead_) {
      if ((*thePFTauRef).bremsRecoveryEOverPLead() > bremsRecoveryEOverPLead_minValue_ &&
	  (*thePFTauRef).bremsRecoveryEOverPLead() < bremsRecoveryEOverPLead_maxValue_) {
	erecovPass = false;
      } else {
	erecovPass = true;
      }
    }
    if (applyCut_electronPreID_) {
      if ((*thePFTauRef).electronPreIDDecision()) {
	epreidPass = false;
      }  else {
	epreidPass = true;
      }
    }

    if (applyCut_electronPreID_2D_) {
      if (
	  ((*thePFTauRef).electronPreIDDecision() &&
	   ((*thePFTauRef).ecalStripSumEOverPLead() < elecPreID1_EOverPLead_maxValue_ ||
	    (*thePFTauRef).hcal3x3OverPLead() > elecPreID1_HOverPLead_minValue_))
	  ||
	  (!(*thePFTauRef).electronPreIDDecision() &&
	   ((*thePFTauRef).ecalStripSumEOverPLead() < elecPreID0_EOverPLead_maxValue_ ||
	    (*thePFTauRef).hcal3x3OverPLead() > elecPreID0_HOverPLead_minValue_))
	  ){
	epreid2DPass = true;
      }  else {
	epreid2DPass = false;
      }
    }

    if (applyCut_PFElectronMVA_) {
      if ((*thePFTauRef).electronPreIDOutput()>pfelectronMVA_maxValue_) {
	mvaPass = false;
      }
    }
    if (applyCut_bremCombined_) {
      if (thePFTauRef->leadPFChargedHadrCand()->trackRef().isNull()) {
        // No KF track found
        return 0;
      }
      if(thePFTauRef->signalPFChargedHadrCands().size()==1 && thePFTauRef->signalPFGammaCands().size()==0) {
	if(thePFTauRef->leadPFChargedHadrCand()->hcalEnergy()/thePFTauRef->leadPFChargedHadrCand()->trackRef()->p()<bremCombined_maxHOP_)
	  bremCombinedPass = false;
      }
      else if(thePFTauRef->signalPFChargedHadrCands().size()==1 && thePFTauRef->signalPFGammaCands().size()>0) {
	//calculate the brem ratio energy
	float bremEnergy=0.;
	float emEnergy=0.;
	for(unsigned int Nc = 0 ;Nc < thePFTauRef->signalPFGammaCands().size();++Nc)
	  {
	    PFCandidatePtr cand = thePFTauRef->signalPFGammaCands().at(Nc);
	    if(fabs(thePFTauRef->leadPFChargedHadrCand()->trackRef()->eta()-cand->eta())<bremCombined_stripSize_)
	      bremEnergy+=cand->energy();
	    emEnergy+=cand->energy();
	  }
	if(bremEnergy/emEnergy>bremCombined_fraction_&&thePFTauRef->mass()<bremCombined_minMass_)
	  bremCombinedPass = false;

      }
    }

    decision = emfPass && htotPass && hmaxPass &&
      h3x3Pass && estripPass && erecovPass && epreidPass && epreid2DPass && mvaPass &&bremCombinedPass;

    return (decision ? 1. : 0.);
}

bool
PFRecoTauDiscriminationAgainstElectron::isInEcalCrack(double eta) const
{
  eta = fabs(eta);
  return (eta < 0.018 ||
	  (eta>0.423 && eta<0.461) ||
	  (eta>0.770 && eta<0.806) ||
	  (eta>1.127 && eta<1.163) ||
	  (eta>1.460 && eta<1.558));
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectron);
