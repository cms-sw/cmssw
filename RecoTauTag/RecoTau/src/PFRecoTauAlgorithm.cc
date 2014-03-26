#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"

// Turn off filtering by pt.  Min pt is set to zero,
//  as this functionality is implemented in the underlying
//  PFTauTagInfo production.  Additional pt filters are applied
//  the discriminators.

#define PFTauAlgo_NeutrHadrCand_minPt_   (0.0)
#define PFTauAlgo_GammaCand_minPt_       (0.0)
#define PFTauAlgo_PFCand_minPt_          (0.0)
#define PFTauAlgo_Track_minPt_           (0.0)
#define PFTauAlgo_ChargedHadrCand_minPt_ (0.0)

using namespace reco;

PFRecoTauAlgorithm::PFRecoTauAlgorithm() : PFRecoTauAlgorithmBase(){}
PFRecoTauAlgorithm::PFRecoTauAlgorithm(const edm::ParameterSet& iConfig):PFRecoTauAlgorithmBase(iConfig){
   LeadPFCand_minPt_                   = iConfig.getParameter<double>("LeadPFCand_minPt");

   UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_
      = iConfig.getParameter<bool>("UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint");

   ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_
      = iConfig.getParameter<double>("ChargedHadrCandLeadChargedHadrCand_tksmaxDZ");

   LeadTrack_minPt_                    = iConfig.getParameter<double>("LeadTrack_minPt");
   UseTrackLeadTrackDZconstraint_      = iConfig.getParameter<bool>("UseTrackLeadTrackDZconstraint");
   TrackLeadTrack_maxDZ_               = iConfig.getParameter<double>("TrackLeadTrack_maxDZ");

   MatchingConeMetric_                 = iConfig.getParameter<std::string>("MatchingConeMetric");
   MatchingConeSizeFormula_            = iConfig.getParameter<std::string>("MatchingConeSizeFormula");
   MatchingConeSize_min_               = iConfig.getParameter<double>("MatchingConeSize_min");
   MatchingConeSize_max_               = iConfig.getParameter<double>("MatchingConeSize_max");
   TrackerSignalConeMetric_            = iConfig.getParameter<std::string>("TrackerSignalConeMetric");
   TrackerSignalConeSizeFormula_       = iConfig.getParameter<std::string>("TrackerSignalConeSizeFormula");
   TrackerSignalConeSize_min_          = iConfig.getParameter<double>("TrackerSignalConeSize_min");
   TrackerSignalConeSize_max_          = iConfig.getParameter<double>("TrackerSignalConeSize_max");
   TrackerIsolConeMetric_              = iConfig.getParameter<std::string>("TrackerIsolConeMetric");
   TrackerIsolConeSizeFormula_         = iConfig.getParameter<std::string>("TrackerIsolConeSizeFormula");
   TrackerIsolConeSize_min_            = iConfig.getParameter<double>("TrackerIsolConeSize_min");
   TrackerIsolConeSize_max_            = iConfig.getParameter<double>("TrackerIsolConeSize_max");
   ECALSignalConeMetric_               = iConfig.getParameter<std::string>("ECALSignalConeMetric");
   ECALSignalConeSizeFormula_          = iConfig.getParameter<std::string>("ECALSignalConeSizeFormula");
   ECALSignalConeSize_min_             = iConfig.getParameter<double>("ECALSignalConeSize_min");
   ECALSignalConeSize_max_             = iConfig.getParameter<double>("ECALSignalConeSize_max");
   ECALIsolConeMetric_                 = iConfig.getParameter<std::string>("ECALIsolConeMetric");
   ECALIsolConeSizeFormula_            = iConfig.getParameter<std::string>("ECALIsolConeSizeFormula");
   ECALIsolConeSize_min_               = iConfig.getParameter<double>("ECALIsolConeSize_min");
   ECALIsolConeSize_max_               = iConfig.getParameter<double>("ECALIsolConeSize_max");
   HCALSignalConeMetric_               = iConfig.getParameter<std::string>("HCALSignalConeMetric");
   HCALSignalConeSizeFormula_          = iConfig.getParameter<std::string>("HCALSignalConeSizeFormula");
   HCALSignalConeSize_min_             = iConfig.getParameter<double>("HCALSignalConeSize_min");
   HCALSignalConeSize_max_             = iConfig.getParameter<double>("HCALSignalConeSize_max");
   HCALIsolConeMetric_                 = iConfig.getParameter<std::string>("HCALIsolConeMetric");
   HCALIsolConeSizeFormula_            = iConfig.getParameter<std::string>("HCALIsolConeSizeFormula");
   HCALIsolConeSize_min_               = iConfig.getParameter<double>("HCALIsolConeSize_min");
   HCALIsolConeSize_max_               = iConfig.getParameter<double>("HCALIsolConeSize_max");

   putNeutralHadronsInP4_ = iConfig.exists("putNeutralHadronsInP4") ?
     iConfig.getParameter<bool>("putNeutralHadronsInP4") : false;

   // get paramaeters for ellipse EELL
   Rphi_				      = iConfig.getParameter<double>("Rphi");
   MaxEtInEllipse_		      = iConfig.getParameter<double>("MaxEtInEllipse");
   AddEllipseGammas_		      = iConfig.getParameter<bool>("AddEllipseGammas");
   // EELL

   AreaMetric_recoElements_maxabsEta_    = iConfig.getParameter<double>("AreaMetric_recoElements_maxabsEta");
   ChargedHadrCand_IsolAnnulus_minNhits_ = iConfig.getParameter<uint32_t>("ChargedHadrCand_IsolAnnulus_minNhits");
   Track_IsolAnnulus_minNhits_           = iConfig.getParameter<uint32_t>("Track_IsolAnnulus_minNhits");

   ElecPreIDLeadTkMatch_maxDR_           = iConfig.getParameter<double>("ElecPreIDLeadTkMatch_maxDR");
   EcalStripSumE_minClusEnergy_          = iConfig.getParameter<double>("EcalStripSumE_minClusEnergy");
   EcalStripSumE_deltaEta_               = iConfig.getParameter<double>("EcalStripSumE_deltaEta");
   EcalStripSumE_deltaPhiOverQ_minValue_ = iConfig.getParameter<double>("EcalStripSumE_deltaPhiOverQ_minValue");
   EcalStripSumE_deltaPhiOverQ_maxValue_ = iConfig.getParameter<double>("EcalStripSumE_deltaPhiOverQ_maxValue");
   maximumForElectrionPreIDOutput_       = iConfig.getParameter<double>("maximumForElectrionPreIDOutput");

   DataType_ = iConfig.getParameter<std::string>("DataType");

   //TFormula computation
   myMatchingConeSizeTFormula      = TauTagTools::computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
   //Charged particles cones
   myTrackerSignalConeSizeTFormula = TauTagTools::computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
   myTrackerIsolConeSizeTFormula   = TauTagTools::computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
   //Gamma candidates cones
   myECALSignalConeSizeTFormula    = TauTagTools::computeConeSizeTFormula(ECALSignalConeSizeFormula_,"ECAL signal cone size");
   myECALIsolConeSizeTFormula      = TauTagTools::computeConeSizeTFormula(ECALIsolConeSizeFormula_,"ECAL isolation cone size");
   //Neutral hadrons cones
   myHCALSignalConeSizeTFormula    = TauTagTools::computeConeSizeTFormula(HCALSignalConeSizeFormula_,"HCAL signal cone size");
   myHCALIsolConeSizeTFormula      = TauTagTools::computeConeSizeTFormula(HCALIsolConeSizeFormula_,"HCAL isolation cone size");
}


PFTau PFRecoTauAlgorithm::buildPFTau(const PFTauTagInfoRef& myPFTauTagInfoRef, const Vertex& myPV)
{
   PFJetRef myPFJet=(*myPFTauTagInfoRef).pfjetRef();  // catch a ref to the initial PFJet
   PFTau myPFTau(std::numeric_limits<int>::quiet_NaN(),myPFJet->p4());   // create the PFTau

   myPFTau.setpfTauTagInfoRef(myPFTauTagInfoRef);

   std::vector<PFCandidatePtr> myPFCands=(*myPFTauTagInfoRef).PFCands();

   PFTauElementsOperators myPFTauElementsOperators(myPFTau);
   double myMatchingConeSize=myPFTauElementsOperators.computeConeSize(myMatchingConeSizeTFormula,MatchingConeSize_min_,MatchingConeSize_max_);

   PFCandidatePtr myleadPFChargedCand=myPFTauElementsOperators.leadPFChargedHadrCand(MatchingConeMetric_,myMatchingConeSize,PFTauAlgo_PFCand_minPt_);

   // These two quantities always taken from the signal cone
   PFCandidatePtr myleadPFNeutralCand;
   PFCandidatePtr myleadPFCand;

   bool myleadPFCand_rectkavailable = false;
   double myleadPFCand_rectkDZ      = 0.;

   // Determine the SIPT of the lead track
   if(myleadPFChargedCand.isNonnull()) {
      myPFTau.setleadPFChargedHadrCand(myleadPFChargedCand);
      TrackRef myleadPFCand_rectk=(*myleadPFChargedCand).trackRef();
      if(myleadPFCand_rectk.isNonnull()) {
         myleadPFCand_rectkavailable=true;
         myleadPFCand_rectkDZ=(*myleadPFCand_rectk).dz(myPV.position());
         if(TransientTrackBuilder_!=0) {
            const TransientTrack myleadPFCand_rectransienttk=TransientTrackBuilder_->build(&(*myleadPFCand_rectk));
            GlobalVector myPFJetdir((*myPFJet).px(),(*myPFJet).py(),(*myPFJet).pz());
            if(IPTools::signedTransverseImpactParameter(myleadPFCand_rectransienttk,myPFJetdir,myPV).first)
               myPFTau.setleadPFChargedHadrCandsignedSipt(
                     IPTools::signedTransverseImpactParameter(myleadPFCand_rectransienttk,myPFJetdir,myPV).second.significance());
         }
      }
   }

   //Building PF Components
   if (myleadPFChargedCand.isNonnull())
   {
      math::XYZVector tauAxis = myleadPFChargedCand->momentum();
      // Compute energy of the PFTau considering only inner constituents
      // (inner == pfcandidates inside a cone which is equal to the maximum value of the signal cone)
      // The axis is built about the lead charged hadron
      std::vector<PFCandidatePtr> myTmpPFCandsInSignalCone =
         myPFTauElementsOperators.PFCandsInCone(tauAxis,TrackerSignalConeMetric_,TrackerSignalConeSize_max_,0.5);
      math::XYZTLorentzVector tmpLorentzVect(0.,0.,0.,0.);

      double jetOpeningAngle = 0.0;
      for (std::vector<PFCandidatePtr>::const_iterator iCand = myTmpPFCandsInSignalCone.begin();
            iCand != myTmpPFCandsInSignalCone.end(); iCand++)
      {
         //find the maximum opening angle of the jet (now a parameter in available TFormulas)
         double deltaRToSeed = TauTagTools::computeDeltaR(tauAxis, (**iCand).momentum());
         if (deltaRToSeed > jetOpeningAngle)
            jetOpeningAngle = deltaRToSeed;

         tmpLorentzVect+=(**iCand).p4();
      }

      //Setting the myPFTau four momentum as the one made from the signal cone constituents.
      double energy = tmpLorentzVect.energy();
      double transverseEnergy = tmpLorentzVect.pt();
      myPFTau.setP4(tmpLorentzVect);

      // Compute the cone sizes
      double myTrackerSignalConeSize = myPFTauElementsOperators.computeConeSize(
            myTrackerSignalConeSizeTFormula, TrackerSignalConeSize_min_, TrackerSignalConeSize_max_, transverseEnergy, energy, jetOpeningAngle);
      double myTrackerIsolConeSize = myPFTauElementsOperators.computeConeSize(
            myTrackerIsolConeSizeTFormula, TrackerIsolConeSize_min_, TrackerIsolConeSize_max_, transverseEnergy, energy, jetOpeningAngle);
      double myECALSignalConeSize = myPFTauElementsOperators.computeConeSize(
            myECALSignalConeSizeTFormula, ECALSignalConeSize_min_, ECALSignalConeSize_max_, transverseEnergy, energy, jetOpeningAngle);
      double myECALIsolConeSize = myPFTauElementsOperators.computeConeSize(
            myECALIsolConeSizeTFormula, ECALIsolConeSize_min_, ECALIsolConeSize_max_, transverseEnergy, energy, jetOpeningAngle);
      double myHCALSignalConeSize = myPFTauElementsOperators.computeConeSize(
            myHCALSignalConeSizeTFormula, HCALSignalConeSize_min_, HCALSignalConeSize_max_, transverseEnergy, energy, jetOpeningAngle);
      double myHCALIsolConeSize = myPFTauElementsOperators.computeConeSize(
            myHCALIsolConeSizeTFormula, HCALIsolConeSize_min_, HCALIsolConeSize_max_, transverseEnergy, energy, jetOpeningAngle);

      // Signal cone collections
      std::vector<PFCandidatePtr> mySignalPFChargedHadrCands, mySignalPFNeutrHadrCands, mySignalPFGammaCands, mySignalPFCands;

      if (UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_ && myleadPFCand_rectkavailable) {
         mySignalPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInCone(tauAxis,
               TrackerSignalConeMetric_, myTrackerSignalConeSize, PFTauAlgo_ChargedHadrCand_minPt_,
               ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_, myleadPFCand_rectkDZ, myPV);
      }
      else {
         mySignalPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInCone(tauAxis,
               TrackerSignalConeMetric_, myTrackerSignalConeSize, PFTauAlgo_ChargedHadrCand_minPt_);
      }

      // Set the Charged hadronics that live in the signal cones
      myPFTau.setsignalPFChargedHadrCands(mySignalPFChargedHadrCands);

      // Set the neurtral hadrons that live in the signal cone
      mySignalPFNeutrHadrCands=myPFTauElementsOperators.PFNeutrHadrCandsInCone(tauAxis,
            HCALSignalConeMetric_, myHCALSignalConeSize, PFTauAlgo_NeutrHadrCand_minPt_);

      myPFTau.setsignalPFNeutrHadrCands(mySignalPFNeutrHadrCands);

      // Compute the gammas that live in the signal cone
      mySignalPFGammaCands=myPFTauElementsOperators.PFGammaCandsInCone(tauAxis,
            ECALSignalConeMetric_,myECALSignalConeSize,PFTauAlgo_GammaCand_minPt_);

      myPFTau.setsignalPFGammaCands(mySignalPFGammaCands);

      // Add charged objects to signal cone, and calculate charge
      if(mySignalPFChargedHadrCands.size() != 0) {
         int mySignalPFChargedHadrCands_qsum=0;
         for(size_t i = 0; i < mySignalPFChargedHadrCands.size(); i++) {
            mySignalPFChargedHadrCands_qsum += mySignalPFChargedHadrCands[i]->charge();
            mySignalPFCands.push_back(mySignalPFChargedHadrCands[i]);
         }
         myPFTau.setCharge(mySignalPFChargedHadrCands_qsum);
      }

      //Add neutral objects to signal cone
      for(size_t i = 0; i < mySignalPFNeutrHadrCands.size(); i++) {
         mySignalPFCands.push_back(mySignalPFNeutrHadrCands[i]);
      }

      // For the signal gammas, keep track of the highest pt object
      double maxSignalGammaPt = 0.;
      for(size_t i = 0; i < mySignalPFGammaCands.size(); i++) {
         if(mySignalPFGammaCands[i]->pt() > maxSignalGammaPt) {
            myleadPFNeutralCand = mySignalPFGammaCands[i];
            maxSignalGammaPt = mySignalPFGammaCands[i]->pt();
         }
         mySignalPFCands.push_back(mySignalPFGammaCands[i]);
      }
      myPFTau.setsignalPFCands(mySignalPFCands);
      // Set leading gamma
      myPFTau.setleadPFNeutralCand(myleadPFNeutralCand);

      // Logic to determine lead PFCand.  If the lead charged object
      // is above the threshold, take that.  If the lead charged object is less
      // than the threshold (but exists), AND there exists a gamma above the threshold
      // take the gamma as the leadPFCand.  Otherwise it is null.

      if(myleadPFChargedCand->pt() > LeadPFCand_minPt_) {
         myPFTau.setleadPFCand(myleadPFChargedCand);
      } else if (maxSignalGammaPt > LeadPFCand_minPt_) {
         myPFTau.setleadPFCand(myleadPFNeutralCand);
      }

      // Declare isolation collections
      std::vector<PFCandidatePtr> myUnfilteredIsolPFChargedHadrCands, myIsolPFNeutrHadrCands, myIsolPFGammaCands, myIsolPFCands;

      // Build unfiltered isolation collection
      if(UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_ && myleadPFCand_rectkavailable) {
         myUnfilteredIsolPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInAnnulus(
               tauAxis,TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,
               PFTauAlgo_ChargedHadrCand_minPt_,ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_,myleadPFCand_rectkDZ, myPV);
      } else {
         myUnfilteredIsolPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInAnnulus(
               tauAxis,TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,
               PFTauAlgo_ChargedHadrCand_minPt_);
      }

      // Filter isolation annulus charge dhadrons with additional nHits quality cut
      // (note that other cuts [pt, chi2, are already cut on])
      std::vector<PFCandidatePtr> myIsolPFChargedHadrCands;
      myIsolPFChargedHadrCands = TauTagTools::filteredPFChargedHadrCandsByNumTrkHits(
            myUnfilteredIsolPFChargedHadrCands, ChargedHadrCand_IsolAnnulus_minNhits_);

      myPFTau.setisolationPFChargedHadrCands(myIsolPFChargedHadrCands);

      // Fill neutral hadrons
      myIsolPFNeutrHadrCands = myPFTauElementsOperators.PFNeutrHadrCandsInAnnulus(
            tauAxis, HCALSignalConeMetric_, myHCALSignalConeSize, HCALIsolConeMetric_,
            myHCALIsolConeSize, PFTauAlgo_NeutrHadrCand_minPt_);
      myPFTau.setisolationPFNeutrHadrCands(myIsolPFNeutrHadrCands);

      // Fill gamma candidates
      myIsolPFGammaCands = myPFTauElementsOperators.PFGammaCandsInAnnulus(
            tauAxis, ECALSignalConeMetric_, myECALSignalConeSize, ECALIsolConeMetric_,
            myECALIsolConeSize, PFTauAlgo_GammaCand_minPt_);
      myPFTau.setisolationPFGammaCands(myIsolPFGammaCands);

      //Incorporate converted gammas from isolation ellipse into signal  ... ELLL
      //Get pair with in/out elements using the isoPFGammaCandidates set by default
      if(AddEllipseGammas_) {
         double rPhi;
         if(Rphi_ >= 1.)
            rPhi = Rphi_*myECALSignalConeSize;
         else
            rPhi = Rphi_;

         std::pair<std::vector<PFCandidatePtr>,std::vector<PFCandidatePtr>> elementsInOutEllipse =
            myPFTauElementsOperators.PFGammaCandsInOutEllipse(myIsolPFGammaCands, *myleadPFCand, rPhi, myECALSignalConeSize, MaxEtInEllipse_);

         std::vector<PFCandidatePtr> elementsInEllipse = elementsInOutEllipse.first;
         std::vector<PFCandidatePtr> elementsOutEllipse = elementsInOutEllipse.second;
         //add the inside elements to signal PFCandidates and reset signal PFCands
         for(std::vector<PFCandidatePtr>::const_iterator inEllipseIt = elementsInEllipse.begin(); inEllipseIt != elementsInEllipse.end(); inEllipseIt++){
            mySignalPFCands.push_back(*inEllipseIt);
            mySignalPFGammaCands.push_back(*inEllipseIt);
         }
         myPFTau.setsignalPFCands(mySignalPFCands);
         //redefine isoPFGammaCandidates to be the outside elements
         myIsolPFGammaCands=elementsOutEllipse;
         myPFTau.setisolationPFGammaCands(myIsolPFGammaCands);
      }


      // Fill isolation collections, and calculate pt sum in isolation cone
      float myIsolPFChargedHadrCands_Ptsum = 0.;
      float myIsolPFGammaCands_Etsum       = 0.;
      for(size_t i = 0; i < myIsolPFChargedHadrCands.size(); i++) {
         myIsolPFChargedHadrCands_Ptsum += myIsolPFChargedHadrCands[i]->pt();
         myIsolPFCands.push_back(myIsolPFChargedHadrCands[i]);
      }
      myPFTau.setisolationPFChargedHadrCandsPtSum(myIsolPFChargedHadrCands_Ptsum);

      // Put neutral hadrons into collection
      for(size_t i = 0; i < myIsolPFNeutrHadrCands.size(); i++) {
         myIsolPFCands.push_back(myIsolPFNeutrHadrCands[i]);
      }

      for(size_t i = 0; i < myIsolPFGammaCands.size(); i++) {
         myIsolPFGammaCands_Etsum += myIsolPFGammaCands[i]->et();
         myIsolPFCands.push_back(myIsolPFGammaCands[i]);
      }
      myPFTau.setisolationPFGammaCandsEtSum(myIsolPFGammaCands_Etsum);
      myPFTau.setisolationPFCands(myIsolPFCands);

      //Making the alternateLorentzVector, i.e. direction with only signal components
      math::XYZTLorentzVector alternatLorentzVect(0.,0.,0.,0.);
      for (std::vector<PFCandidatePtr>::const_iterator iGammaCand = mySignalPFGammaCands.begin();
            iGammaCand != mySignalPFGammaCands.end(); iGammaCand++) {
         alternatLorentzVect+=(**iGammaCand).p4();
      }

      for (std::vector<PFCandidatePtr>::const_iterator iChargedHadrCand = mySignalPFChargedHadrCands.begin();
            iChargedHadrCand != mySignalPFChargedHadrCands.end(); iChargedHadrCand++) {
         alternatLorentzVect+=(**iChargedHadrCand).p4();
      }
      // Alternate lorentz std::vector is always charged + gammas
      myPFTau.setalternatLorentzVect(alternatLorentzVect);

      // Optionally add the neutral hadrons to the p4
      if (putNeutralHadronsInP4_) {
        for (std::vector<PFCandidatePtr>::const_iterator iNeutralHadrCand = mySignalPFNeutrHadrCands.begin();
            iNeutralHadrCand != mySignalPFNeutrHadrCands.end(); iNeutralHadrCand++) {
          alternatLorentzVect+=(**iNeutralHadrCand).p4();
        }
      }
      myPFTau.setP4(alternatLorentzVect);

      // Set tau vertex as PV vertex
      myPFTau.setVertex(math::XYZPoint(myPV.x(), myPV.y(), myPV.z()));
   }

   // set the leading, signal cone and isolation annulus Tracks (the initial list of Tracks was catched through a JetTracksAssociation
   // object, not through the charged hadr. PFCandidates inside the PFJet ;
   // the motivation for considering these objects is the need for checking that a selection by the
   // charged hadr. PFCandidates is equivalent to a selection by the rec. Tracks.)
   TrackRef myleadTk=myPFTauElementsOperators.leadTk(MatchingConeMetric_,myMatchingConeSize,LeadTrack_minPt_);
   myPFTau.setleadTrack(myleadTk);
   if(myleadTk.isNonnull()){
      double myleadTkDZ = (*myleadTk).dz(myPV.position());
      double myTrackerSignalConeSize=myPFTauElementsOperators.computeConeSize(myTrackerSignalConeSizeTFormula,TrackerSignalConeSize_min_,TrackerSignalConeSize_max_);
      double myTrackerIsolConeSize=myPFTauElementsOperators.computeConeSize(myTrackerIsolConeSizeTFormula,TrackerIsolConeSize_min_,TrackerIsolConeSize_max_);
      if (UseTrackLeadTrackDZconstraint_){
         myPFTau.setsignalTracks(myPFTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,PFTauAlgo_Track_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ, myPV));

         TrackRefVector myUnfilteredTracks = myPFTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,PFTauAlgo_Track_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ, myPV);
         TrackRefVector myFilteredTracks = TauTagTools::filteredTracksByNumTrkHits(myUnfilteredTracks, Track_IsolAnnulus_minNhits_);
         myPFTau.setisolationTracks(myFilteredTracks);

      }else{
         myPFTau.setsignalTracks(myPFTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,PFTauAlgo_Track_minPt_));

         TrackRefVector myUnfilteredTracks = myPFTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,PFTauAlgo_Track_minPt_);
         TrackRefVector myFilteredTracks = TauTagTools::filteredTracksByNumTrkHits(myUnfilteredTracks, Track_IsolAnnulus_minNhits_);
         myPFTau.setisolationTracks(myFilteredTracks);
      }
   }


   /* For elecron rejection */
   double myECALenergy             =  0.;
   double myHCALenergy             =  0.;
   double myHCALenergy3x3          =  0.;
   double myMaximumHCALPFClusterE  =  0.;
   double myMaximumHCALPFClusterEt =  0.;
   double myStripClusterE          =  0.;
   double myEmfrac                 = -1.;
   double myElectronPreIDOutput    = -1111.;
   bool   myElecPreid              =  false;
   reco::TrackRef myElecTrk;

   typedef std::pair<reco::PFBlockRef, unsigned> ElementInBlock;
   typedef std::vector< ElementInBlock > ElementsInBlocks;

   //Use the electron rejection only in case there is a charged leading pion
   if(myleadPFChargedCand.isNonnull()){
      myElectronPreIDOutput = myleadPFChargedCand->mva_e_pi();

      math::XYZPointF myElecTrkEcalPos = myleadPFChargedCand->positionAtECALEntrance();
      myElecTrk = myleadPFChargedCand->trackRef();//Electron candidate

      if(myElecTrk.isNonnull()) {
         //FROM AOD
         if(DataType_ == "AOD"){
            // Corrected Cluster energies
            for(int i=0;i<(int)myPFCands.size();i++){
               myHCALenergy += myPFCands[i]->hcalEnergy();
               myECALenergy += myPFCands[i]->ecalEnergy();

               math::XYZPointF candPos;
               if (myPFCands[i]->particleId()==1 || myPFCands[i]->particleId()==2)//if charged hadron or electron
                  candPos = myPFCands[i]->positionAtECALEntrance();
               else
                  candPos = math::XYZPointF(myPFCands[i]->px(),myPFCands[i]->py(),myPFCands[i]->pz());

               double deltaR   = ROOT::Math::VectorUtil::DeltaR(myElecTrkEcalPos,candPos);
               double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(myElecTrkEcalPos,candPos);
               double deltaEta = std::abs(myElecTrkEcalPos.eta()-candPos.eta());
               double deltaPhiOverQ = deltaPhi/(double)myElecTrk->charge();

               if (myPFCands[i]->ecalEnergy() >= EcalStripSumE_minClusEnergy_ && deltaEta < EcalStripSumE_deltaEta_ &&
                     deltaPhiOverQ > EcalStripSumE_deltaPhiOverQ_minValue_  && deltaPhiOverQ < EcalStripSumE_deltaPhiOverQ_maxValue_) {
                  myStripClusterE += myPFCands[i]->ecalEnergy();
               }
               if (deltaR<0.184) {
                  myHCALenergy3x3 += myPFCands[i]->hcalEnergy();
               }
               if (myPFCands[i]->hcalEnergy()>myMaximumHCALPFClusterE) {
                  myMaximumHCALPFClusterE = myPFCands[i]->hcalEnergy();
               }
               if ((myPFCands[i]->hcalEnergy()*fabs(sin(candPos.Theta())))>myMaximumHCALPFClusterEt) {
                  myMaximumHCALPFClusterEt = (myPFCands[i]->hcalEnergy()*fabs(sin(candPos.Theta())));
               }
            }

         } else if(DataType_ == "RECO"){ //From RECO
            // Against double counting of clusters
            std::vector<math::XYZPoint> hcalPosV; hcalPosV.clear();
            std::vector<math::XYZPoint> ecalPosV; ecalPosV.clear();
            for(int i=0;i<(int)myPFCands.size();i++){
               const ElementsInBlocks& elts = myPFCands[i]->elementsInBlocks();
               for(ElementsInBlocks::const_iterator it=elts.begin(); it!=elts.end(); ++it) {
                  const reco::PFBlock& block = *(it->first);
                  unsigned indexOfElementInBlock = it->second;
                  const edm::OwnVector< reco::PFBlockElement >& elements = block.elements();
                  assert(indexOfElementInBlock<elements.size());

                  const reco::PFBlockElement& element = elements[indexOfElementInBlock];

                  if(element.type()==reco::PFBlockElement::HCAL) {
                     math::XYZPoint clusPos = element.clusterRef()->position();
                     double en = (double)element.clusterRef()->energy();
                     double et = (double)element.clusterRef()->energy()*fabs(sin(clusPos.Theta()));
                     if (en>myMaximumHCALPFClusterE) {
                        myMaximumHCALPFClusterE = en;
                     }
                     if (et>myMaximumHCALPFClusterEt) {
                        myMaximumHCALPFClusterEt = et;
                     }
                     if (!checkPos(hcalPosV,clusPos)) {
                        hcalPosV.push_back(clusPos);
                        myHCALenergy += en;
                        double deltaR = ROOT::Math::VectorUtil::DeltaR(myElecTrkEcalPos,clusPos);
                        if (deltaR<0.184) {
                           myHCALenergy3x3 += en;
                        }
                     }
                  } else if(element.type()==reco::PFBlockElement::ECAL) {
                     double en = (double)element.clusterRef()->energy();
                     math::XYZPoint clusPos = element.clusterRef()->position();
                     if (!checkPos(ecalPosV,clusPos)) {
                        ecalPosV.push_back(clusPos);
                        myECALenergy += en;
                        double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(myElecTrkEcalPos,clusPos);
                        double deltaEta = std::abs(myElecTrkEcalPos.eta()-clusPos.eta());
                        double deltaPhiOverQ = deltaPhi/(double)myElecTrk->charge();
                        if (en >= EcalStripSumE_minClusEnergy_ && deltaEta<EcalStripSumE_deltaEta_ && deltaPhiOverQ > EcalStripSumE_deltaPhiOverQ_minValue_ && deltaPhiOverQ < EcalStripSumE_deltaPhiOverQ_maxValue_) {
                           myStripClusterE += en;
                        }
                     }
                  }
               } //end elements in blocks
            } //end loop over PFcands
         } //end RECO case
      } // end check for null electrk
   } // end check for null pfChargedHadrCand

   if ((myHCALenergy+myECALenergy)>0.)
      myEmfrac = myECALenergy/(myHCALenergy+myECALenergy);
   myPFTau.setemFraction((float)myEmfrac);

   // scale the appropriate quantities by the momentum of the electron if it exists
   if (myElecTrk.isNonnull())
   {
      float myElectronMomentum = (float)myElecTrk->p();
      if (myElectronMomentum > 0.)
      {
         myHCALenergy            /= myElectronMomentum;
         myMaximumHCALPFClusterE /= myElectronMomentum;
         myHCALenergy3x3         /= myElectronMomentum;
         myStripClusterE         /= myElectronMomentum;
      }
   }
   myPFTau.sethcalTotOverPLead((float)myHCALenergy);
   myPFTau.sethcalMaxOverPLead((float)myMaximumHCALPFClusterE);
   myPFTau.sethcal3x3OverPLead((float)myHCALenergy3x3);
   myPFTau.setecalStripSumEOverPLead((float)myStripClusterE);
   myPFTau.setmaximumHCALPFClusterEt(myMaximumHCALPFClusterEt);
   myPFTau.setelectronPreIDOutput(myElectronPreIDOutput);
   if (myElecTrk.isNonnull())
      myPFTau.setelectronPreIDTrack(myElecTrk);
   if (myElectronPreIDOutput > maximumForElectrionPreIDOutput_)
      myElecPreid = true;
   myPFTau.setelectronPreIDDecision(myElecPreid);

   // These need to be filled!
   //myPFTau.setbremsRecoveryEOverPLead(my...);

   /* End elecron rejection */

   return myPFTau;
}

bool
PFRecoTauAlgorithm::checkPos(const std::vector<math::XYZPoint>& CalPos,const math::XYZPoint& CandPos) const{
   bool flag = false;
   for (unsigned int i=0;i<CalPos.size();i++) {
      if (CalPos[i] == CandPos) {
         flag = true;
         break;
      }
   }
   return flag;
   //return false;
}
