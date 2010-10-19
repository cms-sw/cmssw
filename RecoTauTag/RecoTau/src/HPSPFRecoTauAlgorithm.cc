#include "RecoTauTag/RecoTau/interface/HPSPFRecoTauAlgorithm.h"
#include "Math/GenVector/VectorUtil.h"
using namespace reco;

HPSPFRecoTauAlgorithm::HPSPFRecoTauAlgorithm():
    PFRecoTauAlgorithmBase()
{
}

HPSPFRecoTauAlgorithm::HPSPFRecoTauAlgorithm(const edm::ParameterSet& config):
    PFRecoTauAlgorithmBase(config)
{
  configure(config);
}

HPSPFRecoTauAlgorithm::~HPSPFRecoTauAlgorithm()
{
  if(candidateMerger_ !=0 ) delete candidateMerger_;
}

PFTau
HPSPFRecoTauAlgorithm::buildPFTau(const PFTauTagInfoRef& tagInfo,const Vertex& vertex)
{
  PFTau pfTau;

  pfTaus_.clear();
  //make the strips globally.
  std::vector<PFCandidateRefVector> strips = candidateMerger_->mergeCandidates(tagInfo->PFCands());


  //Sort the hadrons globally and once!
  PFCandidateRefVector hadrons = tagInfo->PFChargedHadrCands();
  if(hadrons.size()>1)
    TauTagTools::sortRefVectorByPt(hadrons);


  //OK For this Tau Tag Info we should create all the possible taus

  //One Prongs
  if(doOneProngs_)
    buildOneProng(tagInfo,hadrons);

  //One Prong Strips
  if(doOneProngStrips_)
    buildOneProngStrip(tagInfo,strips,hadrons);

  //One Prong TwoStrips
  if(doOneProngTwoStrips_)
    buildOneProngTwoStrips(tagInfo,strips,hadrons);

  //Three Prong
  if(doThreeProngs_)
    buildThreeProngs(tagInfo,hadrons);


  //Lets see if we created any taus
  if(pfTaus_.size()>0) {

    pfTau = getBestTauCandidate(pfTaus_);

    //Set the IP for the leading track
    if(TransientTrackBuilder_!=0 &&pfTau.leadPFChargedHadrCand()->trackRef().isNonnull()) {
      const TransientTrack leadTrack=TransientTrackBuilder_->build(pfTau.leadPFChargedHadrCand()->trackRef());
      if(pfTau.pfTauTagInfoRef().isNonnull())
        if(pfTau.pfTauTagInfoRef()->pfjetRef().isNonnull()) {
          PFJetRef jet = pfTau.pfTauTagInfoRef()->pfjetRef();
          GlobalVector jetDir(jet->px(),jet->py(),jet->pz());
          if(IPTools::signedTransverseImpactParameter(leadTrack,jetDir,vertex).first)
            pfTau.setleadPFChargedHadrCandsignedSipt(IPTools::signedTransverseImpactParameter(leadTrack,jetDir,vertex).second.significance());
        }
    }
  }
  else {      //null PFTau
    //Simone asked that in case there is no tau returned make a tau
    //without refs and the LV of the jet
    pfTau.setpfTauTagInfoRef(tagInfo);
    pfTau.setP4(tagInfo->pfjetRef()->p4());
  }

  return pfTau;
}



//Create one prong tau
void
HPSPFRecoTauAlgorithm::buildOneProng(const reco::PFTauTagInfoRef& tagInfo,const reco::PFCandidateRefVector& hadrons)
{
  PFTauCollection  taus;

  if(hadrons.size()>0)
    for(unsigned int h=0;h<hadrons.size();++h) {
      PFCandidateRef hadron = hadrons.at(h);

      //In the one prong case the lead Track pt should be above the tau Threshold!
      //since all the tau is just one track!
      if(hadron->pt()>tauThreshold_)
        if(hadron->pt()>leadPionThreshold_)
          //The track should be within the matching cone
          if(ROOT::Math::VectorUtil::DeltaR(hadron->p4(),tagInfo->pfjetRef()->p4())<matchingCone_) {
            //OK Lets create a Particle Flow Tau!
            PFTau tau = PFTau(hadron->charge(),hadron->p4(),hadron->vertex());

            //Associate the Tag Info to the tau
            tau.setpfTauTagInfoRef(tagInfo);

            //Put the Hadron in the signal Constituents
            PFCandidateRefVector signal;
            signal.push_back(hadron);

            //Set The signal candidates of the PF Tau
            tau.setsignalPFChargedHadrCands(signal);
            tau.setsignalPFCands(signal);
            tau.setleadPFChargedHadrCand(hadron);
            tau.setleadPFCand(hadron);

            //Fill isolation variables
            associateIsolationCandidates(tau,0.0);

            //Apply Muon rejection algorithms
            applyMuonRejection(tau);
            applyElectronRejection(tau,0.0);

            //Save this candidate
            taus.push_back(tau);
          }
    }
  if(taus.size()>0) {
    pfTaus_.push_back(getBestTauCandidate(taus));
  }

}

//Build one Prong + Strip

void
HPSPFRecoTauAlgorithm::buildOneProngStrip(const reco::PFTauTagInfoRef& tagInfo,const  std::vector<PFCandidateRefVector>& strips,const reco::PFCandidateRefVector & hadrons)

{
  //Create output Collection
  PFTauCollection taus;




  //make taus like this only if there is at least one hadron+ 1 strip
  if(hadrons.size()>0&&strips.size()>0){
    //Combinatorics between strips and clusters
    for(std::vector<PFCandidateRefVector>::const_iterator candVector=strips.begin();candVector!=strips.end();++candVector)
      for(PFCandidateRefVector::const_iterator hadron=hadrons.begin();hadron!=hadrons.end();++hadron) {

        //First Cross cleaning ! If you asked to clusterize the candidates
        //with tracks too then you should not double count the track
        PFCandidateRefVector emConstituents = *candVector;
        removeCandidateFromRefVector(*hadron,emConstituents);

        //Create a LorentzVector for the strip
        math::XYZTLorentzVector strip = createMergedLorentzVector(emConstituents);

        //TEST: Apply Strip Constraint
        applyMassConstraint(strip,0.1349);

        //create the Particle Flow Tau: Hadron plus Strip
        PFTau tau((*hadron)->charge(),
                  (*hadron)->p4()+strip,
                  (*hadron)->vertex());

        //Check tau threshold,  mass, Matching Cone window
        if(tau.pt()>tauThreshold_&&strip.pt()>stripPtThreshold_)
          if(tau.mass()>oneProngStripMassWindow_.at(0)&&tau.mass()<oneProngStripMassWindow_.at(1))//Apply mass window
            if(ROOT::Math::VectorUtil::DeltaR(tau.p4(),tagInfo->pfjetRef()->p4())<matchingCone_) { //Apply matching cone
              //Set The Tag Infor ref
              tau.setpfTauTagInfoRef(tagInfo);

              //Create the signal vectors
              PFCandidateRefVector signal;
              PFCandidateRefVector signalH;
              PFCandidateRefVector signalG;

              //Store the hadron in the PFTau
              signalH.push_back(*hadron);
              signal.push_back(*hadron);

              //calculate the cone size : For the strip use it as one candidate !
              double tauCone=0.0;
              if(coneMetric_ =="angle")
                tauCone=std::max(fabs(ROOT::Math::VectorUtil::Angle(tau.p4(),(*hadron)->p4())),
                                 fabs(ROOT::Math::VectorUtil::Angle(tau.p4(),strip)));
              else if(coneMetric_ == "DR")
                tauCone=std::max(ROOT::Math::VectorUtil::DeltaR(tau.p4(),(*hadron)->p4()),
                                 ROOT::Math::VectorUtil::DeltaR(tau.p4(),strip));

              if(emConstituents.size()>0)
                for(PFCandidateRefVector::const_iterator j=emConstituents.begin();j!=emConstituents.end();++j)  {
                  signal.push_back(*j);
                  signalG.push_back(*j);
                }

              //Set the PFTau
              tau.setsignalPFChargedHadrCands(signalH);
              tau.setsignalPFGammaCands(signalG);
              tau.setsignalPFCands(signal);
              tau.setleadPFChargedHadrCand(*hadron);
              tau.setleadPFNeutralCand(emConstituents.at(0));

              //Set the lead Candidate->Can be the hadron or the leading PFGamma(When we clear the Dataformat we will put the strip)
              if((*hadron)->pt()>emConstituents.at(0)->pt())
                tau.setleadPFCand(*hadron);
              else
                tau.setleadPFCand(emConstituents.at(0));

              //Apply the signal cone size formula
              if(isNarrowTau(tau,tauCone)) {
                //calculate the isolation Deposits
                associateIsolationCandidates(tau,tauCone);
                //Set Muon Rejection
                applyMuonRejection(tau);
                applyElectronRejection(tau,strip.energy());

                taus.push_back(tau);
              }
            }
      }
  }

  if(taus.size()>0) {
    pfTaus_.push_back(getBestTauCandidate(taus));
  }

}

void
HPSPFRecoTauAlgorithm::buildOneProngTwoStrips(const reco::PFTauTagInfoRef& tagInfo,const  std::vector<PFCandidateRefVector>& strips,const reco::PFCandidateRefVector& hadrons)
{


  PFTauCollection taus;

  //make taus like this only if there is at least one hadron+ 2 strips
  if(hadrons.size()>0&&strips.size()>1){
    //Combinatorics between strips and clusters
    for(unsigned int Nstrip1=0;Nstrip1<strips.size()-1;++Nstrip1)
      for(unsigned int Nstrip2=Nstrip1+1;Nstrip2<strips.size();++Nstrip2)
        for(PFCandidateRefVector::const_iterator hadron=hadrons.begin();hadron!=hadrons.end();++hadron) {



          //Create the strips and the vectors .Again cross clean the track if associated
          PFCandidateRefVector emConstituents1 = strips.at(Nstrip1);
          PFCandidateRefVector emConstituents2 = strips.at(Nstrip2);
          removeCandidateFromRefVector(*hadron,emConstituents1);
          removeCandidateFromRefVector(*hadron,emConstituents2);


          //Create a LorentzVector for the strip
          math::XYZTLorentzVector strip1 = createMergedLorentzVector(emConstituents1);
          math::XYZTLorentzVector strip2 = createMergedLorentzVector(emConstituents2);



          //Apply Mass Constraints
          applyMassConstraint(strip1,0.0);
          applyMassConstraint(strip2,0.0);


          PFTau tau((*hadron)->charge(),
                    (*hadron)->p4()+strip1+strip2,
                    (*hadron)->vertex());


          if(tau.pt()>tauThreshold_&&strip1.pt()>stripPtThreshold_&&strip2.pt()>stripPtThreshold_)
            if((strip1+strip2).M() >oneProngTwoStripsPi0MassWindow_.at(0) &&(strip1+strip2).M() <oneProngTwoStripsPi0MassWindow_.at(1) )//pi0 conmstraint for two strips
              if(tau.mass()>oneProngTwoStripsMassWindow_.at(0)&&tau.mass()<oneProngTwoStripsMassWindow_.at(1))//Apply mass window
                if(ROOT::Math::VectorUtil::DeltaR(tau.p4(),tagInfo->pfjetRef()->p4())<matchingCone_) { //Apply matching cone
                  //create the PFTau
                  tau.setpfTauTagInfoRef(tagInfo);


                  //Create the signal vectors
                  PFCandidateRefVector signal;
                  PFCandidateRefVector signalH;
                  PFCandidateRefVector signalG;

                  //Store the hadron in the PFTau
                  signalH.push_back(*hadron);
                  signal.push_back(*hadron);

                  //calculate the cone size from the reconstructed Objects
                  double tauCone=1000.0;
                  if(coneMetric_ =="angle") {
                    tauCone=std::max(std::max(fabs(ROOT::Math::VectorUtil::Angle(tau.p4(),(*hadron)->p4())),
                                              fabs(ROOT::Math::VectorUtil::Angle(tau.p4(),strip1))),
                                              fabs(ROOT::Math::VectorUtil::Angle(tau.p4(),strip2)));
                  }
                  else if(coneMetric_ =="DR") {
                    tauCone=std::max(std::max((ROOT::Math::VectorUtil::DeltaR(tau.p4(),(*hadron)->p4())),
                                              (ROOT::Math::VectorUtil::DeltaR(tau.p4(),strip1))),
                                              (ROOT::Math::VectorUtil::DeltaR(tau.p4(),strip2)));

                  }

                  for(PFCandidateRefVector::const_iterator j=emConstituents1.begin();j!=emConstituents1.end();++j)  {
                    signal.push_back(*j);
                    signalG.push_back(*j);
                  }

                  for(PFCandidateRefVector::const_iterator j=emConstituents2.begin();j!=emConstituents2.end();++j)  {
                    signal.push_back(*j);
                    signalG.push_back(*j);
                  }

                  //Set the PFTau
                  tau.setsignalPFChargedHadrCands(signalH);
                  tau.setsignalPFGammaCands(signalG);
                  tau.setsignalPFCands(signal);
                  tau.setleadPFChargedHadrCand(*hadron);

                  //Set the lead Candidate->Can be the hadron or the leading PFGamma(When we clear the Dataformat we will put the strip)
                  if((*hadron)->pt()>emConstituents1.at(0)->pt())
                    tau.setleadPFCand(*hadron);
                  else
                    tau.setleadPFCand(emConstituents1.at(0));

                  //Apply the cone size formula
                  if(isNarrowTau(tau,tauCone)) {

                    //calculate the isolation Deposits
                    associateIsolationCandidates(tau,tauCone);

                    applyMuonRejection(tau);

                    //For two strips take the nearest strip to the track
                    if(ROOT::Math::VectorUtil::DeltaR(strip1,(*hadron)->p4())<
                       ROOT::Math::VectorUtil::DeltaR(strip2,(*hadron)->p4()))
                      applyElectronRejection(tau,strip1.energy());
                    else
                      applyElectronRejection(tau,strip2.energy());

                    taus.push_back(tau);
                  }
                }
        }
  }

  if(taus.size()>0) {
    pfTaus_.push_back(getBestTauCandidate(taus));
  }
}



void
HPSPFRecoTauAlgorithm::buildThreeProngs(const reco::PFTauTagInfoRef& tagInfo,const reco::PFCandidateRefVector& hadrons)
{
  PFTauCollection taus;
  //get Hadrons


  //Require at least three hadrons
  if(hadrons.size()>2)
    for(unsigned int a=0;a<hadrons.size()-2;++a)
      for(unsigned int b=a+1;b<hadrons.size()-1;++b)
        for(unsigned int c=b+1;c<hadrons.size();++c) {

          PFCandidateRef h1 = hadrons.at(a);
          PFCandidateRef h2 = hadrons.at(b);
          PFCandidateRef h3 = hadrons.at(c);

          //check charge Compatibility and lead track
          int charge=h1->charge()+h2->charge()+h3->charge();
          if(std::abs(charge)==1 && h1->pt()>leadPionThreshold_)
            //check the track refs
            if(h1->trackRef()!=h2->trackRef()&&h1->trackRef()!=h3->trackRef()&&h2->trackRef()!=h3->trackRef())
            {

              //create the tau
              PFTau tau = PFTau(charge,h1->p4()+h2->p4()+h3->p4(),h1->vertex());
              tau.setpfTauTagInfoRef(tagInfo);

              if(tau.pt()>tauThreshold_)//Threshold
                if(ROOT::Math::VectorUtil::DeltaR(tau.p4(),tagInfo->pfjetRef()->p4())<matchingCone_) {//Matching Cone

                  PFCandidateRefVector signal;
                  signal.push_back(h1);
                  signal.push_back(h2);
                  signal.push_back(h3);
                  //calculate the tau cone by getting the maximum distance
                  double tauCone = 10000.0;
                  if(coneMetric_=="DR")
                  {
                    tauCone = std::max(ROOT::Math::VectorUtil::DeltaR(tau.p4(),h1->p4()),
                              std::max(ROOT::Math::VectorUtil::DeltaR(tau.p4(),h2->p4()),
                                       ROOT::Math::VectorUtil::DeltaR(tau.p4(),h3->p4())));
                  }
                  else if(coneMetric_=="angle")
                  {
                    tauCone =std::max(fabs(ROOT::Math::VectorUtil::Angle(tau.p4(),h1->p4())),
                             std::max(fabs(ROOT::Math::VectorUtil::Angle(tau.p4(),h2->p4())),
                                      fabs(ROOT::Math::VectorUtil::Angle(tau.p4(),h3->p4()))));
                  }

                  //Set The PFTau
                  tau.setsignalPFChargedHadrCands(signal);
                  tau.setsignalPFCands(signal);
                  tau.setleadPFChargedHadrCand(h1);
                  tau.setleadPFCand(h1);

                  if(isNarrowTau(tau,tauCone)) {
                    //calculate the isolation Deposits
                    associateIsolationCandidates(tau,tauCone);
                    applyMuonRejection(tau);
                    applyElectronRejection(tau,0.0);
                    taus.push_back(tau);

                  }
                }
            }
        }

  if(taus.size()>0) {
    PFTau bestTau  = getBestTauCandidate(taus);
    if(refitThreeProng(bestTau))
      //Apply mass constraint
      if(bestTau.mass()>threeProngMassWindow_.at(0)&&bestTau.mass()<threeProngMassWindow_.at(1))//MassWindow
        pfTaus_.push_back(bestTau);
  }

}


bool
HPSPFRecoTauAlgorithm::isNarrowTau(const reco::PFTau& tau,double cone)
{
  double allowedConeSize=coneSizeFormula.Eval(tau.energy(),tau.et());
  if (allowedConeSize<minSignalCone_) allowedConeSize=minSignalCone_;
  if (allowedConeSize>maxSignalCone_) allowedConeSize=maxSignalCone_;

  if(cone<allowedConeSize)
    return true;
  else
    return false;
}


void
HPSPFRecoTauAlgorithm::associateIsolationCandidates(reco::PFTau& tau,
                                                    double tauCone)
{


  using namespace reco;

  //Information to get filled
  double sumPT=0;
  double sumET=0;

  if(tau.pfTauTagInfoRef().isNull()) return;

  PFCandidateRefVector hadrons;
  PFCandidateRefVector gammas;
  PFCandidateRefVector neutral;

  //Remove candidates outside the cones
  if(useIsolationAnnulus_)
  {

    for(unsigned int i=0;i<tau.pfTauTagInfoRef()->PFChargedHadrCands().size();++i) {
      double DR = ROOT::Math::VectorUtil::DeltaR(tau.p4(),tau.pfTauTagInfoRef()->PFChargedHadrCands().at(i)->p4());

      if(DR>tauCone && DR<chargeIsolationCone_)
        hadrons.push_back(tau.pfTauTagInfoRef()->PFChargedHadrCands().at(i));
    }

    for(unsigned int i=0;i<tau.pfTauTagInfoRef()->PFGammaCands().size();++i) {
      double DR = ROOT::Math::VectorUtil::DeltaR(tau.p4(),tau.pfTauTagInfoRef()->PFGammaCands().at(i)->p4());

      if(DR>tauCone && DR<gammaIsolationCone_)
        gammas.push_back(tau.pfTauTagInfoRef()->PFGammaCands().at(i));
    }
    for(unsigned int i=0;i<tau.pfTauTagInfoRef()->PFNeutrHadrCands().size();++i) {
      double DR = ROOT::Math::VectorUtil::DeltaR(tau.p4(),tau.pfTauTagInfoRef()->PFNeutrHadrCands().at(i)->p4());
      if(DR>tauCone && DR <neutrHadrIsolationCone_)
        neutral.push_back(tau.pfTauTagInfoRef()->PFNeutrHadrCands().at(i));
    }
  }
  else
  {

    for(unsigned int i=0;i<tau.pfTauTagInfoRef()->PFChargedHadrCands().size();++i) {
      double DR = ROOT::Math::VectorUtil::DeltaR(tau.p4(),tau.pfTauTagInfoRef()->PFChargedHadrCands().at(i)->p4());

      if(DR<chargeIsolationCone_)
        hadrons.push_back(tau.pfTauTagInfoRef()->PFChargedHadrCands().at(i));
    }

    for(unsigned int i=0;i<tau.pfTauTagInfoRef()->PFGammaCands().size();++i) {
      double DR = ROOT::Math::VectorUtil::DeltaR(tau.p4(),tau.pfTauTagInfoRef()->PFGammaCands().at(i)->p4());

      if(DR<gammaIsolationCone_)
        gammas.push_back(tau.pfTauTagInfoRef()->PFGammaCands().at(i));
    }
    for(unsigned int i=0;i<tau.pfTauTagInfoRef()->PFNeutrHadrCands().size();++i) {
      double DR = ROOT::Math::VectorUtil::DeltaR(tau.p4(),tau.pfTauTagInfoRef()->PFNeutrHadrCands().at(i)->p4());
      if(DR <neutrHadrIsolationCone_)
        neutral.push_back(tau.pfTauTagInfoRef()->PFNeutrHadrCands().at(i));
    }

  }

  //remove the signal Constituents from the collections
  for(PFCandidateRefVector::const_iterator i=tau.signalPFChargedHadrCands().begin();i!=tau.signalPFChargedHadrCands().end();++i)
  {
    removeCandidateFromRefVector(*i,hadrons);
  }

  for(PFCandidateRefVector::const_iterator i=tau.signalPFGammaCands().begin();i!=tau.signalPFGammaCands().end();++i)
  {
    removeCandidateFromRefVector(*i,gammas);
    removeCandidateFromRefVector(*i,hadrons);//special case where we included a hadron if the strip!
  }


  //calculate isolation deposits
  for(unsigned int i=0;i<hadrons.size();++i)
  {
    sumPT+=hadrons.at(i)->pt();
  }

  for(unsigned int i=0;i<gammas.size();++i)
  {
    sumET+=gammas.at(i)->pt();
  }


  tau.setisolationPFChargedHadrCandsPtSum(sumPT);
  tau.setisolationPFGammaCandsEtSum(sumET);
  tau.setisolationPFChargedHadrCands(hadrons);
  tau.setisolationPFNeutrHadrCands(neutral);
  tau.setisolationPFGammaCands(gammas);

  PFCandidateRefVector isoAll = hadrons;
  for(unsigned int i=0;i<gammas.size();++i)
    isoAll.push_back(gammas.at(i));

  for(unsigned int i=0;i<neutral.size();++i)
    isoAll.push_back(neutral.at(i));

  tau.setisolationPFCands(isoAll);
}

void
HPSPFRecoTauAlgorithm::applyMuonRejection(reco::PFTau& tau)
{

  // Require that no signal track has segment matches

  //Also:
  //The segment compatibility is the number of matched Muon Segments
  //the old available does not exist in the muons anymore so i will fill the data format with that
  bool decision=true;
  float caloComp=0.0;
  float segComp=0.0;

  if(tau.leadPFChargedHadrCand().isNonnull()) {
    MuonRef mu =tau.leadPFChargedHadrCand()->muonRef();
    if(mu.isNonnull()){
      segComp=(float)(mu->matches().size());
      if(mu->caloCompatibility()>caloComp)
        caloComp = mu->caloCompatibility();

      if(segComp<1.0)
        decision=false;

      tau.setCaloComp(caloComp);
      tau.setSegComp(segComp);
      tau.setMuonDecision(decision);
    }

  }
}


void
HPSPFRecoTauAlgorithm::applyElectronRejection(reco::PFTau& tau,double stripEnergy )
{
  //Here we apply the common electron rejection variables.
  //The only not common is the E/P that is applied in the decay mode
  //construction


  if(tau.leadPFChargedHadrCand().isNonnull()) {
    PFCandidateRef leadCharged = tau.leadPFChargedHadrCand();
    math::XYZVector caloDir(leadCharged->positionAtECALEntrance().x(),
                            leadCharged->positionAtECALEntrance().y(),
                            leadCharged->positionAtECALEntrance().z());

    tau.setmaximumHCALPFClusterEt(leadCharged->hcalEnergy()*sin(caloDir.theta()));



    if(leadCharged->trackRef().isNonnull()) {
      TrackRef track = leadCharged->trackRef();
      tau.setemFraction(leadCharged->ecalEnergy()/(leadCharged->ecalEnergy()+leadCharged->hcalEnergy()));
      //For H/P trust particle Flow ! :Just take HCAL energy of the candidate
      //end of story
      tau.sethcalTotOverPLead(leadCharged->hcalEnergy()/track->p());
      tau.sethcalMaxOverPLead(leadCharged->hcalEnergy()/track->p());
      tau.sethcal3x3OverPLead(leadCharged->hcalEnergy()/track->p());
      tau.setelectronPreIDTrack(track);
      tau.setelectronPreIDOutput(leadCharged->mva_e_pi());
      //Since PF uses brem recovery we will store the default ecal energy here
      tau.setbremsRecoveryEOverPLead(leadCharged->ecalEnergy()/track->p());
      tau.setecalStripSumEOverPLead((leadCharged->ecalEnergy()-stripEnergy)/track->p());
      bool electronDecision;
      if(std::abs(leadCharged->pdgId())==11)
        electronDecision=true;
      else
        electronDecision=false;
      tau.setelectronPreIDDecision(electronDecision);
    }
  }
}




void
HPSPFRecoTauAlgorithm::configure(const edm::ParameterSet& p)
{
  emMerger_                      = p.getParameter<std::string>("emMergingAlgorithm");
  overlapCriterion_              = p.getParameter<std::string>("candOverlapCriterion");
  doOneProngs_                   = p.getParameter<bool>("doOneProng");
  doOneProngStrips_              = p.getParameter<bool>("doOneProngStrip");
  doOneProngTwoStrips_           = p.getParameter<bool>("doOneProngTwoStrips");
  doThreeProngs_                 = p.getParameter<bool>("doThreeProng");
  tauThreshold_                  = p.getParameter<double>("tauPtThreshold");
  leadPionThreshold_             = p.getParameter<double>("leadPionThreshold");
  stripPtThreshold_               = p.getParameter<double>("stripPtThreshold");
  chargeIsolationCone_           = p.getParameter<double>("chargeHadrIsolationConeSize");
  gammaIsolationCone_            = p.getParameter<double>("gammaIsolationConeSize");
  neutrHadrIsolationCone_        = p.getParameter<double>("neutrHadrIsolationConeSize");
  useIsolationAnnulus_           = p.getParameter<bool>("useIsolationAnnulus");
  oneProngStripMassWindow_       = p.getParameter<std::vector<double> >("oneProngStripMassWindow");
  oneProngTwoStripsMassWindow_   = p.getParameter<std::vector<double> >("oneProngTwoStripsMassWindow");
  oneProngTwoStripsPi0MassWindow_= p.getParameter<std::vector<double> >("oneProngTwoStripsPi0MassWindow");
  threeProngMassWindow_          = p.getParameter<std::vector<double> >("threeProngMassWindow");
  matchingCone_                  = p.getParameter<double>("matchingCone");
  coneMetric_                    = p.getParameter<std::string>("coneMetric");
  coneSizeFormula_               = p.getParameter<std::string>("coneSizeFormula");
  minSignalCone_                 = p.getParameter<double>("minimumSignalCone");
  maxSignalCone_                 = p.getParameter<double>("maximumSignalCone");



  //Initialize The Merging Algorithm!
  if(emMerger_ =="StripBased")
    candidateMerger_ =  new PFCandidateStripMerger(p);
  //Add the Pi0 Merger from Evan here


  if(oneProngStripMassWindow_.size()!=2)
    throw cms::Exception("") << "OneProngStripMassWindow must be a vector of size 2 [min,max] " << std::endl;
  if(oneProngTwoStripsMassWindow_.size()!=2)
    throw cms::Exception("") << "OneProngTwoStripsMassWindow must be a vector of size 2 [min,max] " << std::endl;
  if(threeProngMassWindow_.size()!=2)
    throw cms::Exception("") << "ThreeProngMassWindow must be a vector of size 2 [min,max] " << std::endl;
  if(coneMetric_!= "angle" && coneMetric_ != "DR")
    throw cms::Exception("") << "Cone Metric should be angle or DR " << std::endl;

  coneSizeFormula = TauTagTools::computeConeSizeTFormula(coneSizeFormula_,"Signal cone size Formula");


}



math::XYZTLorentzVector
HPSPFRecoTauAlgorithm::createMergedLorentzVector(const reco::PFCandidateRefVector& cands)
{
  math::XYZTLorentzVector sum;
  for(unsigned int i=0;i<cands.size();++i) {
    sum+=cands.at(i)->p4();
  }
  return sum;
}

void
HPSPFRecoTauAlgorithm::removeCandidateFromRefVector(const reco::PFCandidateRef& cand,reco::PFCandidateRefVector& vec)
{
  PFCandidateRefVector newVec;

  PFCandidateRefVector::iterator it;
  it = std::find(vec.begin(),vec.end(),cand);
  if(it!=vec.end())
    vec.erase(it);
}

void
HPSPFRecoTauAlgorithm::applyMassConstraint(math::XYZTLorentzVector& vec,double mass)
{
  double factor = sqrt(vec.energy()*vec.energy()-mass*mass)/vec.P();
  vec.SetCoordinates(vec.px()*factor,vec.py()*factor,vec.pz()*factor,vec.energy());
}




bool
HPSPFRecoTauAlgorithm::refitThreeProng(reco::PFTau& tau)
{
  bool response=false;
  //Get Hadrons
  reco::PFCandidateRefVector hadrons = tau.signalPFChargedHadrCands();
  PFCandidateRef  h1  = hadrons.at(0);
  PFCandidateRef  h2  = hadrons.at(1);
  PFCandidateRef  h3  = hadrons.at(2);

  //Make transient tracks
  std::vector<TransientTrack> transientTracks;
  transientTracks.push_back(TransientTrackBuilder_->build(h1->trackRef()));
  transientTracks.push_back(TransientTrackBuilder_->build(h2->trackRef()));
  transientTracks.push_back(TransientTrackBuilder_->build(h3->trackRef()));

  //Apply the Vertex Fit
  KalmanVertexFitter fitter(true);
  TransientVertex myVertex = fitter.vertex(transientTracks);

  //Just require a valid vertex+ 3 refitted tracks
  if(myVertex.isValid()&&
     myVertex.hasRefittedTracks()&&
     myVertex.refittedTracks().size()==3) {

    response=true;
    math::XYZPoint vtx(myVertex.position().x(),myVertex.position().y(),myVertex.position().z());

    //Create a LV for each refitted track
    math::XYZTLorentzVector p1(myVertex.refittedTracks().at(0).track().px(),
                               myVertex.refittedTracks().at(0).track().py(),
                               myVertex.refittedTracks().at(0).track().pz(),
                               sqrt(myVertex.refittedTracks().at(0).track().momentum().mag2() +0.139*0.139));

    math::XYZTLorentzVector p2(myVertex.refittedTracks().at(1).track().px(),
                               myVertex.refittedTracks().at(1).track().py(),
                               myVertex.refittedTracks().at(1).track().pz(),
                               sqrt(myVertex.refittedTracks().at(1).track().momentum().mag2() +0.139*0.139));

    math::XYZTLorentzVector p3(myVertex.refittedTracks().at(2).track().px(),
                               myVertex.refittedTracks().at(2).track().py(),
                               myVertex.refittedTracks().at(2).track().pz(),
                               sqrt(myVertex.refittedTracks().at(2).track().momentum().mag2() +0.139*0.139));

    //Update the tau p4
    tau.setP4(p1+p2+p3);
    //Update the vertex
    tau.setVertex(vtx);
  }
  return response;

}


reco::PFTau
HPSPFRecoTauAlgorithm::getBestTauCandidate(reco::PFTauCollection& taus)
{
  reco::PFTauCollection::iterator it;
  if(overlapCriterion_ =="Isolation"){
    HPSTauIsolationSorter sorter;
    it = std::min_element(taus.begin(),taus.end(),sorter);

  }
  else if(overlapCriterion_ =="Pt"){
    HPSTauPtSorter sorter;
    it = std::min_element(taus.begin(),taus.end(),sorter);
  }

  return *it;
}

