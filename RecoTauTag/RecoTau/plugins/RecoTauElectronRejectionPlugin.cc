/*
 * ===========================================================================
 *
 *       Filename:  RecoTauElectronRejectionPlugin.cc
 *
 *    Description:  Add electron rejection information to PFTau
 *
 *         Authors:  Chi Nhan Nguyen, Simone Gennai, Evan Friis
 *
 * ===========================================================================
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include <Math/VectorUtil.h>
#include <algorithm>

namespace reco { namespace tau {

class RecoTauElectronRejectionPlugin : public RecoTauModifierPlugin {
  public:
    explicit RecoTauElectronRejectionPlugin(const edm::ParameterSet& pset);
    virtual ~RecoTauElectronRejectionPlugin() {}
    void operator()(PFTau&) const;
  private:
    double ElecPreIDLeadTkMatch_maxDR_;
    double EcalStripSumE_minClusEnergy_;
    double EcalStripSumE_deltaEta_;
    double EcalStripSumE_deltaPhiOverQ_minValue_;
    double EcalStripSumE_deltaPhiOverQ_maxValue_;
    double maximumForElectrionPreIDOutput_;
    std::string DataType_;
};

RecoTauElectronRejectionPlugin::RecoTauElectronRejectionPlugin(
    const edm::ParameterSet& pset):RecoTauModifierPlugin(pset) {
  // Load parameters
  ElecPreIDLeadTkMatch_maxDR_ =
    pset.getParameter<double>("ElecPreIDLeadTkMatch_maxDR");
  EcalStripSumE_minClusEnergy_ =
    pset.getParameter<double>("EcalStripSumE_minClusEnergy");
  EcalStripSumE_deltaEta_ =
    pset.getParameter<double>("EcalStripSumE_deltaEta");
  EcalStripSumE_deltaPhiOverQ_minValue_ =
    pset.getParameter<double>("EcalStripSumE_deltaPhiOverQ_minValue");
  EcalStripSumE_deltaPhiOverQ_maxValue_ =
    pset.getParameter<double>("EcalStripSumE_deltaPhiOverQ_maxValue");
  maximumForElectrionPreIDOutput_ =
    pset.getParameter<double>("maximumForElectrionPreIDOutput");
  DataType_ = pset.getParameter<std::string>("DataType");
}

namespace {
bool checkPos(std::vector<math::XYZPoint> CalPos,math::XYZPoint CandPos) {
  bool flag = false;
  for (unsigned int i=0;i<CalPos.size();i++) {
    if (CalPos[i] == CandPos) {
      flag = true;
      break;
    }
  }
  return flag;
}
}

void RecoTauElectronRejectionPlugin::operator()(PFTau& tau) const {
  // copy pasted from PFRecoTauAlgorithm...
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

  PFCandidatePtr myleadPFChargedCand = tau.leadPFChargedHadrCand();
  // Build list of PFCands in tau
  std::vector<PFCandidatePtr> myPFCands;
  myPFCands.reserve(tau.isolationPFCands().size()+tau.signalPFCands().size());

  std::copy(tau.isolationPFCands().begin(), tau.isolationPFCands().end(),
      std::back_inserter(myPFCands));
  std::copy(tau.signalPFCands().begin(), tau.signalPFCands().end(),
      std::back_inserter(myPFCands));

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
          double deltaEta = abs(myElecTrkEcalPos.eta()-candPos.eta());
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
                double deltaEta = abs(myElecTrkEcalPos.eta()-clusPos.eta());
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
  tau.setemFraction((float)myEmfrac);

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
  tau.sethcalTotOverPLead((float)myHCALenergy);
  tau.sethcalMaxOverPLead((float)myMaximumHCALPFClusterE);
  tau.sethcal3x3OverPLead((float)myHCALenergy3x3);
  tau.setecalStripSumEOverPLead((float)myStripClusterE);
  tau.setmaximumHCALPFClusterEt(myMaximumHCALPFClusterEt);
  tau.setelectronPreIDOutput(myElectronPreIDOutput);
  if (myElecTrk.isNonnull())
    tau.setelectronPreIDTrack(myElecTrk);
  if (myElectronPreIDOutput > maximumForElectrionPreIDOutput_)
    myElecPreid = true;
  tau.setelectronPreIDDecision(myElecPreid);

  // These need to be filled!
  //tau.setbremsRecoveryEOverPLead(my...);

  /* End elecron rejection */
}
}} // end namespace reco::tau
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
    reco::tau::RecoTauElectronRejectionPlugin,
    "RecoTauElectronRejectionPlugin");
