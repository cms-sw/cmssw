#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"

#include "PhysicsTools/PatUtils/interface/ElectronSelector.h"

using pat::ElectronSelector;

//______________________________________________________________________________
const pat::ParticleStatus
ElectronSelector::filter( const unsigned int&        index, 
                          const edm::View<Electron>& electrons,
                          const ElectronIDmap&       electronIDs,
                          const reco::ClusterShape*  clusterShape 
                          ) const
{

  // List of possible selections
  if      ( config_.selectionType == "none"       ) 
    {
      return GOOD;
    }
  else if ( config_.selectionType == "cut"        ) 
    {
      if ( electronID_(index,electrons,electronIDs)->cutBasedDecision() ) return GOOD;
      return BAD;
    }
  else if ( config_.selectionType == "likelihood" )
    {
      if ( electronID_(index,electrons,electronIDs)->likelihood() > config_.value ) return GOOD;
      return BAD;
    }
  else if ( config_.selectionType == "neuralnet" ) // FIXME: Check sign of comparison!
    {
      if ( electronID_(index,electrons,electronIDs)->neuralNetOutput() > config_.value ) return GOOD;
      return BAD;
    }
  else if ( config_.selectionType == "custom"     ) 
    {
      return customSelection_( index, electrons, clusterShape );
    }


  // Throw! unknown configuration
  throw edm::Exception(edm::errors::Configuration) 
    << "Unknown electron ID selection " << config_.selectionType;

}


//______________________________________________________________________________
const reco::ElectronIDRef& 
ElectronSelector::electronID_( const unsigned int& index,
                               const edm::View<Electron>& electrons,
                               const ElectronIDmap& electronIDs
                               ) const
{
  // Find electron ID for electron with index index
  edm::Ref<std::vector<Electron> > elecsRef = electrons.refAt(index).castTo<edm::Ref<std::vector<Electron> > >();
  ElectronIDmap::const_iterator electronID = electronIDs.find( elecsRef );

  // Return corresponding elecID
  return electronID->val;
}


//______________________________________________________________________________
const pat::ParticleStatus
ElectronSelector::customSelection_( const unsigned int&        index,
                                    const edm::View<Electron>& electrons,
                                    const reco::ClusterShape*  clusterShape
                                    ) const
{

  // Note: this is all taken from SusyAnalyzer

  const reco::GsfElectron& electron = electrons[index];

  // Retrieve information
  float eta          = fabs(electron.p4().Eta());
  float eOverPin     = electron.eSuperClusterOverP();
  float pin          = electron.trackMomentumAtVtx().R();   
  float pout         = electron.trackMomentumOut().R(); 
  float fBrem        = (pin-pout)/pin;
  float hOverE       = electron.hadronicOverEm();
  float deltaPhiIn   = electron.deltaPhiSuperClusterTrackAtVtx();
  float deltaEtaIn   = electron.deltaEtaSuperClusterTrackAtVtx();
  float deltaPhiOut  = electron.deltaPhiSeedClusterTrackAtCalo();
  float invEOverInvP = (1./electron.caloEnergy())-(1./electron.trackMomentumAtVtx().R());
  float sigmaee      = sqrt(clusterShape->covEtaEta());
  float sigmapp      = sqrt(clusterShape->covPhiPhi());
  float E9overE25    = clusterShape->e3x3()/clusterShape->e5x5();

  bool inEndCap = false; // Switch between barrel (0) and endcap (2)
  if (eta > 1.479) { // See EgammaAnalysis/ElectronIDAlgos/src/CutBasedElectronID.cc
    inEndCap = true; 
    sigmaee = sigmaee - 0.02*(fabs(eta) - 2.3);   // Correct sigmaetaeta dependence on eta in endcap
  }

  // Now do the selection
  // These ones come straight from E/gamma algo
  if ( (eOverPin < 0.8) && (fBrem < 0.2) ) return BAD;
  if ( config_.doBremEoverPcomp && (eOverPin < 0.9*(1-fBrem)) ) return BAD;

  if (  (hOverE > config_.HoverEBarmax       && !inEndCap )
     || (hOverE > config_.HoverEEndmax       &&  inEndCap ) )
    return HOVERE;

  if (  (E9overE25 < config_.E9overE25Barmin && !inEndCap )
     || (E9overE25 < config_.E9overE25Endmin &&  inEndCap ) )
    return SHOWER;

  if (  (sigmaee > config_.SigmaEtaEtaBarmax && !inEndCap )
     || (sigmaee > config_.SigmaEtaEtaEndmax &&  inEndCap ) )
    return SHOWER;

  if (  (sigmapp > config_.SigmaPhiPhiBarmax && !inEndCap )
     || (sigmapp > config_.SigmaPhiPhiEndmax &&  inEndCap ) )
    return SHOWER;

  if (  (eOverPin < config_.EoverPInBarmin   && !inEndCap )
     || (eOverPin < config_.EoverPInEndmin   &&  inEndCap ) )
    return MATCHING;

  if (  (fabs(deltaEtaIn) > config_.DeltaEtaInBarmax   && !inEndCap )
     || (fabs(deltaEtaIn) > config_.DeltaEtaInEndmax   &&  inEndCap ) )
    return MATCHING;

  if (  (fabs(deltaPhiIn) < config_.DeltaPhiInBarmax   && !inEndCap )
     || (fabs(deltaPhiIn) < config_.DeltaPhiInEndmax   &&  inEndCap ) )
    return MATCHING;

  if (  (fabs(deltaPhiOut) < config_.DeltaPhiOutBarmax && !inEndCap )
     || (fabs(deltaPhiOut) < config_.DeltaPhiOutEndmax &&  inEndCap ) )
    return MATCHING;

  if (  (invEOverInvP > config_.InvEMinusInvPBarmax && !inEndCap )
     || (invEOverInvP > config_.InvEMinusInvPEndmax &&  inEndCap ) )
    return MATCHING;
   
  return GOOD;

}
