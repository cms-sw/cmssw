#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"

#include "PhysicsTools/PatUtils/interface/ElectronSelector.h"

using pat::ElectronSelector;

//______________________________________________________________________________
ElectronSelector::ElectronSelector( const edm::ParameterSet& config ) :
  selectionCfg_(config),
  selectionType_( config.getParameter<std::string>("type"))
{

  // Retrieve configuration config.ters only once
  if ( selectionType_ == "likelihood" || selectionType_ == "neuralnet" )
    {
      value_ = selectionCfg_.getParameter<double>("value");
    }
  else if ( selectionType_ == "custom" )
    {
      HoverEBarmax_        = config.getParameter<double>("HoverEBarmax");
      SigmaEtaEtaBarmax_   = config.getParameter<double>("SigmaEtaEtaBarmax");
      SigmaPhiPhiBarmax_   = config.getParameter<double>("SigmaPhiPhiBarmax");
      DeltaEtaInBarmax_    = config.getParameter<double>("DeltaEtaInBarmax");
      DeltaPhiInBarmax_    = config.getParameter<double>("DeltaPhiInBarmax");
      DeltaPhiOutBarmax_   = config.getParameter<double>("DeltaPhiOutBarmax");
      EoverPInBarmin_      = config.getParameter<double>("EoverPInBarmin");
      EoverPOutBarmin_     = config.getParameter<double>("EoverPOutBarmin");
      InvEMinusInvPBarmax_ = config.getParameter<double>("InvEMinusInvPBarmax");
      E9overE25Barmin_     = config.getParameter<double>("E9overE25Barmin");
      HoverEEndmax_        = config.getParameter<double>("HoverEEndmax");
      SigmaEtaEtaEndmax_   = config.getParameter<double>("SigmaEtaEtaEndmax");
      SigmaPhiPhiEndmax_   = config.getParameter<double>("SigmaPhiPhiEndmax");
      DeltaEtaInEndmax_    = config.getParameter<double>("DeltaEtaInEndmax");
      DeltaPhiInEndmax_    = config.getParameter<double>("DeltaPhiInEndmax");
      DeltaPhiOutEndmax_   = config.getParameter<double>("DeltaPhiOutEndmax");
      EoverPInEndmin_      = config.getParameter<double>("EoverPInEndmin");
      EoverPOutEndmin_     = config.getParameter<double>("EoverPOutEndmin");
      InvEMinusInvPEndmax_ = config.getParameter<double>("InvEMinusInvPEndmax");
      E9overE25Endmin_     = config.getParameter<double>("E9overE25Endmin");
      doBremEoverPcomp_    = config.getParameter<bool>  ("doBremEoverPcomp");
    }
}


//______________________________________________________________________________
const pat::ParticleStatus
ElectronSelector::filter( const unsigned int&        index, 
                          const edm::View<Electron>& electrons,
                          const ElectronIDmap&       electronIDs,
                          const reco::ClusterShape*  clusterShape 
                          ) const
{

  // List of possible selections
  if      ( selectionType_ == "none"       ) 
    {
      return GOOD;
    }
  else if ( selectionType_ == "cut"        ) 
    {
      if ( electronID(index,electrons,electronIDs)->cutBasedDecision() ) return GOOD;
      return BAD;
    }
  else if ( selectionType_ == "likelihood" )
    {
      if ( electronID(index,electrons,electronIDs)->likelihood() > value_ ) return GOOD;
      return BAD;
    }
  else if ( selectionType_ == "neuralnet" ) // FIXME: Check sign of comparison!
    {
      if ( electronID(index,electrons,electronIDs)->neuralNetOutput() > value_ ) return GOOD;
      return BAD;
    }
  else if ( selectionType_ == "custom"     ) 
    {
      return customSelection_( index, electrons, clusterShape );
    }


  // Throw! unknown configuration
  throw edm::Exception(edm::errors::Configuration) 
    << "Unknown electron ID selection " << selectionType_;

}


//______________________________________________________________________________
const reco::ElectronIDRef& 
ElectronSelector::electronID( const unsigned int& index,
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

  const reco::PixelMatchGsfElectron& electron = electrons[index];

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
  if ( doBremEoverPcomp_ && (eOverPin < 0.9*(1-fBrem)) ) return BAD;

  if (  (hOverE > HoverEBarmax_       && !inEndCap )
     || (hOverE > HoverEEndmax_       &&  inEndCap ) )
    return HOVERE;

  if (  (E9overE25 < E9overE25Barmin_ && !inEndCap )
     || (E9overE25 < E9overE25Endmin_ &&  inEndCap ) )
    return SHOWER;

  if (  (sigmaee > SigmaEtaEtaBarmax_ && !inEndCap )
     || (sigmaee > SigmaEtaEtaEndmax_ &&  inEndCap ) )
    return SHOWER;

  if (  (sigmapp > SigmaPhiPhiBarmax_ && !inEndCap )
     || (sigmapp > SigmaPhiPhiEndmax_ &&  inEndCap ) )
    return SHOWER;

  if (  (eOverPin < EoverPInBarmin_   && !inEndCap )
     || (eOverPin < EoverPInEndmin_   &&  inEndCap ) )
    return MATCHING;

  if (  (fabs(deltaEtaIn) > DeltaEtaInBarmax_   && !inEndCap )
     || (fabs(deltaEtaIn) > DeltaEtaInEndmax_   &&  inEndCap ) )
    return MATCHING;

  if (  (fabs(deltaPhiIn) < DeltaPhiInBarmax_   && !inEndCap )
     || (fabs(deltaPhiIn) < DeltaPhiInEndmax_   &&  inEndCap ) )
    return MATCHING;

  if (  (fabs(deltaPhiOut) < DeltaPhiOutBarmax_ && !inEndCap )
     || (fabs(deltaPhiOut) < DeltaPhiOutEndmax_ &&  inEndCap ) )
    return MATCHING;

  if (  (invEOverInvP > InvEMinusInvPBarmax_ && !inEndCap )
     || (invEOverInvP > InvEMinusInvPEndmax_ &&  inEndCap ) )
    return MATCHING;
   
  return GOOD;

}
