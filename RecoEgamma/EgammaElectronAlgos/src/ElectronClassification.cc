#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
// 12/2005
// See GsfElectron::Classification
//===================================================================

using namespace reco;

void ElectronClassification::classify( GsfElectron & electron )
 {
  if ((!electron.isEB())&&(!electron.isEE()))
   {
    edm::LogWarning("")
      << "ElectronClassification::init(): Undefined electron, eta = "
      << electron.eta() << "!!!!" ;
    electron.setClassification(GsfElectron::UNKNOWN) ;
    return ;
   }

  if ( electron.isEBEEGap() || electron.isEBEtaGap() || electron.isEERingGap() )
   {
    electron.setClassification(GsfElectron::GAP) ;
    return ;
   }

  //float pin  = electron.trackMomentumAtVtx().R() ;
  float fbrem = electron.trackFbrem() ;
  int nbrem = electron.numberOfBrems() ;

  if (nbrem == 0 && fbrem < 0.5) // part (pin - scEnergy)/pin < 0.1 removed - M.D.
   { electron.setClassification(GsfElectron::GOLDEN) ; }
  else if (nbrem == 0 && fbrem >= 0.5) // part (pin - scEnergy)/pin < 0.1 removed - M.D.
   { electron.setClassification(GsfElectron::BIGBREM) ; }
  else
   { electron.setClassification(GsfElectron::SHOWERING) ; }

 }

void ElectronClassification::refineWithPflow( GsfElectron & electron )
 {
  if ((!electron.isEB())&&(!electron.isEE()))
   { return ; }

  if ( electron.isEBEEGap() || electron.isEBEtaGap() || electron.isEERingGap() )
   { return ; }

  if ((electron.superClusterFbrem()-electron.trackFbrem())>=0.15)
   { electron.setClassification(GsfElectron::BADTRACK) ; }
 }

