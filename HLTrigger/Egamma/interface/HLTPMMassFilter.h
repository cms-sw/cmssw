#ifndef HLTPMMassFilter_h
#define HLTPMMassFilter_h

/** \class HLTPMMassFilter
 *
 *  Original Author: Jeremy Werner 
 *  Institution: Princeton University, USA
 *  Contact: Jeremy.Werner@cern.ch
 *  Date: February 21, 2007
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

// TEST
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TFile.h"
#include "TDirectory.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLorentzVector.h"
#include "TVector3.h"
//
// class declaration
//

class HLTPMMassFilter : public HLTFilter {

   public:
      explicit HLTPMMassFilter(const edm::ParameterSet&);
      ~HLTPMMassFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);      

   private:
      TLorentzVector approxMomAtVtx( const MagneticField *magField, const GlobalPoint& xvert, const reco::SuperClusterRef sc, int charge) ;

      FTSFromVertexToPointFactory theFTSFactory;
      edm::ESHandle<MagneticField> theMagField;

      edm::InputTag candTag_;     // input tag identifying product contains filtered egammas
      edm::InputTag beamSpot_;    // input tag identifying beamSpot product
      double lowerMassCut_;
      double upperMassCut_;
      // double lowerPtCut_;
      int    nZcandcut_;           // number of Z candidates required
      bool   reqOppCharge_;

      bool   relaxed_;
      bool   isElectron1_;
      bool   isElectron2_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 

};

#endif //HLTPMMassFilter_h


