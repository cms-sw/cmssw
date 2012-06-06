// -*- C++ -*-
//
// Package:    METProducers
// Class:      METProducer
// 
/**\class METProducer METProducer.h RecoMET/METProducers/interface/METProducer.h

 Description: An EDProducer which produces MET

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Rick Cavanaugh
//         Created:  May 14, 2005
// $Id$
//
//

#ifndef METProducer_h
#define METProducer_h

#include <vector>
#include <cstdlib>
#include <string.h>
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMET/METAlgorithms/interface/METAlgo.h" 
#include "RecoMET/METAlgorithms/interface/TCMETAlgo.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/PFClusterMETFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "TH2F.h"

class TCMETAlgo;

namespace metsig{
    class SignAlgoResolutions;
}

namespace cms 
{
  class METProducer: public edm::EDProducer 
    {
    public:
      typedef math::XYZTLorentzVector LorentzVector;
      typedef math::XYZPoint Point;
      typedef edm::OwnVector<reco::Candidate> CandidateCollection;
      explicit METProducer(const edm::ParameterSet&);
      explicit METProducer();
      virtual ~METProducer();
      //const CandidateCollection* convert( const reco::CaloJetCollection* );
      virtual void produce(edm::Event&, const edm::EventSetup&);

    private:
      METAlgo alg_; 
      edm::InputTag inputLabel;
      std::string inputType;
      std::string METtype;
      std::string alias;
      
      //Calculate MET Significance (not necessary at HLT)
      bool calculateSignificance_;
      metsig::SignAlgoResolutions *resolutions_;
      edm::InputTag jetsLabel_; //used for jet-based significance calculation
     
      //Use HF in CaloMET calculation?
      bool noHF;
      
      //Use an Et threshold on all of the objects in the CaloMET calculation?
      double globalThreshold;

      //Use only fiducial GenParticles in GenMET calculation? 
      bool onlyFiducial;

      //Use Pt instaed of Et
      bool usePt; 

      TCMETAlgo* tcmetalgorithm;
      int myResponseFunctionType;

    };
}

#endif // METProducer_h
