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
// $Id: METProducer.h,v 1.27 2012/06/06 18:41:37 sakuma Exp $
//
//

#ifndef METProducer_h
#define METProducer_h

#include <string.h>
#include "FWCore/Framework/interface/EDProducer.h"
#include "RecoMET/METAlgorithms/interface/METAlgo.h" 
#include "FWCore/Utilities/interface/InputTag.h"


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TCMETAlgo;

namespace metsig {
    class SignAlgoResolutions;
}

namespace cms 
{
  class METProducer: public edm::EDProducer 
    {
    public:
      explicit METProducer(const edm::ParameterSet&);
      explicit METProducer();
      virtual ~METProducer();
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
