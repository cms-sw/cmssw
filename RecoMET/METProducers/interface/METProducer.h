#ifndef METProducer_h
#define METProducer_h

/** \class METProducer
 *
 * METProducer is the EDProducer subclass which runs 
 * the METAlgo MET finding algorithm.
 *
 * \author R. Cavanaugh, The University of Florida
 *
 * \version 1st Version May 14, 2005
 *
 */

#include <vector>
#include <cstdlib>
#include <string.h>
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMET/METAlgorithms/interface/METAlgo.h" 
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/OwnVector.h"

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
      const CandidateCollection* convert( const reco::CaloJetCollection* );
      virtual void produce(edm::Event&, const edm::EventSetup&);
    private:
      METAlgo alg_; 
      std::string inputLabel;
      std::string inputType;
      std::string METtype;
      std::string alias;
      double globalThreshold;
      CandidateCollection tempCol;
    };
}

#endif // METProducer_h
