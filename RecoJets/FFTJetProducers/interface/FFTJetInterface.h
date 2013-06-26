// -*- C++ -*-
//
// Package:    FFTJetProducers
// Class:      FFTJetInterface
// 
/**\class FFTJetInterface FFTJetInterface.h RecoJets/FFTJetProducers/interface/FFTJetInterface.hh

 Description: common facilities for the FFTJet interface code

*/
//
// Original Author:  Igor Volobouev
//         Created:  June 29 2010
// $Id: FFTJetInterface.h,v 1.3 2012/08/06 21:36:54 igv Exp $
//
//

#ifndef RecoJets_FFTJetProducers_FFTJetInterface_h
#define RecoJets_FFTJetProducers_FFTJetInterface_h

#include <memory>
#include <vector>
#include <cassert>

// FFTJet headers
#include "fftjet/Grid2d.hh"

// framework include files
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Candidate/interface/Particle.h"

#include "RecoJets/JetProducers/interface/AnomalousTower.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"

// local FFTJet-related definitions
#include "RecoJets/FFTJetAlgorithms/interface/fftjetTypedefs.h"
#include "RecoJets/FFTJetProducers/interface/JetType.h"

//
// class declaration
//
namespace fftjetcms {
  class FFTJetInterface
  {
  public:
    virtual ~FFTJetInterface() {}

  protected:
    explicit FFTJetInterface(const edm::ParameterSet&);

    template<class Ptr>
    void checkConfig(const Ptr& ptr, const char* message)
    {
      if (ptr.get() == NULL)
	throw cms::Exception("FFTJetBadConfig") << message << std::endl;
    }

    void loadInputCollection(const edm::Event&);
    void discretizeEnergyFlow();
    double getEventScale() const;
    bool storeInSinglePrecision() const;

    const reco::Particle::Point& vertexUsed() const {return vertex_;}

    // Label for the input collection
    const edm::InputTag inputLabel;

    // Label for the output objects
    const std::string outputLabel;

    // Jet type to produce
    const JetType jetType;

    // Vertex correction-related stuff
    const bool doPVCorrection;

    // Label for the vertices
    const edm::InputTag srcPVs;

    // Try to equalize magnitudes in the energy flow grid?
    const std::vector<double> etaDependentMagnutideFactors;

    // Functor for finding anomalous towers
    const AnomalousTower anomalous;

    // Event data 4-vectors
    std::vector<fftjetcms::VectorLike> eventData;

    // Candidate number which corresponds to the item in the "eventData"
    std::vector<unsigned> candidateIndex;

    // The energy discretization grid
    std::auto_ptr<fftjet::Grid2d<fftjetcms::Real> > energyFlow;

    // The input handle for the collection of candidates
    edm::Handle<reco::CandidateView> inputCollection;

  private:
    // Explicitly disable other ways to construct this object
    FFTJetInterface();
    FFTJetInterface(const FFTJetInterface&);
    FFTJetInterface& operator=(const FFTJetInterface&);

    const bool insertCompleteEvent;
    const double completeEventScale;
    reco::Particle::Point vertex_;
  };
}

#endif // RecoJets_FFTJetProducers_FFTJetInterface_h
