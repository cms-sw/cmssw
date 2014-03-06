#ifndef ME0Segment_ME0MuonConverter_h
#define ME0Segment_ME0MuonConverter_h

/** \class ME0MuonConverter 
 * Produces a collection of ME0Segment's in endcap muon ME0s. 
 *
 * $Date: 2010/03/11 23:48:11 $
 *
 * \author David Nash
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "TH1.h" 
#include "TFile.h"


#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"



class ME0MuonConverter : public edm::EDProducer {
public:
    /// Constructor
    explicit ME0MuonConverter(const edm::ParameterSet&);
    /// Destructor
    ~ME0MuonConverter();
    /// Produce the converted collection
    virtual void produce(edm::Event&, const edm::EventSetup&);

    

private:

    int iev; // events through
};

#endif
