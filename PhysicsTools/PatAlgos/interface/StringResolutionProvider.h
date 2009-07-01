#ifndef PhysicsTools_PatAlgos_StringResolutionProvider_H
#define PhysicsTools_PatAlgos_StringResolutionProvider_H
#include "DataFormats/PatCandidates/interface/CandKinResolution.h"
#include "PhysicsTools/PatAlgos/interface/KinematicResolutionProvider.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"

class StringResolutionProvider : public KinematicResolutionProvider {
    public:
        typedef StringObjectFunction<reco::Candidate> Function;
        StringResolutionProvider(const edm::ParameterSet &iConfig) ;
        virtual ~StringResolutionProvider() ; 
        virtual pat::CandKinResolution getResolution(const reco::Candidate &c) const ;
    private:
        std::auto_ptr<Function> resols_[4]; // StringObjectFunction is not default constructible :-(
	std::vector<pat::CandKinResolution::Scalar> constraints_;
        pat::CandKinResolution::Parametrization parametrization_;
        int dimension_;
};
#endif
