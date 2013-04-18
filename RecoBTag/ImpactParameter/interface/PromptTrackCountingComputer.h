#ifndef ImpactParameter_PromptTrackCountingComputer_h
#define ImpactParameter_PromptTrackCountingComputer_h

#include "RecoBTag/ImpactParameter/interface/TrackCountingComputer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// This returns a discriminator equal to the number of prompt tracks in the jet
// It is intended for exotica physics, not b tagging.
// Author: Ian Tomalin

class PromptTrackCountingComputer : public TrackCountingComputer
{
 public:
  PromptTrackCountingComputer(const edm::ParameterSet & parameters ) : TrackCountingComputer(parameters)
  {
     maxImpactParameterSig = parameters.getParameter<double>("maxImpactParameterSig");
  }
 
  float discriminator(const TagInfoHelper & ti) const 
   {
     const reco::TrackIPTagInfo & tkip = ti.get<reco::TrackIPTagInfo>();
     std::multiset<float> significances = orderedSignificances(tkip);
     std::multiset<float>::iterator sig;
     unsigned int nPromptTrk = 0;
     for(sig=significances.begin(); sig!=significances.end(); sig++) {
       if (fabs(*sig) < maxImpactParameterSig) nPromptTrk++;
       //       edm::LogDebug("") << "Track "<< nPromptTrk << " sig=" << *sig;       
     }
     return double(nPromptTrk);
   }

 private:

  double maxImpactParameterSig;
};

#endif
