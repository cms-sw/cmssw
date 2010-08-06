#ifndef Fireworks_FWInterface_FWFFMetadataManagerUpdateRequest_h
#define Fireworks_FWInterface_FWFFMetadataManagerUpdateRequest_h

#include "Fireworks/Core/interface/FWJobMetadataUpdateRequest.h"
#include "FWCore/Framework/interface/Event.h"

class FWFFMetadataUpdateRequest : public FWJobMetadataUpdateRequest
{
public:
   FWFFMetadataUpdateRequest(const edm::Event &event)
      {}
};

#endif
