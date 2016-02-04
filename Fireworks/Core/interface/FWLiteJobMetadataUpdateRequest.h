#ifndef Fireworks_Core_FWLiteJobMetadataUpdateRequest
#define Fireworks_Core_FWLiteJobMetadataUpdateRequest

#include "Fireworks/Core/interface/FWJobMetadataUpdateRequest.h"

namespace fwlite
{
   class Event;
}

class TFile;


class FWLiteJobMetadataUpdateRequest : public FWJobMetadataUpdateRequest
{
public:
   FWLiteJobMetadataUpdateRequest(const fwlite::Event *event, 
                                  const TFile *file)
      : event_(event), file_(file)
   {
      
   }
   
   const fwlite::Event *event_;
   const TFile *file_;
};

#endif