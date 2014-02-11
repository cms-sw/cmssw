#ifndef UnpackerCollections_h
#define UnpackerCollections_h

#include "FWCore/Framework/interface/Event.h"

namespace l1t {
   class L1TRawToDigi;

   class UnpackerCollections {
      public:
         UnpackerCollections(edm::Event& event);
         ~UnpackerCollections();

         static void registerCollections(L1TRawToDigi*);

      private:
         // Keep this a singular object.
         UnpackerCollections(const UnpackerCollections&);
         UnpackerCollections& operator=(const UnpackerCollections&);

         edm::Event& event_;
   };
}

#endif
