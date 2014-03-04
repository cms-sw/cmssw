#ifndef UnpackerCollections_h
#define UnpackerCollections_h

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

namespace l1t {
   class L1TRawToDigi;

   class UnpackerCollections {
      public:
         UnpackerCollections(edm::Event& event);
         ~UnpackerCollections();

         inline JetBxCollection * const getJetCollection() const { return jets_.get(); };
         inline TauBxCollection * const getTauCollection() const { return taus_.get(); };

         static void registerCollections(L1TRawToDigi*);

      private:
         // Keep this a singular object.
         UnpackerCollections(const UnpackerCollections&);
         UnpackerCollections& operator=(const UnpackerCollections&);

         edm::Event& event_;

         std::auto_ptr<JetBxCollection> jets_;
         std::auto_ptr<TauBxCollection> taus_;
   };
}

#endif
