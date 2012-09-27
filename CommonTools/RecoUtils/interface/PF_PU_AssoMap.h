#ifndef PF_PU_AssoMap_h
#define PF_PU_AssoMap_h


/**\class PF_PU_AssoMap PF_PU_AssoMap.cc CommonTools/RecoUtils/plugins/PF_PU_AssoMap.cc

 Description: Produces a map with association between tracks and their particular most probable vertex with a quality of this association

*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
//

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/RecoUtils/interface/PF_PU_AssoMapAlgos.h"
   
//
// class declaration
//

class PF_PU_AssoMap : public edm::EDProducer {
   public:
      explicit PF_PU_AssoMap(const edm::ParameterSet&);
      ~PF_PU_AssoMap();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------

      edm::InputTag input_VertexCollection_;
      edm::InputTag input_TrackCollection_;

      bool input_VertexAssOneDim_;
      bool input_VertexAssClosest_;
      bool input_VertexAssUseAbsDistance_;

      edm::InputTag input_GsfElectronCollection_;
      edm::InputTag ConversionsCollection_;

      edm::InputTag KshortCollection_;
      edm::InputTag LambdaCollection_;

      edm::InputTag NIVertexCollection_;

      bool UseBeamSpotCompatibility_;
      edm::InputTag input_BeamSpot_;

      int maxNumWarnings_; // CV: print Warning if TrackExtra objects don't exist in input file,
                           //     but only a few times
      int numWarnings_;

      bool ignoremissingpfcollection_;
};


#endif
