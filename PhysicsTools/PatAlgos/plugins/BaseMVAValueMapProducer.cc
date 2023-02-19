// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      BaseMVAValueMapProducer
// 
/**\class BaseMVAValueMapProducer BaseMVAValueMapProducer.cc PhysicsTools/PatAlgos/plugins/BaseMVAValueMapProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andre Rizzi
//         Created:  Mon, 07 Sep 2017 09:18:03 GMT
//
//

#include "PhysicsTools/PatAlgos/interface/BaseMVAValueMapProducer.h"

typedef BaseMVAValueMapProducer<pat::Jet> JetBaseMVAValueMapProducer;
typedef BaseMVAValueMapProducer<pat::Muon> MuonBaseMVAValueMapProducer;
typedef BaseMVAValueMapProducer<pat::Electron> EleBaseMVAValueMapProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(JetBaseMVAValueMapProducer);
DEFINE_FWK_MODULE(MuonBaseMVAValueMapProducer);
DEFINE_FWK_MODULE(EleBaseMVAValueMapProducer);

