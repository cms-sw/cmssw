#ifndef PhysicsTools_TensorFlow_TfGraphRecord_h
#define PhysicsTools_TensorFlow_TfGraphRecord_h
// -*- C++ -*-
//
// Package:     PhysicsTools/TensorFlow
// Class  :     TfGraphRecord
//
/**\class TfGraphRecord TfGraphRecord.h PhysicsTools/TensorFlow/interface/TfGraphRecord.h
 Description: Class to hold Record of a Tensorflow GraphDef that can be used to serve a pretrained tensorflow model for inference
 Usage:
    Used by DataFormats/TrackTfGraph to produce the GraphRecord and RecoTrack/FinalTrackSelection/plugins/TrackTfClassifier.cc to evaluate a track using the graph.
*/
//
// Author:      Joona Havukainen
// Created:     Fri, 24 Jul 2020 07:39:35 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class TfGraphRecord : public edm::eventsetup::EventSetupRecordImplementation<TfGraphRecord> {};

#endif
