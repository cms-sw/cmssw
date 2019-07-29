// -*- C++ -*-
//
// Package:    RecoBTag/PixelCluster
// Class:      PixelClusterTagInfoProducer
// 
/**\class PixelClusterTagInfoProducer PixelCluster RecoBTag/PixelCluster/plugins/PixelClusterTagInfoProducer.cc

 Description: Produces a collection of PixelClusterTagInfo objects,
  that contain the pixel cluster hit multiplicity in each pixel layer or disk
  in a narrow cone around the jet axis.

 Implementation:
     If the event does not fulfill minimum conditions (at leats one jet above threshold,
     and a valid primary vertex) and empty collection is filled. Otherwise, a loop over
     the pixel cluster collection filles a vector of reco::PixelClusterProperties that
     contains the geometrical position and the charge of the cluster above threshold.
     A second loop on jets performs the dR association, and fills the TagInfo collection.
*/
//
// Original Author:  Alberto Zucchetta (UniZ) [zucchett]
//         Created:  Wed, 03 Jul 2019 12:37:30 GMT
//
//

#ifndef RecoBTag_PixelCluster_PixelClusterTagInfoProducer_h
#define RecoBTag_PixelCluster_PixelClusterTagInfoProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// TagInfo
#include "DataFormats/BTauReco/interface/PixelClusterTagInfo.h"

// For vertices
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// For jet
#include "DataFormats/JetReco/interface/PFJet.h" 
#include "DataFormats/JetReco/interface/PFJetCollection.h"

// For pixel clusters
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

// Pixel topology
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

// ROOT
#include "TVector3.h"
#include "TLorentzVector.h"



class PixelClusterTagInfoProducer : public edm::stream::EDProducer<> {
   public:
      explicit PixelClusterTagInfoProducer(const edm::ParameterSet&);
      ~PixelClusterTagInfoProducer() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:

      void produce(edm::Event&, const edm::EventSetup&) override;

      edm::ParameterSet iConfig;
      
      edm::EDGetTokenT<edm::View<reco::Jet> >                 m_jets;
      edm::EDGetTokenT<reco::VertexCollection >               m_vertices;
      edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > m_pixelhit;
      bool m_isPhase1;
      bool m_addFPIX;
      int m_minADC;
      double m_minJetPt;
      double m_maxJetEta;
      int m_nLayers;
};

#endif
