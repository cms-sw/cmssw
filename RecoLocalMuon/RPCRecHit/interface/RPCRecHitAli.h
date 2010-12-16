#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include<fstream>


//
// class decleration
//

class RPCRecHitAli : public edm::EDProducer {
   public:
      explicit RPCRecHitAli(const edm::ParameterSet&);
      edm::ESHandle<RPCGeometry> rpcGeo;
      std::map<int,float> alignmentinfo;

      ~RPCRecHitAli();

   private:
      edm::InputTag rpcRecHitsLabel;

      RPCRecHitCollection* _ThePoints;
      edm::OwnVector<RPCRecHit> RPCPointVector;
            
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      std::string AlignmentinfoFile;

      bool debug;
      // ----------member data ---------------------------
};

