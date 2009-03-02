// -*- C++ -*-
//
// Package:    RPCCSC
// Class:      RPCCSC
// 
/**\class RPCCSC RPCCSC.cc TESTCSCRPC/RPCCSC/src/RPCCSC.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
//
// Original Author:  Camilo Andres Carrillo Montoya
//         Created:  Wed Feb 25 18:09:15 CET 2009
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include "FWCore/Framework/interface/MakerMacros.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

//
// class decleration
//
class CSCStationIndex{
    public:
        CSCStationIndex():_region(0),_station(0),_ring(0),_chamber(0){}
        CSCStationIndex(int region, int station, int ring, int chamber):
            _region(region),
            _station(station),
            _ring(ring),
            _chamber(chamber){}
        ~CSCStationIndex(){}
        int region() const {return _region;}
        int station() const {return _station;}
        int ring() const {return _ring;}
        int chamber() const {return _chamber;}
        bool operator<(const CSCStationIndex& cscind) const{
            if(cscind.region()!=this->region())
                return cscind.region()<this->region();
            else if(cscind.station()!=this->station())
                return cscind.station()<this->station();
            else if(cscind.ring()!=this->ring())
                return cscind.ring()<this->ring();
            else if(cscind.chamber()!=this->chamber())
                return cscind.chamber()<this->chamber();
            return false;
        }

    private:
        int _region;
        int _station;
        int _ring;  
        int _chamber;
};

class RPCCSC : public edm::EDAnalyzer {
    public:
        explicit RPCCSC(const edm::ParameterSet&);
        ~RPCCSC();


    private:
        virtual void beginJob(const edm::EventSetup&) ;
        virtual void analyze(const edm::Event&, const edm::EventSetup&);
        virtual void endJob() ;

        // ----------member data ---------------------------
        std::map<CSCStationIndex,std::set<RPCDetId> > rollstoreCSC;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//





RPCCSC::RPCCSC(const edm::ParameterSet& iConfig)

{
    //now do what ever initialization is needed

}


RPCCSC::~RPCCSC()
{

    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

int mySegment(RPCDetId rpcId){
    int seg=0;
    int nsec=36;
    int nsub=6;
    if (rpcId.ring()==1 && rpcId.station() > 1) {
        nsub=3;
        nsec=18;
    }
    seg =rpcId.subsector()+nsub*(rpcId.sector()-1);
    if(seg==nsec+1)seg=1;
    return seg;
}



// ------------ method called to for each event  ------------
void RPCCSC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
}


// ------------ method called once each job just before starting event loop  ------------
    void 
RPCCSC::beginJob(const edm::EventSetup& iSetup)
{
    using namespace std;
    edm::ESHandle<RPCGeometry> pRPCGeom;
    iSetup.get<MuonGeometryRecord>().get(pRPCGeom);
    const RPCGeometry* rpcGeometry = (const RPCGeometry*)&*pRPCGeom;
    edm::ESHandle<CSCGeometry> pCSCGeom;
    iSetup.get<MuonGeometryRecord>().get(pCSCGeom);
    const CSCGeometry* cscGeometry = (const CSCGeometry*)&*pCSCGeom;

    for (TrackingGeometry::DetContainer::const_iterator it=rpcGeometry->dets().begin();it<rpcGeometry->dets().end();it++){
        if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
            RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
            std::vector< const RPCRoll*> roles = (ch->rolls());
            for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
                RPCDetId rpcId = (*r)->id();
                int region=rpcId.region();
                //booking all histograms
                RPCGeomServ rpcsrv(rpcId);
                std::string nameRoll = rpcsrv.name();
                //std::cout<<"Booking for "<<nameRoll<<std::endl;

                if(region==1){
                    const TrapezoidalStripTopology* topE_=dynamic_cast<const TrapezoidalStripTopology*>(&((*r)->topology()));
                    float stripl = topE_->stripLength();
                    float stripw = topE_->pitch();
                    //meCollection[nameRoll] = bookDetUnitSeg(rpcId,(*r)->nstrips(),stripw,stripl);
                    //std::cout<<"--Filling the cscstore"<<rpcId<<std::endl;
                    int region=rpcId.region();
                    int station=rpcId.station();
                    int ring=rpcId.ring();
                    int cscring=ring;
                    int cscstation=station;
                    RPCGeomServ rpcsrv(rpcId);
                    int rpcsegment = mySegment(rpcId); //This replace rpcsrv.segment();
                    //std::cout<<"My segment="<<mySegment(rpcId)<<" GeomServ="<<rpcsrv.segment()<<std::endl;
                    int cscchamber = rpcsegment;//FIX THIS ACCORDING TO RPCGeomServ::segment()Definition
                    if((station==2||station==3)&&ring==3){//Adding Ring 3 of RPC to the CSC Ring 2
                        cscring = 2;
                    }

                    CSCStationIndex ind(region,cscstation,cscring,cscchamber);
                    std::set<RPCDetId> myrolls;
                    if (rollstoreCSC.find(ind)!=rollstoreCSC.end()){
                        myrolls=rollstoreCSC[ind];
                    }
                    myrolls.insert(rpcId);
                    rollstoreCSC[ind]=myrolls;

                }
            }
        }
    }


    // Now check binding
    const std::vector<CSCChamber*> cscChambers = cscGeometry->chambers();
    for(std::vector<CSCChamber*>::const_iterator CSCChamberIter = cscChambers.begin(); CSCChamberIter != cscChambers.end(); CSCChamberIter++)
    {   
        CSCDetId cscChamber = (*CSCChamberIter)->id();
        int Endcap = cscChamber.endcap();
        int Station = cscChamber.station();
        int Ring = cscChamber.ring();
        int Chamber = cscChamber.chamber();
        int Layer = cscChamber.layer();
        CSCStationIndex index(Endcap, Station, Ring, Chamber);
        // 1=Forward(+), 2=Backward(-)
        if(Endcap != 1)
            continue;
        if(Station == 4)
            continue;
        if(Ring == 1 || Ring == 4)
            continue;
        cout << "Checking +Z endcap CSC chamber: " << cscChamber.rawId() << ", Endcap: " << Endcap << ", Station: " << Station << ", Ring: " << Ring << ", Chamber: " << Chamber << ", layer: " << Layer << endl;
        const Bounds& cscBounds = (*CSCChamberIter)->surface().bounds();
        const CSCLayer* cscLayer = (*CSCChamberIter)->layer(1);
        const CSCStripTopology* cscTop = cscLayer->geometry()->topology();
        int nCSCStrip0 = cscLayer->geometry()->numberOfStrips();
        int nCSCStrip1 = cscTop->nstrips();
        cout << "CSC layer strips from layer geometry: " << nCSCStrip0 << ", from layer topology: " << nCSCStrip1 << endl;
        GlobalPoint Edge0 = cscLayer->toGlobal(cscTop->localPosition(0));
        GlobalPoint Edge1 = cscLayer->toGlobal(cscTop->localPosition(nCSCStrip1));
        cout << "CSC Phi range from " << Edge0.phi() << " to " << Edge1.phi() << endl;
        cout << "CSC Phi degree range from " << Edge0.phi().degrees() << " to " << Edge1.phi().degrees() << endl;
        if(rollstoreCSC.find(index) != rollstoreCSC.end())
        {
            std::set<RPCDetId> RPCRolls;
            RPCRolls = rollstoreCSC[index];
            cout <<"Bind to " << RPCRolls.size() << " RPC Rolls" << endl;
            for(std::set<RPCDetId>::const_iterator RPCRollIter = RPCRolls.begin(); RPCRollIter != RPCRolls.end(); RPCRollIter++)
            {
                RPCGeomServ rpcSrv(*RPCRollIter);
                int rEndcap = RPCRollIter->region();
                int rStation = RPCRollIter->station();
                int rRing = RPCRollIter->ring();
                int rSegment = rpcSrv.segment();
                int rRoll = RPCRollIter->roll();
                const RPCRoll* rpcRoll = rpcGeometry->roll(*RPCRollIter);
                int nRPCStrip = rpcRoll->nstrips();

                // Camilo's method
                const BoundPlane & RPCSurface = rpcRoll->surface();
                GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
                std::cout<<"CSC \t \t \t Center (0,0,0) of the Roll in Global"<<CenterPointRollGlobal<<", with phi "<<CenterPointRollGlobal.phi()<<std::endl;
                GlobalPoint CenterPointCSCGlobal = (*CSCChamberIter)->toGlobal(LocalPoint(0,0,0));
                std::cout<<"CSC \t \t \t Center (0,0,0) of the CSC in Global"<<CenterPointCSCGlobal<<", with phi "<<CenterPointCSCGlobal.phi()<<std::endl;
                
                float rpcphi=0;
                float cscphi=0;
                (CenterPointRollGlobal.barePhi()<0)? 
                    rpcphi = 2*3.141592+CenterPointRollGlobal.barePhi():rpcphi=CenterPointRollGlobal.barePhi();
                (CenterPointCSCGlobal.barePhi()<0)? 
                    cscphi = 2*3.1415926536+CenterPointCSCGlobal.barePhi():cscphi=CenterPointCSCGlobal.barePhi();
                float df=fabs(cscphi-rpcphi); 
                float dr=fabs(CenterPointRollGlobal.perp()-CenterPointCSCGlobal.perp());
                float diffz=CenterPointRollGlobal.z()-CenterPointCSCGlobal.z();
                float dfg=df*180./3.14159265;
                if(dr>200.||fabs(diffz)>55.||dfg>1.)
                    cout << "Wrong binding for RPC Roll in Camilo's method" << RPCRollIter->rawId() << ", Endcap: " << rEndcap << ", Station: " << rStation << ", Ring: " << rRing << ", Segment: " << rSegment << ", Roll: " << rRoll << endl;

                GlobalPoint insideEdge0 = rpcRoll->toGlobal(rpcRoll->centreOfStrip(0));
                GlobalPoint insideEdge1 = rpcRoll->toGlobal(rpcRoll->centreOfStrip(nRPCStrip));
                cout << "RPC roll: " << RPCRollIter->rawId() <<" Phi range from " << insideEdge0.phi() << " to " << insideEdge1.phi() << endl;
                cout << "RPC roll: " << RPCRollIter->rawId() <<" Phi degree range from " << insideEdge0.phi().degrees() << " to " << insideEdge1.phi().degrees() << endl;
                LocalPoint inside0 = LocalPoint((*CSCChamberIter)->surface().toLocal(insideEdge0).x(), (*CSCChamberIter)->surface().toLocal(insideEdge0).y(), 0);
                LocalPoint inside1 = LocalPoint((*CSCChamberIter)->surface().toLocal(insideEdge1).x(), (*CSCChamberIter)->surface().toLocal(insideEdge1).y(), 0);

                if((Edge0.phi()-insideEdge0.phi()).value()*(Edge1.phi()-insideEdge1.phi()).value() < 0)
                {
                    cout << "Well binding for RPC Roll: " << RPCRollIter->rawId() << ", Endcap: " << rEndcap << ", Station: " << rStation << ", Ring: " << rRing << ", Segment: " << rSegment << ", Roll: " << rRoll << endl;
                }
                else
                {
                    cout << "Wrong binding for RPC Roll " << RPCRollIter->rawId() << ", Endcap: " << rEndcap << ", Station: " << rStation << ", Ring: " << rRing << ", Segment: " << rSegment << ", Roll: " << rRoll << endl;
                }
            }
        }
        else
        {
            cout << "Could not find the binding RPC roll" << endl;
        }
    }

}

// ------------ method called once each job just after ending the event loop  ------------
void 
RPCCSC::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCCSC);
