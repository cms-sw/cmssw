/** Derived from DTGeometryAnalyzer by Nicola Amapane
 *
 *  \author M. Maggi - INFN Bari
 */

#include <memory>
#include <fstream>
#include <FWCore/Framework/interface/Frameworkfwd.h>

#include <FWCore/Framework/interface/one/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

using namespace std;

class RPCGeometryAnalyzer : public edm::one::EDAnalyzer<> {

 public: 
  RPCGeometryAnalyzer( const edm::ParameterSet& pset);

  ~RPCGeometryAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
 
  const std::string& myName() { return myName_;}

 private: 

  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  std::ofstream ofos;
};

RPCGeometryAnalyzer::RPCGeometryAnalyzer( const edm::ParameterSet& /*iConfig*/ )
  : dashedLineWidth_(104), dashedLine_( std::string(dashedLineWidth_, '-') ), 
    myName_( "RPCGeometryAnalyzer" ) 
{ 
  ofos.open("MytestOutput.out"); 
  std::cout <<"======================== Opening output file"<< std::endl;
}


RPCGeometryAnalyzer::~RPCGeometryAnalyzer() 
{
  ofos.close();
  std::cout <<"======================== Closing output file"<< std::endl;
}

void
RPCGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
  edm::ESHandle<RPCGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get( pDD );     

  std::cout << myName() << ": Analyzer..." << std::endl;
  std::cout << "start " << dashedLine_ << std::endl;


  std::cout << " Geometry node for RPCGeom is  " << &(*pDD) << std::endl;   
  cout << " I have "<<pDD->detTypes().size()    << " detTypes" << endl;
  cout << " I have "<<pDD->detUnits().size()    << " detUnits" << endl;
  cout << " I have "<<pDD->dets().size()        << " dets" << endl;
  cout << " I have "<<pDD->rolls().size()       << " rolls" << endl;
  cout << " I have "<<pDD->chambers().size()    << " chambers" << endl;

  std::cout << myName() << ": Begin iteration over geometry..." << std::endl;
  std::cout << "iter " << dashedLine_ << std::endl;

  std::cout << "\n  #     id(hex)      id(dec)                   "
    "  g(x=0)   g(y=0)   g(z=0)  g(z=-1)  g(z=+1)  Ns "
    "  phi(0)  phi(s1)  phi(sN)    dphi    dphi'      ds     off"
    "       uR       uL       lR       lL" << std::endl;

    int iRPCCHcount = 0;

    //  const double dPi = 3.14159265358;
   //   const double radToDeg = 180. / dPi; //@@ Where to get pi from?

   std::set<RPCDetId> sids;
   std::vector<LocalPoint> vlp;
   vlp.emplace_back(LocalPoint(-1, 0, 0));
   vlp.emplace_back(LocalPoint( 0, 0, 0));
   vlp.emplace_back(LocalPoint( 1, 0, 0));
   vlp.emplace_back(LocalPoint( 0,-1, 0));
   vlp.emplace_back(LocalPoint( 0, 0, 0));
   vlp.emplace_back(LocalPoint( 0, 1, 0));
   vlp.emplace_back(LocalPoint( 0, 0,-1));
   vlp.emplace_back(LocalPoint( 0, 0, 0));
   vlp.emplace_back(LocalPoint( 0, 0, 1));


   for(auto it : pDD->dets()){

     if( std::static_pointer_cast< RPCChamber >( it ) != nullptr ){
       ++iRPCCHcount;
       auto ch = std::static_pointer_cast< RPCChamber >( it ); 

       
       RPCDetId detId=ch->id();
       int idRaf = detId.rawId();
       std::cout<<"Num = "<<iRPCCHcount<<"  "<<"RPCDet = "<<idRaf<<"  Num Rolls =" <<ch->nrolls()<<std::endl;

       for(auto & r : ch->rolls()){

	 if(r->id().region() == 0){
	 std::cout<<"RPCDetId = "<<r->id().rawId()<<std::endl;
	 std::cout<<"Region = "<<r->id().region()<<"  Ring = "<<r->id().ring()
		  <<"  Station = "<<r->id().station()<<"  Sector = "<<r->id().sector()
		  <<"  Layer = "<<r->id().layer()<<"  Subsector = "<<r->id().subsector()
		  <<"  Roll = "<<r->id().roll()<<std::endl;
	 }
       }
     }
   }
   std::cout <<std::endl;
   std::cout << dashedLine_ << " end" << std::endl;
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(RPCGeometryAnalyzer);
