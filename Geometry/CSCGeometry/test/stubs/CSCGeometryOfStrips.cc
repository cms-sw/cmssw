#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>

#include <string>
#include <cmath>
#include <iomanip> // for setw() etc.
#include <vector>

class CSCGeometryOfStrips : public edm::EDAnalyzer {

   public:
 
     explicit CSCGeometryOfStrips( const edm::ParameterSet& );
      ~CSCGeometryOfStrips();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );
 
      const std::string& myName() { return myName_;}

   private: 

      const int dashedLineWidth_;
      const std::string dashedLine_;
      const std::string myName_;
};

CSCGeometryOfStrips::CSCGeometryOfStrips( const edm::ParameterSet& iConfig )
 : dashedLineWidth_(101), dashedLine_( std::string(dashedLineWidth_, '-') ), 
  myName_( "CSCGeometryOfStrips" )
{
}


CSCGeometryOfStrips::~CSCGeometryOfStrips()
{
}

void
 CSCGeometryOfStrips::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   //   const double dPi = Geom::pi();
   //   const double radToDeg = 180. / dPi;

   std::cout << myName() << ": Analyzer..." << std::endl;
   std::cout << "start " << dashedLine_ << std::endl;

   edm::ESHandle<CSCGeometry> pDD;
   iSetup.get<MuonGeometryRecord>().get( pDD );     
   std::cout << " Geometry node for CSCGeometry is  " << &(*pDD) << std::endl;   
   std::cout << " I have "<<pDD->detTypes().size()    << " detTypes" << std::endl;
   std::cout << " I have "<<pDD->detUnits().size()    << " detUnits" << std::endl;
   std::cout << " I have "<<pDD->dets().size()        << " dets" << std::endl;
   std::cout << " I have "<<pDD->layers().size()      << " layers" << std::endl;
   std::cout << " I have "<<pDD->chambers().size()    << " chambers" << std::endl;


   std::cout << myName() << ": Begin iteration over geometry..." << std::endl;
   std::cout << "iter " << dashedLine_ << std::endl;

   int icount = 0;

   // Check the DetUnits
   for( CSCGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); ++it ){
     // Do we really have a CSC layer?

     CSCLayer* layer = dynamic_cast<CSCLayer*>( *it );
     
      if( layer ) {
        ++icount;
        DetId detId = layer->geographicalId();
        int id = detId(); // or detId.rawId()

 	std::cout << "\n" << "Testing CSC layer# " << icount <<
	     " id= " << id << " = " << std::oct << id << std::dec << " (octal) " << 
             "   E" << CSCDetId::endcap(id) << 
             " S" << CSCDetId::station(id) << 
             " R" << CSCDetId::ring(id) << 
             " C" << CSCDetId::chamber(id) << 
 	     " L" << CSCDetId::layer(id) << std::endl;

        const CSCLayerGeometry* geom = layer->geometry();
	//        std::cout << *geom;

        const CSCStripTopology* topol = geom->topology();
	//        std::cout << "\n" << *topol;

        int nStrips = geom->numberOfStrips();

        float mps = 30.;
        if ( nStrips < static_cast<int>(mps) ) mps = 8.;

	MeasurementPoint mp( mps, 0.3 );
	MeasurementError merr( 0.25, 0, 0.25 );

	LocalPoint lp = topol->localPosition(mp);
	LocalError lerr = topol->localError(mp,merr);

	MeasurementPoint mp2 = topol->measurementPosition(lp);
	MeasurementError merr2 = topol->measurementError(lp,lerr);

	float strip = 30.;
        if ( nStrips < static_cast<int>(strip) ) strip = 8.;

	LocalPoint lps = topol->localPosition(strip);
	float stripLP = topol->strip(lps);

	std::cout << "Strip check: strip in=" << strip << ", strip out=" << stripLP << std::endl;
	std::cout << "mp.x in=" << mp.x() << ",  mp.x out=" << mp2.x() << std::endl;
	std::cout << "mp.y in=" << mp.y() << ",  mp.y out=" << mp2.y() << std::endl;

	const float kEps = 1.E-04;

	if( fabs( strip - stripLP ) > kEps ){
	  std::cout << "...strip mismatch! " << std::endl;
	}

	if( fabs(mp.x()-mp2.x())>kEps || fabs(mp.y()-mp2.y())>kEps ){
	  std::cout << "...measurement point mismatch!" << std::endl;
	}

	std::cout << "merr.uu in=" << merr.uu() << ", merr.uu out=" << merr2.uu() << std::endl;
	std::cout << "merr.uv in=" << merr.uv() << ", merr.uv out=" << merr2.uv() << std::endl;
	std::cout << "merr.vv in=" << merr.vv() << ", merr.vv out=" << merr2.vv() << std::endl;

	if( fabs(merr.uu()-merr2.uu())>kEps || fabs(merr.vv()-merr2.vv())>kEps || fabs(merr.uv()-merr2.uv())>kEps ){
	  std::cout << "...measurement point error mismatch!" << std::endl;
	}

      }
    else {
      std::cout << "Could not dynamic_cast to a CSCLayer* " << std::endl;
    }
   }
   std::cout << dashedLine_ << " end" << std::endl;
   std::cout << " UPDATED 28.10.2012" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCGeometryOfStrips);
