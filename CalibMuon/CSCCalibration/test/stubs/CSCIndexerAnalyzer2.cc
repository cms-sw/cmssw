// Test CSCIndexer 22.11.2012 ptc

#include <memory>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <DataFormats/GeometryVector/interface/Pi.h>

#include <CalibMuon/CSCCalibration/interface/CSCIndexerBase.h>
#include <CalibMuon/CSCCalibration/interface/CSCIndexerRecord.h>

#include <cmath>
#include <iomanip> // for setw() etc.
#include <string>
#include <vector>

class CSCIndexerAnalyzer2 : public edm::EDAnalyzer {

public:
 
  explicit CSCIndexerAnalyzer2( const edm::ParameterSet& );
  ~CSCIndexerAnalyzer2();

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
 
  const std::string& myName() const { return myName_;}
  const std::string& myAlgo() const { return algoName_;}

private: 

  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  
  std::string algoName_;
};

CSCIndexerAnalyzer2::CSCIndexerAnalyzer2( const edm::ParameterSet& iConfig )
  : dashedLineWidth_(146), dashedLine_( std::string(dashedLineWidth_, '-') ), 
    myName_( "CSCIndexerAnalyzer2" ), algoName_( "UNKNOWN" )
{
  std::cout << dashedLine_ << std::endl;
  std::cout << "Welcome to " << myName_ << std::endl;
  std::cout << dashedLine_ << std::endl;
  std::cout << "At present my CSCIndexer algorithm is    " << myAlgo() << std::endl;
  std::cout << dashedLine_ << std::endl;
  std::cout << "I will build the CSC geometry, then iterate over all layers." << std::endl;
  std::cout << "From each CSCDetId I will build the associated linear index, including ME1a layers." << std::endl;
  std::cout << "I will build this index once from the layer labels and once from the CSCDetId, and check they agree." << std::endl;
  std::cout << "I will output one line per layer, listing the CSCDetId, the labels, and these two indexes." << std::endl;
  std::cout << "I will append the strip-channel indexes for the two edge strips and the central strip." << std::endl;
  std::cout << "Finally, I will rebuild a CSCDetId from the layer index, and check it is the same as the original," << std::endl;
  std::cout << "and rebuild a CSCDetId from the final strip-channel index, and check it is the same as the original." << std::endl;
  std::cout << "If any of these tests fail, you will see assert failure messages in the output." << std::endl;
  std::cout << "If there are no such failures then the tests passed." << std::endl;
  std::cout << dashedLine_ << std::endl;
}


CSCIndexerAnalyzer2::~CSCIndexerAnalyzer2(){}

void CSCIndexerAnalyzer2::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   CSCDetId testCSCDetId;

  const double dPi = Geom::pi();
  const double radToDeg = 180. / dPi;

  std::cout << myName() << "::analyze(..) start" << std::endl;
  std::cout << dashedLine_ << std::endl;
  std::cout << "pi = " << dPi << ", radToDeg = " << radToDeg << std::endl;

  edm::ESHandle<CSCGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get( pDD );     

  std::cout << " Geometry node for CSCGeom is  " << &(*pDD) << std::endl;   
  std::cout << " I have "<<pDD->detTypes().size()    << " detTypes" << std::endl;
  std::cout << " I have "<<pDD->detUnits().size()    << " detUnits" << std::endl;
  std::cout << " I have "<<pDD->dets().size()        << " dets" << std::endl;
  std::cout << " I have "<<pDD->layers().size()      << " layers" << std::endl;
  std::cout << " I have "<<pDD->chambers().size()    << " chambers" << std::endl;



  // Get the CSCIndexer algorithm from EventSetup

  edm::ESHandle<CSCIndexerBase> theIndexer;
  iSetup.get<CSCIndexerRecord>().get(theIndexer);

  algoName_ = theIndexer->name();

  std::cout << dashedLine_ << std::endl;
  std::cout << "Found CSCIndexer algorithm    " << myAlgo() << "    in EventSetup" << std::endl;
  std::cout << dashedLine_ << std::endl;


  bool ganged = 1;
  if ( myAlgo() == "CSCIndexerPostls1" ) ganged = 0;


  std::cout << myName() << ": Begin iteration over geometry..." << std::endl;
  std::cout << dashedLine_ << std::endl;

  std::cout << "\n  #     id(dec)      id(oct)                        "
    "lindex     lindex2      cindex       label       strip  sindex   strip  sindex   strip  sindex" << std::endl;


  int icount = 0;
  int icountAll = 0;

  // Iterate over the DetUnits in the CSCGeometry
  for( CSCGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); ++it ){

    // Check each DetUnit really is a CSC layer
    CSCLayer* layer = dynamic_cast<CSCLayer*>( *it );
     
    if( layer ) {
      ++icountAll; // how many layers we see

      // Get DetId in various ways

      DetId detId = layer->geographicalId();
      int id = detId(); // or detId.rawId()
      CSCDetId cscDetId = layer->id();

      // There is going to be a lot of messing with field width (and precision) so
      // save input values...
      int iw = std::cout.width(); // save current width
      int ip = std::cout.precision(); // save current precision

      short ie = CSCDetId::endcap(id);
      short is = CSCDetId::station(id);
      short ir = CSCDetId::ring(id);
      short ic = CSCDetId::chamber(id);
      short il = CSCDetId::layer(id);

      ++icount; 

      std::cout <<
	std::setw( 4 ) << icount << 
	std::setw(12) << id << std::oct << std::setw(12) << id << std::dec << std::setw( iw ) <<
	"   E" << ie << " S" << is << " R" << ir << " C" << std::setw( 2 ) << ic << std::setw( iw ) << " L" << il;

      unsigned cind = theIndexer->chamberIndex( ie, is, ir, ic );
      unsigned lind = theIndexer->layerIndex( ie, is, ir, ic, il );
      unsigned scind = theIndexer->startChamberIndexInEndcap( ie, is, ir ) + ic - 1;
      unsigned cind2 = theIndexer->chamberIndex( cscDetId );
      unsigned lind2 = theIndexer->layerIndex( cscDetId );

      std::cout << std::setw(12) << lind << std::setw(12) << lind2 << std::setw(12)<< scind << std::setw(12) <<  theIndexer->chamberLabelFromChamberIndex(scind) << "     " ;

      // Index a few strips
      unsigned short nchan = theIndexer->stripChannelsPerOnlineLayer(is,ir);
      unsigned int sc1 = theIndexer->stripChannelIndex(ie, is, ir, ic, il, 1);
      unsigned int scm = theIndexer->stripChannelIndex(ie, is, ir, ic, il, nchan/2);
      unsigned int scn = theIndexer->stripChannelIndex(ie, is, ir, ic, il, nchan);

      std::cout << "      1  " << std::setw(6) << sc1 << "      " << std::setw(2) << nchan/2 << "  " <<
	std::setw(6) << scm << "      " << std::setw(2) << nchan << "  " <<
	std::setw(6) << scn << std::endl;

      // Reset the values we changed
      std::cout << std::setprecision( ip ) << std::setw( iw );


  
      // ASSERTS
      // =======

      // Check layer indices are consistent  

      assert ( cind2 == cind );
      assert ( lind2 == lind );

      // Build CSCDetId from layer index and check it is same as original

      CSCDetId cscDetId2 = theIndexer->detIdFromLayerIndex( lind ); // folds ME1/1a into ME1/1

      testCSCDetId = cscDetId;

      if ( ir==4 ) testCSCDetId = CSCDetId(ie, is, 1, ic, il);

      assert( cscDetId2 == testCSCDetId );

      // Build CSCDetId from the strip-channel index for strip "nchan" and check it matches
      // Ganged ME1/1a returns ME1/1 CSCDetId and channel 65-80

      std::pair<CSCDetId, unsigned short int> p = theIndexer->detIdFromStripChannelIndex( scn );
      CSCDetId cscDetId3 = p.first;
      unsigned short iscn = p.second;

      if ( ir == 4 && ganged ) iscn -=64; // reset ganged ME1a channel from 65-80 to 1-16

      assert( iscn == nchan );

      if ( ir == 4 && !ganged ) testCSCDetId = cscDetId; // unganged ME1/1a needs its own CSCDetId

      assert( cscDetId3 == testCSCDetId );
          
      // Check idToDetUnit
      const GeomDetUnit * gdu = pDD->idToDetUnit(detId);
      assert(gdu==layer);
      // Check idToDet
      const GeomDet * gd = pDD->idToDet(detId);
      assert(gd==layer);
    }
    else {
      std::cout << myName() << ": something wrong ... could not dynamic_cast Det* to CSCLayer* " << std::endl;
    }
  }

  std::cout << dashedLine_ << std::endl;
  std::cout << myName() << " end." << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCIndexerAnalyzer2);
