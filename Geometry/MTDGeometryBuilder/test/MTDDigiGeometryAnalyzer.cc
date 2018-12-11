// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetType.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// class declaration

class MTDDigiGeometryAnalyzer : public edm::one::EDAnalyzer<>
{
public:
  explicit MTDDigiGeometryAnalyzer( const edm::ParameterSet& ) {}
  ~MTDDigiGeometryAnalyzer() override = default;
  
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void analyseRectangle( const GeomDetUnit& det);
  void checkRotation( const GeomDetUnit& det);

};

// ------------ method called to produce the data  ------------
void
MTDDigiGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  edm::ESHandle<MTDTopology> mtdTopo;
  iSetup.get<MTDTopologyRcd>().get( mtdTopo );     
  
  //
  // get the MTDGeometry
  //
  edm::ESHandle<MTDGeometry> pDD;
  iSetup.get<MTDDigiGeometryRecord>().get( pDD );     
  edm::LogInfo("MTDDigiGeometryAnalyzer") << "Geometry node for MTDGeom is  " << &(*pDD) << "\n" 
                                          <<" # detectors = " <<pDD->detUnits().size() <<"\n"
                                          <<" # types     = " <<pDD->detTypes().size() << "\n";
  for(auto const & it : pDD->detUnits()){
    if(dynamic_cast<const MTDGeomDetUnit*>((it))!=nullptr){
      const BoundPlane& p = (dynamic_cast<const MTDGeomDetUnit*>((it)))->specificSurface();
      edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "---------------------------------------------------------- \n"
                                                  << mtdTopo->print(it->geographicalId()) 
                                                  <<" RadLeng Pixel "<<p.mediumProperties().radLen() 
                                                  <<" Xi Pixel "<<p.mediumProperties().xi();
      
      const GeomDetUnit theDet = *(dynamic_cast<const MTDGeomDetUnit*>(it));
      analyseRectangle( theDet );
      
    }
  }	
  
  for (auto const & it  :pDD->detTypes() ){
    if (dynamic_cast<const MTDGeomDetType*>((it))!=nullptr){
      const PixelTopology& p = (dynamic_cast<const MTDGeomDetType*>((it)))->specificTopology();
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(p);  
      edm::LogVerbatim("MTDDigiGeometryAnalyzer")<<"\n Subdetector " << it->subDetector() << " MTD Det " << it->name() << "\n"
                                             <<" Rows     "<<topo.nrows() <<" Columns "<<topo.ncolumns()
                                             <<" ROCS X   "<<topo.rocsX() <<" ROCS Y  "<<topo.rocsY()
                                             <<" Rows/ROC "<<topo.rowsperroc() 
                                             <<" Cols/ROC "<<topo.colsperroc();
    }
  }
}
void MTDDigiGeometryAnalyzer::analyseRectangle( const GeomDetUnit& det)
{

  const double safety = 0.9999;

  const Bounds& bounds = det.surface().bounds();
  const RectangularPlaneBounds* tb = dynamic_cast<const RectangularPlaneBounds*>(&bounds);
  if (tb == nullptr) return; // not trapezoidal

  const GlobalPoint& pos = det.position();
  double length = tb->length();
  double width = tb->width();
  double thickness = tb->thickness();

  GlobalVector yShift = det.surface().toGlobal( LocalVector( 0, 0, safety*length/2.));
  GlobalPoint outerMiddle = pos + yShift;
  GlobalPoint innerMiddle = pos + (-1.*yShift);
  if (outerMiddle.perp() < innerMiddle.perp()) std::swap( outerMiddle, innerMiddle);


  edm::LogVerbatim("MTDDigigeometryAnalyzer")  << "Det at pos " << pos << " radius " << std::sqrt(pos.x()*pos.x()+pos.y()*pos.y())
                                               <<" has length " << length <<" width " << width << " thickness " << thickness << "\n"
                                               << "det center inside bounds? " << tb->inside( det.surface().toLocal(pos)) << "\n"
                                               << "outerMiddle " << outerMiddle;
  
  checkRotation( det);

}


void MTDDigiGeometryAnalyzer::checkRotation( const GeomDetUnit& det)
{

  const double eps = std::numeric_limits<float>::epsilon();
  static int first = 0;
  if (first == 0) {
    edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "numeric_limits<float>::epsilon() " << std::numeric_limits<float>::epsilon();
    first =1;
  }

  const Surface::RotationType& rot( det.surface().rotation());
  GlobalVector a( rot.xx(), rot.xy(), rot.xz());
  GlobalVector b( rot.yx(), rot.yy(), rot.yz());
  GlobalVector c( rot.zx(), rot.zy(), rot.zz());
  GlobalVector cref = a.cross(b);
  GlobalVector aref = b.cross(c);
  GlobalVector bref = c.cross(a);
  if ((a-aref).mag() > eps || (b-bref).mag() > eps || (c-cref).mag() > eps) {
    edm::LogWarning("MTDDigiGeometryAnalyzer") << " Rotation not good by cross product: "
                                               << (a-aref).mag() << ", "
                                               << (b-bref).mag() << ", "
                                               << (c-cref).mag() 
                                               << " for det at pos " << det.surface().position();

  }
  if ( fabs(a.mag() - 1.) > eps || fabs(b.mag() - 1.) > eps || fabs(c.mag() - 1.) > eps ) {
    edm::LogWarning("MTDDigiGeometryAnalyzer") << " Rotation not good by bector mag: "
                                               << (a).mag() << ", "
                                               << (b).mag() << ", "
                                               << (c).mag() 
                                               << " for det at pos " << det.surface().position();
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDDigiGeometryAnalyzer);
