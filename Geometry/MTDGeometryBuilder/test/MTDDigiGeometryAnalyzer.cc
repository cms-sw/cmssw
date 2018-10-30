// -*- C++ -*-
//
// Package:    MTDDigiGeometryAnalyzer
// Class:      MTDDigiGeometryAnalyzer
// 
/**\class MTDDigiGeometryAnalyzer MTDDigiGeometryAnalyzer.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filippo Ambroglini
//         Created:  Tue Jul 26 08:47:57 CEST 2005
//
//


// system include files
#include <memory>

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
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetType.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
//
// class decleration
//


// #define PRINT(X) edm::LogInfo(X)
#define PRINT(X) std::cout << X << ": "

class MTDDigiGeometryAnalyzer : public edm::one::EDAnalyzer<>
{
public:
      explicit MTDDigiGeometryAnalyzer( const edm::ParameterSet& );
      ~MTDDigiGeometryAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void analyseTrapezoidal( const GeomDetUnit& det);
  void checkRotation( const GeomDetUnit& det);
  std::ostream& cylindrical( std::ostream& os, const GlobalPoint& gp) const;
};

MTDDigiGeometryAnalyzer::MTDDigiGeometryAnalyzer( const edm::ParameterSet& iConfig )
{}

MTDDigiGeometryAnalyzer::~MTDDigiGeometryAnalyzer()
{}

// ------------ method called to produce the data  ------------
void
MTDDigiGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  PRINT("MTDDigiGeometryAnalyzer")<< "Here I am" << std::endl;
   //
   // get the TrackerGeom
   //
   edm::ESHandle<MTDGeometry> pDD;
   iSetup.get<MTDDigiGeometryRecord>().get( pDD );     
   PRINT("MTDDigiGeometryAnalyzer")<< " Geometry node for MTDGeom is  "<<&(*pDD) << std::endl;   
   PRINT("MTDDigiGeometryAnalyzer")<<" I have "<<pDD->detUnits().size() <<" detectors"<<std::endl;
   PRINT("MTDDigiGeometryAnalyzer")<<" I have "<<pDD->detTypes().size() <<" types"<<std::endl;

   for(auto const & it : pDD->detUnits()){
       if(dynamic_cast<const MTDGeomDetUnit*>((it))!=nullptr){
	const BoundPlane& p = (dynamic_cast<const MTDGeomDetUnit*>((it)))->specificSurface();
	PRINT("MTDDigiGeometryAnalyzer") << it->geographicalId()
              <<" RadLeng Pixel "<<p.mediumProperties().radLen()<<' ' <<" Xi Pixel "<<p.mediumProperties().xi()<<'\n';
       }        
    }	

   for (auto const & it  :pDD->detTypes() ){
     if (dynamic_cast<const MTDGeomDetType*>((it))!=nullptr){
       const PixelTopology& p = (dynamic_cast<const MTDGeomDetType*>((it)))->specificTopology();
       PRINT("MTDDigiGeometryAnalyzer")<<" PIXEL Det " // << it->geographicalId()
                      <<"    Rows    "<<p.nrows() <<"    Columns "<<p.ncolumns()<<'\n';
     }
   }
}
void MTDDigiGeometryAnalyzer::analyseTrapezoidal( const GeomDetUnit& det)
{

  // checkRotation( det);

  const double safety = 0.9999;

  const Bounds& bounds = det.surface().bounds();
  const TrapezoidalPlaneBounds* tb = dynamic_cast<const TrapezoidalPlaneBounds*>(&bounds);
  if (tb == nullptr) return; // not trapezoidal
  
  const GlobalPoint& pos = det.position();
  double length = tb->length();
  double width = tb->width();

  const std::array<const float, 4> & par = tb->parameters();
  double top = std::max(par[1], par[0]);
  double bot = std::min(par[1], par[0]);

  std::cout << std::endl;
  std::cout  << "Det at pos " << pos << " has length " << length
	<< " width " << width << " pars ";
  for (int i = 0; i<4; i++) std::cout << par[i] << ", ";
  std::cout << std::endl;

  std::cout << "det center inside bounds? " << tb->inside( det.surface().toLocal(pos)) << std::endl;

  //   double outerScale = (pos.perp()+safety*length/2.) / pos.perp();
  //   GlobalPoint outerMiddle = GlobalPoint( outerScale*pos.x(), outerScale*pos.y(), pos.z());

  GlobalVector yShift = det.surface().toGlobal( LocalVector( 0, safety*length/2., 0));
  GlobalPoint outerMiddle = pos + yShift;
  GlobalPoint innerMiddle = pos + (-1.*yShift);
  if (outerMiddle.perp() < innerMiddle.perp()) std::swap( outerMiddle, innerMiddle);

  GlobalVector upperShift = det.surface().toGlobal( LocalVector( safety*top, 0, 0));

  GlobalPoint ulc =  outerMiddle+upperShift;
  GlobalPoint urc =  outerMiddle+(-1.*upperShift);
  std::cout << "outerMiddle " << outerMiddle 
       << " upperShift " << upperShift 
       << " ulc " 
       << "(" << ulc.perp() 
       << "," << ulc.phi() 
       << "," << ulc.z()
       << " urc " 
       << "(" << urc.perp() 
       << "," << urc.phi() 
       << "," << urc.z()
       << std::endl;

  std::cout << "upper corners inside bounds? " 
       << tb->inside( det.surface().toLocal( ulc)) << " "
       << tb->inside( det.surface().toLocal( urc)) << std::endl;

  //   double innerScale = (pos.perp()-safety*length/2.) / pos.perp();
  //   GlobalPoint innerMiddle = GlobalPoint( innerScale*pos.x(), innerScale*pos.y(), pos.z());

  GlobalVector lowerShift = det.surface().toGlobal( LocalVector( safety*bot, 0, 0));

  std::cout << "lower corners inside bounds? " 
       << tb->inside( det.surface().toLocal( innerMiddle+lowerShift)) << " "
       << tb->inside( det.surface().toLocal( innerMiddle+(-1.*lowerShift))) << std::endl;

}


void MTDDigiGeometryAnalyzer::checkRotation( const GeomDetUnit& det)
{

  const double eps = std::numeric_limits<float>::epsilon();
  static int first = 0;
  if (first == 0) {
    std::cout << "numeric_limits<float>::epsilon() " << std::numeric_limits<float>::epsilon() << std::endl;
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
    std::cout << " Rotation not good by cross product: "
	 << (a-aref).mag() << ", "
	 << (b-bref).mag() << ", "
	 << (c-cref).mag() 
	 << " for det at pos " << det.surface().position() << std::endl;

  }
  if ( fabs(a.mag() - 1.) > eps || fabs(b.mag() - 1.) > eps || fabs(c.mag() - 1.) > eps ) {
    std::cout << " Rotation not good by bector mag: "
	 << (a).mag() << ", "
	 << (b).mag() << ", "
	 << (c).mag() 
	 << " for det at pos " << det.surface().position() << std::endl;
  }

}

std::ostream& MTDDigiGeometryAnalyzer::cylindrical( std::ostream& os, 
							const GlobalPoint& gp) const
{
  os << "(" << gp.perp() 
     << "," << gp.phi() 
     << "," << gp.z();
  return os;
}



//define this as a plug-in
DEFINE_FWK_MODULE(MTDDigiGeometryAnalyzer);
