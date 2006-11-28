#include "MagneticField/Engine/interface/MagneticField.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"

#include "TH1.h"
#include "TAxis.h"
#include <iostream>

using namespace std;

MagneticFieldMap*
MagneticFieldMap::myself=0; 

MagneticFieldMap* 
MagneticFieldMap::instance(const MagneticField* pMF,
			   TrackerInteractionGeometry* myGeo)
{
  if (!myself) myself = new MagneticFieldMap(pMF,myGeo);
  myself->initialize();
  return myself;
}

MagneticFieldMap* 
MagneticFieldMap::instance() {
  return myself;
}

MagneticFieldMap::MagneticFieldMap(const MagneticField* pMF,
				   TrackerInteractionGeometry* myGeo) : 
  pMF_(pMF), geometry_(myGeo) {;}

void
MagneticFieldMap::initialize()
{
  
  std::list<TrackerLayer>::iterator cyliter;
  std::list<TrackerLayer>::iterator cylitBeg=geometry_->cylinderBegin();
  std::list<TrackerLayer>::iterator cylitEnd=geometry_->cylinderEnd();
  
  // Prepare the histograms
  cout << "Prepare magnetic field local database for FAMOS speed-up" << endl;
  for ( cyliter=cylitBeg; cyliter != cylitEnd; ++cyliter ) {
    int layer = cyliter->layerNumber();
    //    cout << " Fill Histogram " << hist << endl;

    // Cylinder bounds
    double zmin = 0.;
    double zmax; 
    double rmin = 0.;
    double rmax; 
    if ( cyliter->forward() ) {
      zmax = cyliter->disk()->position().z();
      rmax = cyliter->disk()->outerRadius();
    } else {
      zmax = cyliter->cylinder()->bounds().length()/2.;
      rmax = cyliter->cylinder()->bounds().width()/2.
	   - cyliter->cylinder()->bounds().thickness()/2.;
    }

    // Histograms
    int bins=101;
    double step;

    // Disk histogram
    string histEndcap = Form("LayerEndCap_%u",layer);
    step = (rmax-rmin)/(bins-1);
    fieldEndcapHistos[layer] = 
      new TH1D(histEndcap.c_str(),"",bins,rmin,rmax+step);
    for ( double radius=rmin+step/2.; radius<rmax+step; radius+=step ) {
      double field = inTeslaZ(GlobalPoint(radius,0.,zmax));
      fieldEndcapHistos[layer]->Fill(radius,field);
    }

    // Barrel Histogram
    string histBarrel = Form("LayerBarrel_%u",layer);
    step = (zmax-zmin)/(bins-1);
    fieldBarrelHistos[layer] = 
      new TH1D(histBarrel.c_str(),"",bins,0.,zmax+step);
    for ( double zed=zmin+step/2.; zed<zmax+step; zed+=step ) {
      double field = inTeslaZ(GlobalPoint(rmax,0.,zed));
      fieldBarrelHistos[layer]->Fill(zed,field);
    }
  }
}


const GlobalVector
MagneticFieldMap::inTesla( const GlobalPoint& gp) const {

  if (!instance()) {
    return GlobalVector( 0., 0., 4.);
  } else {
    return pMF_->inTesla(gp);
  }

}

const GlobalVector
MagneticFieldMap::inTesla(const TrackerLayer& aLayer, double coord, int success) const {

  if (!instance()) {
    return GlobalVector( 0., 0., 4.);
  } else {
    return GlobalVector(0.,0.,inTeslaZ(aLayer,coord,success));
  }

}

const GlobalVector 
MagneticFieldMap::inKGauss( const GlobalPoint& gp) const {
  
  return inTesla(gp) * 10.;

}

const GlobalVector
MagneticFieldMap::inInverseGeV( const GlobalPoint& gp) const {
  
  return inKGauss(gp) * 2.99792458e-4;

} 

double 
MagneticFieldMap::inTeslaZ(const GlobalPoint& gp) const {

    return instance() ? pMF_->inTesla(gp).z() : 4.0;

}

double
MagneticFieldMap::inTeslaZ(const TrackerLayer& aLayer, double coord, int success) const 
{

  if (!instance()) {
    return 4.;
  } else {
    // Find the relevant histo
    TH1* theHisto; 
    if ( success == 1 ) 
      theHisto = fieldBarrelHistos.find(aLayer.layerNumber())->second;
    else
      theHisto = fieldEndcapHistos.find(aLayer.layerNumber())->second;
    
    // Find the relevant bin
    TAxis* theAxis = theHisto->GetXaxis();
    double x = fabs(coord);
    int bin = theAxis->FindBin(x);
    double binWidth = theHisto->GetBinWidth(bin);
    double x1 = theHisto->GetBinLowEdge(bin)+binWidth/2.;
    double x2 = x1+binWidth;

    // Determine the field
    double field1 = theHisto->GetBinContent(bin);
    double field2 = theHisto->GetBinContent(bin+1);

    //    if ( bin == 0 || bin == 1001 ) 
    //      std::cout << "WARNING bin = " << bin 
    //		<< " field " << field1 << " " << field2 
    //		<< std::endl;
	
    /*
    std::cout << "Layer " << aLayer.layerNumber() 
	      << " coord " << coord << " x " << x 
	      << " bin " << bin 
	      << " x1 " << x1 << " x2 " << x2 
	      << " field1 " << field1 << " field2 " << field2 << std::endl;
    */
    return field1 + (field2-field1) * (x-x1)/(x2-x1);
  }

}

double 
MagneticFieldMap::inKGaussZ(const GlobalPoint& gp) const {

    return inTeslaZ(gp)/10.;

}

double 
MagneticFieldMap::inInverseGeVZ(const GlobalPoint& gp) const {

   return inKGaussZ(gp) * 2.99792458e-4;

}

