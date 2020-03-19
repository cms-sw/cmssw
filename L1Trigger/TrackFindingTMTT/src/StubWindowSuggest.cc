#include "L1Trigger/TrackFindingTMTT/interface/StubWindowSuggest.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace TMTT {

std::vector< double >                StubWindowSuggest::barrelCut_;
std::vector< std::vector< double > > StubWindowSuggest::ringCut_;
std::vector< std::vector< double > > StubWindowSuggest::tiltedCut_;
std::vector< double >                StubWindowSuggest::barrelNTilt_;

//=== Analyse stub window required for this stub.

void StubWindowSuggest::process(const Stub* stub) {

  // Half-size of FE chip bend window corresponding to Pt range in which tracks are to be found.
  const double invPtMax = 1/ptMin_;
  double bendHalfWind = invPtMax/fabs(stub->qOverPtOverBend());
  // Increase half-indow size to allow for resolution in bend.				       
  bendHalfWind += stub->bendResInFrontend();
  // Stub bend is measured here in half-integer values.
  bendHalfWind = int(2*bendHalfWind)/2.;

  // Compare with half-size of FE bend window stored in arrays.
  this->updateStoredWindow(stub, bendHalfWind);
}

//===  Update stored stub window size with this stub.

void StubWindowSuggest::updateStoredWindow(const Stub* stub, double bendHalfWind) {

  // Values set according to L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py
  // parameter NTiltedRings for whichever tracker geometry (T3, T4, T5 ...) is used..
  const vector<double> barrelNTilt_T5_init = {0., 12., 12., 12., 0., 0., 0.};
  if (stub->trackerGeometryVersion() == "T5") {
    barrelNTilt_ = barrelNTilt_T5_init;
  } else {
    throw cms::Exception("StubWindowSuggest: the tracker geometry you are using is not yet known to StubWindowSuggest. Please update constant barrelNTilt_T*_init inside it.")<<" Geometry="<<stub->trackerGeometryVersion()<<endl; 
  }

  // This code should be kept almost identical to that in  
  // L1Trigger/TrackTrigger/src/TTStubAlgorithm_official.cc
  // The only exceptions are lines marked "Modified by TMTT group"

  DetId stDetId(stub->idDet());

  // Modified by TMTT group, so we can update the numbers in the window size arrays.
  //int window = 0;

  if (stDetId.subdetId()==StripSubdetector::TOB)
  {
    unsigned int layer  = theTrackerTopo_->layer(stDetId);
    unsigned int ladder = theTrackerTopo_->tobRod(stDetId);
    int type   = 2*theTrackerTopo_->tobSide(stDetId)-3; // -1 for tilted-, 1 for tilted+, 3 for flat
    double corr=0;

    if (type<3) // Only for tilted modules
    {
      corr   = (barrelNTilt_.at(layer)+1)/2.;
      ladder = corr-(corr-ladder)*type; // Corrected ring number, bet 0 and barrelNTilt.at(layer), in ascending |z|
      // Modified by TMTT group, to expland arrays if necessary, divide by 2, & update the stored window sizes.
      if (tiltedCut_.size() < (layer+1)) tiltedCut_.resize(layer+1); 
      if (tiltedCut_.at(layer).size() < (ladder+1)) tiltedCut_.at(layer).resize(ladder+1, 0.); 
      double& storedHalfWindow = (tiltedCut_.at(layer)).at(ladder);
      if (storedHalfWindow < bendHalfWind) storedHalfWindow = bendHalfWind;
    }
    else // Classic barrel window otherwise
    {
      // Modified by TMTT group, to expland arrays if necessary, divide by 2, & update the stored window sizes.
      if (barrelCut_.size() < (layer+1)) barrelCut_.resize(layer+1, 0.); 
      double& storedHalfWindow = barrelCut_.at( layer );
      if (storedHalfWindow < bendHalfWind) storedHalfWindow = bendHalfWind;
    }
 
  }
  else if (stDetId.subdetId()==StripSubdetector::TID)
  {
    // Modified by TMTT group, to expland arrays if necessary, divide by 2, & update the stored window sizes
    unsigned int wheel = theTrackerTopo_->tidWheel(stDetId);
    unsigned int ring  = theTrackerTopo_->tidRing(stDetId);
    if (ringCut_.size() < (wheel+1)) ringCut_.resize(wheel+1); 
    if (ringCut_.at(wheel).size() < (ring+1)) ringCut_.at(wheel).resize(ring+1, 0.); 
    double& storedHalfWindow = ringCut_.at(wheel).at(ring);
    if (storedHalfWindow < bendHalfWind) storedHalfWindow = bendHalfWind;
  }
}

//=== Print results (should be done in endJob();

void StubWindowSuggest::printResults() {

  cout<<"=============================================================================="<<endl;
  cout<<" Stub window sizes that TMTT suggests putting inside "<<endl;   
  cout<<"   /L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py"<<endl;
  cout<<" (These should give good efficiency, but tighter windows may be needed to"<<endl;
  cout<<"  limit the data rate from the FE tracker electronics)."<<endl;
  cout<<"=============================================================================="<<endl;

  int old_precision = cout.precision();
  cout.precision(1); // Set significant digits for print.

  string str;

  str = "BarrelCut = cms.vdouble( ";
  for (const auto& cut : barrelCut_) {cout << str  << cut;   str = ", ";}
  cout<<"),"<<endl;

  cout << "TiltedBarrelCutSet = cms.VPSET(" << endl;
  cout << "     cms.PSet( TiltedCut = cms.vdouble( 0 ";
  for (const auto& cutVec : tiltedCut_) {
    str = "     cms.PSet( TiltedCut = cms.vdouble( ";
    for (const auto& cut : cutVec) {cout << str  << cut;   str = ", ";}
    cout<<") ),"<<endl;
  }
  cout<<"),"<<endl;

  cout << "EndcapCutSet = cms.VPSET(" << endl;
  cout << "     cms.PSet( EndcapCut = cms.vdouble( 0 ";
  for (const auto& cutVec : ringCut_) {
    str = "     cms.PSet( EndcapCut = cms.vdouble( ";
    for (const auto& cut : cutVec) {cout << str  << cut;   str = ", ";}
    cout<<") ),"<<endl;
  }
  cout<<"),"<<endl;

  cout.precision(old_precision);

  cout<<"=============================================================================="<<endl;
}

}
