#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Fireworks/Core/interface/FWGeometry.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include <TFile.h>
#include <TH1.h>

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <algorithm>

using namespace std;

template<class T>
typename enable_if<!numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return abs(x-y) <= numeric_limits<T>::epsilon() * abs(x+y) * ulp
        // unless the result is subnormal
        || abs(x-y) < numeric_limits<T>::min();
}

using namespace edm;

class DTGeometryValidate : public one::EDAnalyzer<> 
{
public:
  explicit DTGeometryValidate(const ParameterSet&);
  ~DTGeometryValidate() override {}
  
private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;  

  void validateDTChamberGeometry();
  void validateDTLayerGeometry();

  void compareTransform(const GlobalPoint&, const TGeoMatrix*);
  void compareShape(const GeomDet*, const float*);
  
  float getDistance(const GlobalPoint&, const GlobalPoint&);
  float getDiff(const float, const float);

  void makeHistograms(const char*);
  void makeHistogram(const string&, vector<float>&);

  void clearData() {
    globalDistances_.clear();
    topWidths_.clear();
    bottomWidths_.clear();
    lengths_.clear();
    thicknesses_.clear();
  }
  
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeometryToken_;
  edm::ESHandle<DTGeometry> dtGeometry_;
  FWGeometry                fwGeometry_;
  TFile*                    outFile_;
  vector<float> globalDistances_;
  vector<float> topWidths_;
  vector<float> bottomWidths_;
  vector<float> lengths_;
  vector<float> thicknesses_;
  string infileName_;
  string outfileName_;
  int    tolerance_;
};


DTGeometryValidate::DTGeometryValidate(const edm::ParameterSet& iConfig)
  : dtGeometryToken_{esConsumes<DTGeometry, MuonGeometryRecord>(edm::ESInputTag{})},
    infileName_(iConfig.getUntrackedParameter<string>("infileName", "cmsGeom10.root")),
    outfileName_(iConfig.getUntrackedParameter<string>("outfileName", "validateDTGeometry.root")),
    tolerance_(iConfig.getUntrackedParameter<int>("tolerance", 6))
{
  fwGeometry_.loadMap(infileName_.c_str());
  outFile_ = new TFile(outfileName_.c_str(), "RECREATE");
}

void 
DTGeometryValidate::analyze(const edm::Event& event, const edm::EventSetup& eventSetup)
{
  dtGeometry_ = eventSetup.getHandle(dtGeometryToken_);

  if(dtGeometry_.isValid()) {
    LogVerbatim("DTGeometry") << "Validating DT chamber geometry";
    validateDTChamberGeometry();

    LogVerbatim("DTGeometry") <<"Validating DT layer geometry";
    validateDTLayerGeometry();
  }
  else
    LogVerbatim("DTGeometry") << "Invalid DT geometry"; 
}

void 
DTGeometryValidate::validateDTChamberGeometry() {

  clearData();

  for(auto const& it : dtGeometry_->chambers()) {
    DTChamberId chId = it->id();
    GlobalPoint gp = it->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0)); 

    const TGeoMatrix* matrix = fwGeometry_.getMatrix(chId.rawId());
    
    if(!matrix) {
      LogVerbatim("DTGeometry") << "Failed to get matrix of DT chamber with detid: " 
                                << chId.rawId();
      continue;
    }

    compareTransform(gp, matrix);
    
    auto const& shape = fwGeometry_.getShapePars(chId.rawId());
    
    if(!shape) {
      LogVerbatim("DTGeometry") << "Failed to get shape of DT chamber with detid: "
                                << chId.rawId();
      continue;
    }
    
    compareShape(it, shape);
  }

  makeHistograms("DT Chamber");
}

void 
DTGeometryValidate::validateDTLayerGeometry() {

  clearData();
  
  vector<float> wire_positions;

  for(auto const& it : dtGeometry_->layers()) {
    DTLayerId layerId = it->id();
    GlobalPoint gp = it->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0)); 
     
    const TGeoMatrix* matrix = fwGeometry_.getMatrix(layerId.rawId());
 
    if (!matrix) {     
        LogVerbatim("DTGeometry") << "Failed to get matrix of DT layer with detid: " 
                                  << layerId.rawId();
        continue;
    }

    compareTransform(gp, matrix);

    auto const& shape = fwGeometry_.getShapePars(layerId.rawId());

    if(!shape) {
      LogVerbatim("DTGeometry") << "Failed to get shape of DT layer with detid: "
                                << layerId.rawId();
      continue;
    }
      
    compareShape(it, shape);
    
    auto const& parameters = fwGeometry_.getParameters(layerId.rawId());
      
    if(parameters == nullptr) {
      LogVerbatim("DTGeometry") << "Parameters empty for DT layer with detid: " 
				<< layerId.rawId();
      continue;
    }
           
    float width = it->surface().bounds().width();
    assert(width == parameters[6]); 

    float thickness = it->surface().bounds().thickness();
    assert(thickness == parameters[7]);

    float length = it->surface().bounds().length();
    assert(length == parameters[8]);

    int firstChannel = it->specificTopology().firstChannel();
    assert(firstChannel == parameters[3]);

    int lastChannel = it->specificTopology().lastChannel();
    int nChannels = parameters[5];
    assert(nChannels == (lastChannel-firstChannel)+1);

    for(int wireN = firstChannel; wireN - lastChannel <= 0; ++wireN) {
      float localX1 = it->specificTopology().wirePosition(wireN);
      float localX2 = (wireN -(firstChannel-1)-0.5f)*parameters[0] - nChannels/2.0f*parameters[0];
      wire_positions.emplace_back(getDiff(localX1, localX2));
    }
  }
  
  makeHistogram("DT Layer Wire localX", wire_positions);
  makeHistograms("DT Layer");
}

void
DTGeometryValidate::compareTransform(const GlobalPoint& gp,
				     const TGeoMatrix* matrix)
{
  double local[3] = { 0.0, 0.0, 0.0 };  
  double global[3];

  matrix->LocalToMaster(local, global);

  float distance = getDistance(GlobalPoint(global[0], global[1], global[2]), gp);
  globalDistances_.push_back(distance);
}

void 
DTGeometryValidate::compareShape(const GeomDet* det, const float* shape)
{
  float shapeTopWidth;
  float shapeBottomWidth;
  float shapeLength;
  float shapeThickness;

  if(shape[0] == 1) {
    shapeTopWidth = shape[2];
    shapeBottomWidth = shape[1];
    shapeLength = shape[4];
    shapeThickness = shape[3];
  }
  else if(shape[0] == 2) {
    shapeTopWidth = shape[1];
    shapeBottomWidth = shape[1];
    shapeLength = shape[2];
    shapeThickness = shape[3];
  }
  else {
    LogVerbatim("DTGeometry") << "Failed to get box or trapezoid from shape";
    return;
  }

  float topWidth, bottomWidth;
  float length, thickness;

  const Bounds* bounds = &(det->surface().bounds());
 
  if(const TrapezoidalPlaneBounds* tpbs = dynamic_cast<const TrapezoidalPlaneBounds*>(bounds))
  {
    array<const float, 4> const & ps = tpbs->parameters();
    
    assert(ps.size() == 4);
    
    bottomWidth = ps[0];
    topWidth = ps[1];
    thickness = ps[2];
    length = ps[3];
  }
  else if((dynamic_cast<const RectangularPlaneBounds*>(bounds))) {
    length = det->surface().bounds().length()*0.5;
    topWidth = det->surface().bounds().width()*0.5;
    bottomWidth = topWidth;
    thickness = det->surface().bounds().thickness()*0.5;
  }
  else {
    LogVerbatim("DTGeometry") << "Failed to get bounds";
    return;
  }
  topWidths_.push_back(fabs(shapeTopWidth - topWidth));
  bottomWidths_.push_back(fabs(shapeBottomWidth - bottomWidth));
  lengths_.push_back(fabs(shapeLength - length));
  thicknesses_.push_back(fabs(shapeThickness - thickness));
}

float
DTGeometryValidate::getDistance(const GlobalPoint& p1, const GlobalPoint& p2)
{
  return sqrt((p1.x()-p2.x())*(p1.x()-p2.x())+
              (p1.y()-p2.y())*(p1.y()-p2.y())+
              (p1.z()-p2.z())*(p1.z()-p2.z()));
}

float
DTGeometryValidate::getDiff(const float val1, const float val2) {
  if(almost_equal(val1, val2, tolerance_))
    return 0.0f;
  else
    return (val1 - val2);
}

void
DTGeometryValidate::makeHistograms(const char* detector)
{
  outFile_->cd();

  string d(detector);
  
  string gdn = d+": distance between points in global coordinates";
  makeHistogram(gdn, globalDistances_);

  string twn = d + ": absolute difference between top widths (along X)";
  makeHistogram(twn, topWidths_);

  string bwn = d + ": absolute difference between bottom widths (along X)";
  makeHistogram(bwn, bottomWidths_);

  string ln = d + ": absolute difference between lengths (along Y)";
  makeHistogram(ln, lengths_);

  string tn = d + ": absolute difference between thicknesses (along Z)";
  makeHistogram(tn, thicknesses_);
}

void
DTGeometryValidate::makeHistogram(const string& name, vector<float>& data)
{
  if(data.empty())
    return;

  const auto [minE, maxE] = minmax_element(begin(data), end(data));
  
  TH1D hist(name.c_str(), name.c_str(), 100, *minE*(1+0.10), *maxE*(1+0.10));

  for(auto const& it : data)
    hist.Fill(it);
  
  hist.GetXaxis()->SetTitle("[cm]");
  hist.Write();
}

void 
DTGeometryValidate::beginJob() {
  outFile_->cd();
}

void 
DTGeometryValidate::endJob() {
  LogVerbatim("DTGeometry") << "Done.";
  LogVerbatim("DTGeometry") << "Results written to "<< outfileName_;
  outFile_->Close();
}

DEFINE_FWK_MODULE(DTGeometryValidate);
