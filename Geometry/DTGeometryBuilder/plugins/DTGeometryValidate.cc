#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

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

template <class T>
typename enable_if<!numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return abs(x - y) <= numeric_limits<T>::epsilon() * abs(x + y) * ulp
         // unless the result is subnormal
         || abs(x - y) < numeric_limits<T>::min();
}

using namespace edm;

class DTGeometryValidate : public one::EDAnalyzer<> {
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

  bool hasPosRF(const int, const int);

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
  FWGeometry fwGeometry_;
  TFile* outFile_;
  vector<float> globalDistances_;
  vector<float> topWidths_;
  vector<float> bottomWidths_;
  vector<float> lengths_;
  vector<float> thicknesses_;
  string infileName_;
  string outfileName_;
  int tolerance_;
};

DTGeometryValidate::DTGeometryValidate(const edm::ParameterSet& iConfig)
    : dtGeometryToken_{esConsumes<DTGeometry, MuonGeometryRecord>(edm::ESInputTag{})},
      infileName_(iConfig.getUntrackedParameter<string>("infileName", "Geometry/DTGeometryBuilder/data/cmsRecoGeom-2021.root")),
      outfileName_(iConfig.getUntrackedParameter<string>("outfileName", "validateDTGeometry.root")),
      tolerance_(iConfig.getUntrackedParameter<int>("tolerance", 6)) {
  edm::FileInPath fp(infileName_.c_str());
  fwGeometry_.loadMap(fp.fullPath().c_str());
  outFile_ = TFile::Open(outfileName_.c_str(), "RECREATE");
}

void DTGeometryValidate::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  dtGeometry_ = eventSetup.getHandle(dtGeometryToken_);

  if (dtGeometry_.isValid()) {
    LogVerbatim("DTGeometry") << "Validating DT chamber geometry";
    validateDTChamberGeometry();
    LogVerbatim("DTGeometry") << "Validating DT layer geometry";
    validateDTLayerGeometry();
  } else
    LogVerbatim("DTGeometry") << "Invalid DT geometry";
}

bool DTGeometryValidate::hasPosRF(int wh,int sec){
    return  wh>0 || (wh==0 && sec%4>1);
}


void DTGeometryValidate::validateDTChamberGeometry() {
  clearData();

  //my-code                                                                                                                                                        

  double step=0.001; //v4 was produced with a step of 0.01

  double xcoordinate=0.;
  double pointX=0;


  for (auto const& it : dtGeometry_->chambers()) {
    DTChamberId chId = it->id();
    GlobalPoint gp = it->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));

    const TGeoMatrix* matrix = fwGeometry_.getMatrix(chId.rawId());


    double width=it->surface().bounds().width();


    int thisec = chId.sector();                                                                                                                                                                                                           

    if (thisec == 13)
        thisec = 4;
    if (thisec == 14)
        thisec = 10;


    DTLayerId SL1_layer2Id(chId,1,2);
    const int lastChannel_sl1_la2 = dtGeometry_->layer(SL1_layer2Id)->specificTopology().lastChannel();
    double localX_sl1_la2 = dtGeometry_->layer(SL1_layer2Id)->specificTopology().wirePosition(1);
    double localX_LC_sl1_la2 = dtGeometry_->layer(SL1_layer2Id)->specificTopology().wirePosition(lastChannel_sl1_la2);
    GlobalPoint gp_w1_sl1_la2 = dtGeometry_->layer(SL1_layer2Id)->surface().toGlobal(LocalPoint(localX_sl1_la2, 0.,-0.65)); //do we need to correct from the wire position to the sl position?                                             
    GlobalPoint gp_wLC_sl1_la2 = dtGeometry_->layer(SL1_layer2Id)->surface().toGlobal(LocalPoint(localX_LC_sl1_la2, 0.,-0.65));

    DTLayerId SL3_layer2Id(chId,3,2);
    const int lastChannel_sl3_la2 = dtGeometry_->layer(SL1_layer2Id)->specificTopology().lastChannel();
    double localX_sl3_la2 = dtGeometry_->layer(SL3_layer2Id)->specificTopology().wirePosition(1);
    double localX_LC_sl3_la2 = dtGeometry_->layer(SL3_layer2Id)->specificTopology().wirePosition(lastChannel_sl3_la2);
    GlobalPoint gp_w1_sl3_la2 = dtGeometry_->layer(SL3_layer2Id)->surface().toGlobal(LocalPoint(localX_sl3_la2, 0.,-0.65));
    GlobalPoint gp_wLC_sl3_la2 = dtGeometry_->layer(SL3_layer2Id)->surface().toGlobal(LocalPoint(localX_LC_sl3_la2, 0.,-0.65)); //z=--0.65?                                                                                                

    LocalPoint gp_w1_sl1_la2_inCH = it->surface().toLocal(gp_w1_sl1_la2);
    LocalPoint gp_w1_sl3_la2_inCH = it->surface().toLocal(gp_w1_sl3_la2);

    LocalPoint gp_wLC_sl1_la2_inCH = it->surface().toLocal(gp_wLC_sl1_la2);
    LocalPoint gp_wLC_sl3_la2_inCH = it->surface().toLocal(gp_wLC_sl3_la2);

    double delta=gp_w1_sl3_la2_inCH.x()-gp_w1_sl1_la2_inCH.x();
    double deltaZ=gp_w1_sl3_la2_inCH.z()-gp_w1_sl1_la2_inCH.z();

    DTWireId wire1_sl1_la2(SL1_layer2Id,1);
    DTWireId wire1_sl3_la2(SL3_layer2Id,1);

    std::cout<<"map \t chId:"<<chId<<" width:"<<width<<" "<<chId.rawId()<<" phi(0,0,0):"<<gp.phi()<<" secphi(0,0,0):"<<gp.phi()-0.5235988*(thisec-1)

             <<" Delta_SL3_SL1_inCH:"<<delta

             <<" DeltaZ_SL3-SL1_inCH:"<<deltaZ<<" = "<<gp_w1_sl3_la2_inCH.z()<<" - "<<gp_w1_sl1_la2_inCH.z()

             <<" gp_w1_sl1_la2_inCH.x:"<<gp_w1_sl1_la2_inCH.x()
             <<" gp_wLC_sl1_la2_inCH.x:"<<gp_wLC_sl1_la2_inCH.x()

             <<" gp_w1_sl1_la2.phi:"<<gp_w1_sl1_la2.phi()
             <<" gp_wLC_sl1_la2.phi:"<<gp_wLC_sl1_la2.phi()

             <<" gp_w1_sl1_la2.phisec:"<<gp_w1_sl1_la2.phi()-0.5235988*(thisec-1)
             <<" gp_wLC_sl1_la2.phisec:"<<gp_wLC_sl1_la2.phi()-0.5235988*(thisec-1)


             <<" gp_w1_sl3_la2_inCH.x:"<<gp_w1_sl3_la2_inCH.x()
             <<" gp_wLC_sl3_la2_inCH.x:"<<gp_wLC_sl3_la2_inCH.x()

             <<" gp_w1_sl3_la2.phi:"<<gp_w1_sl3_la2.phi()
             <<" gp_wLC_sl3_la2.phi:"<<gp_wLC_sl3_la2.phi()

             <<" gp_w1_sl3_la2.phisec:"<<gp_w1_sl3_la2.phi()-0.5235988*(thisec-1)
             <<" gp_wLC_sl3_la2.phisec:"<<gp_wLC_sl3_la2.phi()-0.5235988*(thisec-1)

             <<" shift_info_cmssw_jm "<<wire1_sl1_la2.rawId()<<" "<<gp_w1_sl1_la2_inCH.x()
             <<" shift_info_cmssw_jm "<<wire1_sl3_la2.rawId()<<" "<<gp_w1_sl3_la2_inCH.x()

             <<std::endl;

    //double left_limit=gp_w1_sl1_la2_inCH.x();//-2.1;                                                                                                                                                                                     
    //if(delta<0)                                                                                                                                                                                                                          
    //left_limit=gp_w1_sl3_la2_inCH.x()-2.1;                                                                                                                                                                                               

    //double z = 0.;                                                                                                                                                                                                                       
    //if (chId.station() >= 3)                                                                                                                                                                                                             
    //z = -1.8;                                                                                                                                                                                                                            

    double secphi=0;


    double min_secphi=999.;
    double min_pointX=999.;
    bool found=false;

    double distance_test=80.;

    //loop over SL1 line:----------------------------------------------------                                                                                                                                                              

    secphi=0;

    DTSuperLayerId SL1Id(chId,1);

    double widthSL1=dtGeometry_->layer(SL1_layer2Id)->surface().bounds().width();
    double widthSL1_with_wires=fabs( dtGeometry_->layer(SL1_layer2Id)->specificTopology().wirePosition(1) - dtGeometry_->layer(SL1_layer2Id)->specificTopology().wirePosition(lastChannel_sl1_la2) );

    double jm_point_layer_frame_sl1=dtGeometry_->layer(SL1_layer2Id)->specificTopology().wirePosition(1);

    found=false;

    min_secphi=999.;
    min_pointX=999.;

    for(pointX=-5.*widthSL1_with_wires;pointX<=5.*widthSL1_with_wires;pointX=pointX+step){
        xcoordinate=jm_point_layer_frame_sl1+pointX;
        GlobalPoint gp = dtGeometry_->layer(SL1_layer2Id)->surface().toGlobal(LocalPoint(xcoordinate,0.0, -0.65));
        secphi = gp.phi()-0.5235988*(thisec-1);
        if(fabs(secphi)<=fabs(min_secphi)){
            min_secphi=secphi;
            min_pointX=pointX;
            found =true;
        }
    }

    if(found==true){
        double perp=(dtGeometry_->layer(SL1_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl1+min_pointX,0.,-0.65))).perp();
	std::cout<<"phi_sector_0 "<<SL1Id.rawId()<<" "<<min_secphi
                 <<" confirmation "<<(dtGeometry_->layer(SL1_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl1+min_pointX,0.0, -0.65))).phi()-0.5235988*(thisec-1)
                 <<" slname "<<SL1Id
                 <<" perp "<<perp
                 <<" min_pointX "<<min_pointX
                 <<std::endl;

        //if(!hasPosRF(chId.wheel(),chId.sector())) min_pointX=min_pointX*(-1)+jm_point_layer_frame_sl1;                                                                                                                                   
        //if(!hasPosRF(chId.wheel(),chId.sector())) jm_point_layer_frame_sl1=jm_point_layer_frame_sl1*(-1);                                                                                                                                


        double geometry_position=((dtGeometry_->layer(SL1_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl1+distance_test,0.0, 0.0))).phi()-0.5235988*(thisec-1));

        double angle_arctan=TMath::ATan((min_pointX-distance_test)/perp);
        if(!hasPosRF(chId.wheel(),chId.sector())) angle_arctan*=-1;

        double diff_wrt_phisec0=angle_arctan-geometry_position;

        if(fabs(diff_wrt_phisec0)>0.02)std::cout<<"phisl0 warning diff_geo_cmssw= "<<diff_wrt_phisec0<<" "<<SL1Id<<" widthSL1:"<<widthSL1<<" widthSL1_with_wires:"<<widthSL1_with_wires<<std::endl;

        //if(hasPosRF(chId.wheel(),chId.sector()))std::cout<<"test_arctan(X_10cm): "<<diff_wrt_phisec0 <<std::endl;                                                                                                                        
	std::cout<<"test_arctan(X_10cm): "<<diff_wrt_phisec0 <<std::endl;

    }else{
	std::cout<<"phisl0 not found for "<<SL1Id
                 <<"looping from "<<-5.*widthSL1_with_wires<<" to "<<5.*widthSL1_with_wires
                 <<" or from phi="<<(dtGeometry_->layer(SL1_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl1-5.*widthSL1_with_wires,0.0, 0.0))).phi()-0.5235988*(thisec-1)
		 <<" to phi="<<(dtGeometry_->layer(SL1_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl1+5.*widthSL1_with_wires,0.0, 0.0))).phi()-0.5235988*(thisec-1)
                 <<std::endl;
    }





    //loop over SL3 line:-------------------------------------------------------------                                                                                                                                                     


    secphi=0;

    DTSuperLayerId SL3Id(chId,3);

    double widthSL3=dtGeometry_->layer(SL3_layer2Id)->surface().bounds().width();
    double widthSL3_with_wires=fabs( dtGeometry_->layer(SL3_layer2Id)->specificTopology().wirePosition(1) - dtGeometry_->layer(SL3_layer2Id)->specificTopology().wirePosition(lastChannel_sl3_la2) );

    double jm_point_layer_frame_sl3=dtGeometry_->layer(SL3_layer2Id)->specificTopology().wirePosition(1);

    found=false;

    min_secphi=999.;
    min_pointX=999.;

    for(pointX=-5.*widthSL3_with_wires;pointX<=5.*widthSL3_with_wires;pointX=pointX+step){
        xcoordinate=jm_point_layer_frame_sl3+pointX;
        GlobalPoint gp = dtGeometry_->layer(SL3_layer2Id)->surface().toGlobal(LocalPoint(xcoordinate,0.0, -0.65));
        secphi = gp.phi()-0.5235988*(thisec-1);
        if(fabs(secphi)<=fabs(min_secphi)){
            min_secphi=secphi;
            min_pointX=pointX;
            found =true;
        }
    }

    if(found==true){
        double perp=(dtGeometry_->layer(SL3_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl3+min_pointX,0.,-0.65))).perp();
	std::cout<<"phi_sector_0 "<<SL3Id.rawId()<<" "<<min_secphi
                 <<" confirmation "<<(dtGeometry_->layer(SL3_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl3+min_pointX,0.0, -0.65))).phi()-0.5235988*(thisec-1)
                 <<" slname "<<SL3Id
                 <<" perp "<<perp
                 <<" min_pointX "<<min_pointX
                 <<std::endl;

        //if(!hasPosRF(chId.wheel(),chId.sector())){min_pointX=min_pointX*(-1)+jm_point_layer_frame_sl3;                                                                                                                                   
        //if(!hasPosRF(chId.wheel(),chId.sector())) jm_point_layer_frame_sl3=jm_point_layer_frame_sl3*(-1);                                                                                                                                


        double geometry_position=((dtGeometry_->layer(SL3_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl3+distance_test,0.0, 0.0))).phi()-0.5235988*(thisec-1));

        double angle_arctan=TMath::ATan((min_pointX-distance_test)/perp);
        if(!hasPosRF(chId.wheel(),chId.sector())) angle_arctan*=-1.;

        double diff_wrt_phisec0=angle_arctan-geometry_position;

        if(fabs(diff_wrt_phisec0)>0.02)std::cout<<"phisl0 warning diff_geo_cmssw= "<<diff_wrt_phisec0<<" "<<SL3Id<<" widthSL3:"<<widthSL3<<" widthSL3_with_wires:"<<widthSL3_with_wires<<std::endl;

        //if(hasPosRF(chId.wheel(),chId.sector())) std::cout<<"test_arctan(X_10cm): "<<diff_wrt_phisec0 <<std::endl;                                                                                                                       
	std::cout<<"test_arctan(X_10cm): "<<diff_wrt_phisec0 <<std::endl;


    }else{
	std::cout<<"phisl0 not found for "<<SL3Id
                 <<"looping from "<<-5.*widthSL3_with_wires<<" to "<<5.*widthSL3_with_wires
                 <<" or from phi="<<(dtGeometry_->layer(SL3_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl3-5.*widthSL3_with_wires,0.0, 0.0))).phi()-0.5235988*(thisec-1)
		 <<" to phi="<<(dtGeometry_->layer(SL3_layer2Id)->surface().toGlobal(LocalPoint(jm_point_layer_frame_sl3+5.*widthSL3_with_wires,0.0, 0.0))).phi()-0.5235988*(thisec-1)
                 <<std::endl;
    }


    //my-code//          

    if (!matrix) {
      LogVerbatim("DTGeometry") << "Failed to get matrix of DT chamber with detid: " << chId.rawId();
      continue;
    }

    compareTransform(gp, matrix);

    auto const& shape = fwGeometry_.getShapePars(chId.rawId());

    if (!shape) {
      LogVerbatim("DTGeometry") << "Failed to get shape of DT chamber with detid: " << chId.rawId();
      continue;
    }

    compareShape(it, shape);
  }

  makeHistograms("DT Chamber");
}

void DTGeometryValidate::validateDTLayerGeometry() {
  clearData();

  vector<float> wire_positions;

  for (auto const& it : dtGeometry_->layers()) {
    DTLayerId layerId = it->id();
    GlobalPoint gp = it->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));

    const TGeoMatrix* matrix = fwGeometry_.getMatrix(layerId.rawId());

    if (!matrix) {
      LogVerbatim("DTGeometry") << "Failed to get matrix of DT layer with detid: " << layerId.rawId();
      continue;
    }

    compareTransform(gp, matrix);

    auto const& shape = fwGeometry_.getShapePars(layerId.rawId());

    if (!shape) {
      LogVerbatim("DTGeometry") << "Failed to get shape of DT layer with detid: " << layerId.rawId();
      continue;
    }

    compareShape(it, shape);

    auto const& parameters = fwGeometry_.getParameters(layerId.rawId());

    if (parameters == nullptr) {
      LogVerbatim("DTGeometry") << "Parameters empty for DT layer with detid: " << layerId.rawId();
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
    assert(nChannels == (lastChannel - firstChannel) + 1);

    for (int wireN = firstChannel; wireN - lastChannel <= 0; ++wireN) {
      float localX1 = it->specificTopology().wirePosition(wireN);
      float localX2 = (wireN - (firstChannel - 1) - 0.5f) * parameters[0] - nChannels / 2.0f * parameters[0];
      wire_positions.emplace_back(getDiff(localX1, localX2));
    }
  }

  makeHistogram("DT Layer Wire localX", wire_positions);
  makeHistograms("DT Layer");
}

void DTGeometryValidate::compareTransform(const GlobalPoint& gp, const TGeoMatrix* matrix) {
  double local[3] = {0.0, 0.0, 0.0};
  double global[3];

  matrix->LocalToMaster(local, global);

  float distance = getDistance(GlobalPoint(global[0], global[1], global[2]), gp);
  globalDistances_.push_back(distance);
}

void DTGeometryValidate::compareShape(const GeomDet* det, const float* shape) {
  float shapeTopWidth;
  float shapeBottomWidth;
  float shapeLength;
  float shapeThickness;

  if (shape[0] == 1) {
    shapeTopWidth = shape[2];
    shapeBottomWidth = shape[1];
    shapeLength = shape[4];
    shapeThickness = shape[3];
  } else if (shape[0] == 2) {
    shapeTopWidth = shape[1];
    shapeBottomWidth = shape[1];
    shapeLength = shape[2];
    shapeThickness = shape[3];
  } else {
    LogVerbatim("DTGeometry") << "Failed to get box or trapezoid from shape";
    return;
  }

  float topWidth, bottomWidth;
  float length, thickness;

  const Bounds* bounds = &(det->surface().bounds());

  if (const TrapezoidalPlaneBounds* tpbs = dynamic_cast<const TrapezoidalPlaneBounds*>(bounds)) {
    array<const float, 4> const& ps = tpbs->parameters();

    assert(ps.size() == 4);

    bottomWidth = ps[0];
    topWidth = ps[1];
    thickness = ps[2];
    length = ps[3];
  } else if ((dynamic_cast<const RectangularPlaneBounds*>(bounds))) {
    length = det->surface().bounds().length() * 0.5;
    topWidth = det->surface().bounds().width() * 0.5;
    bottomWidth = topWidth;
    thickness = det->surface().bounds().thickness() * 0.5;
  } else {
    LogVerbatim("DTGeometry") << "Failed to get bounds";
    return;
  }
  topWidths_.push_back(fabs(shapeTopWidth - topWidth));
  bottomWidths_.push_back(fabs(shapeBottomWidth - bottomWidth));
  lengths_.push_back(fabs(shapeLength - length));
  thicknesses_.push_back(fabs(shapeThickness - thickness));
}

float DTGeometryValidate::getDistance(const GlobalPoint& p1, const GlobalPoint& p2) {
  return sqrt((p1.x() - p2.x()) * (p1.x() - p2.x()) + (p1.y() - p2.y()) * (p1.y() - p2.y()) +
              (p1.z() - p2.z()) * (p1.z() - p2.z()));
}

float DTGeometryValidate::getDiff(const float val1, const float val2) {
  if (almost_equal(val1, val2, tolerance_))
    return 0.0f;
  else
    return (val1 - val2);
}

void DTGeometryValidate::makeHistograms(const char* detector) {
  outFile_->cd();

  string d(detector);

  string gdn = d + ": distance between points in global coordinates";
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

void DTGeometryValidate::makeHistogram(const string& name, vector<float>& data) {
  if (data.empty())
    return;

  const auto [minE, maxE] = minmax_element(begin(data), end(data));

  TH1D hist(name.c_str(), name.c_str(), 100, *minE * (1 + 0.10), *maxE * (1 + 0.10));

  for (auto const& it : data)
    hist.Fill(it);

  hist.GetXaxis()->SetTitle("[cm]");
  hist.Write();
}

void DTGeometryValidate::beginJob() { outFile_->cd(); }

void DTGeometryValidate::endJob() {
  LogVerbatim("DTGeometry") << "Done.";
  LogVerbatim("DTGeometry") << "Results written to " << outfileName_;
  outFile_->Close();
}

DEFINE_FWK_MODULE(DTGeometryValidate);
