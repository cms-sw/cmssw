#ifndef HGCalCommonData_HGCalDDDConstants_h
#define HGCalCommonData_HGCalDDDConstants_h

/** \class HGCalDDDConstants
 *
 * this class reads the constant section of the numbering
 * xml-files of the  high granulairy calorimeter
 *  
 *  $Date: 2014/03/20 00:06:50 $
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include <string>
#include <vector>
#include <iostream>
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"

#include <unordered_map>

class HGCalDDDConstants {

public:

  HGCalDDDConstants(const HGCalParameters* hp, const std::string& name);
  ~HGCalDDDConstants();

  std::pair<int,int>  assignCell(float x, float y, int lay, int subSec,
				 bool reco) const;
  std::pair<int,int>  assignCellSquare(float x, float y, float h, float bl, 
				       float tl, float alpha, 
				       float cellSize) const;
  std::pair<int,int>  assignCellHexagon(float x, float y) const;
  double              cellSizeHex(int type) const;
  std::pair<int,int>  findCell(int cell, int lay, int subSec, bool reco) const;
  std::pair<int,int>  findCellSquare(int cell, float h, float bl, float tl, 
				     float alpha, float cellSize) const;
  HGCalGeometryMode::GeometryMode geomMode() const {return mode_;}
  bool                isValid(int lay, int mod, int cell, bool reco) const;
  bool                isValidCell(int layindex, int wafer, int cell) const;
  unsigned int        layers(bool reco) const;
  unsigned int        layersInit(bool reco) const;
  std::pair<float,float> locateCell(int cell, int lay, int type, 
				    bool reco) const;
  std::pair<float,float> locateCellHex(int cell, int wafer, bool reco) const;
  int                 levelTop() const {return hgpar_->levelT_;}
  int                 maxCells(bool reco) const;
  int                 maxCells(int lay, bool reco) const;
  int                 maxCellsSquare(float h, float bl, float tl, float alpha,
				     float cellSize) const;
  int                 maxModules() const {return modHalf_;}
  int                 maxRows(int lay, bool reco) const;
  double              minSlope() const {return hgpar_->slopeMin_;}
  int                 modules(int lay, bool reco) const;
  int                 modulesInit(int lay, bool reco) const;
  std::pair<int,int>  newCell(int cell, int layer, int sector, int subsector,
			      int incrx, int incry, bool half) const;
  std::pair<int,int>  newCell(int cell, int layer, int subsector, int incrz,
			      bool half) const;
  int                 newCell(int kx, int ky, int lay, int subSec) const;
  std::vector<int>    numberCells(int lay, bool reco) const;
  std::vector<int>    numberCellsSquare(float h, float bl, float tl, 
					float alpha, float cellSize) const;
  int                 numberCellsHexagon(int wafer) const;
  std::pair<int,int>  rowColumnWafer(const int wafer) const;
  int                 sectors() const {return hgpar_->nSectors_;}
  std::pair<int,int>  simToReco(int cell, int layer, int mod, bool half) const;
  unsigned int        volumes() const {return hgpar_->moduleLayR_.size();}
  int                 waferFromCopy(int copy) const;
  void                waferFromPosition(const double x, const double y,
					int& wafer, int& icell, 
					int& celltyp) const;
  bool                waferInLayer(int wafer, int lay, bool reco) const;
  int                 waferCount(const int type) const {return ((type == 0) ? waferMax_[2] : waferMax_[3]);}
  int                 waferMax() const {return waferMax_[1];}
  int                 waferMin() const {return waferMax_[0];}
  std::pair<double,double> waferPosition(int wafer, bool reco=true) const;
  int                 wafers() const;
  int                 wafers(int layer, int type) const;
  int                 waferToCopy(int wafer) const {return ((wafer>=0)&&(wafer< (int)(hgpar_->waferCopy_.size()))) ? hgpar_->waferCopy_[wafer] : (int)(hgpar_->waferCopy_.size());}
  // wafer transverse thickness classification (2 = coarse, 1 = fine)
  int                 waferTypeT(int wafer) const {return ((wafer>=0)&&(wafer<(int)(hgpar_->waferTypeT_.size()))) ? hgpar_->waferTypeT_[wafer] : 0;}
  // wafer longitudinal thickness classification (1 = 100um, 2 = 200um, 3=300um)
  int                 waferTypeL(int wafer) const {return ((wafer>=0)&&(wafer<(int)(hgpar_->waferTypeL_.size()))) ? hgpar_->waferTypeL_[wafer] : 0;}
  bool                isHalfCell(int waferType, int cell) const;
  double              waferZ(int layer, bool reco) const;

  HGCalParameters::hgtrap getModule(unsigned int k, bool hexType, bool reco) const;
  std::vector<HGCalParameters::hgtrap> getModules() const; 

  unsigned int getTrFormN() const {return hgpar_->trformIndex_.size();}
  HGCalParameters::hgtrform getTrForm(unsigned int k) const {return hgpar_->getTrForm(k);}
  std::vector<HGCalParameters::hgtrform> getTrForms() const ;
  
  std::pair<int,float> getIndex(int lay, bool reco) const;

private:
  int cellHex(double xx, double yy, const double& cellR, 
	      const std::vector<double>& posX,
	      const std::vector<double>& posY) const;  
  void getParameterSquare(int lay, int subSec, bool reco, float& h, float& bl,
			  float& tl, float& alpha) const;
  bool waferInLayer(int wafer, int lay) const;

  typedef std::array<std::vector<int32_t>, 2> Simrecovecs;  
  typedef std::array<int,3>                   HGCWaferParam;
  const HGCalParameters*          hgpar_;
  constexpr static double         tan30deg_ = 0.5773502693;
  double                          rmax_, hexside_;
  HGCalGeometryMode::GeometryMode mode_;
  int32_t                         tot_wafers_, modHalf_;
  std::array<uint32_t,2>          tot_layers_;
  Simrecovecs                     max_modules_layer_;
  std::map<int,HGCWaferParam>     waferLayer_;
  std::array<int,4>               waferMax_;
};

#endif
