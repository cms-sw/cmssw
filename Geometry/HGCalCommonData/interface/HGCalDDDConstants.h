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

#include<string>
#include<vector>
#include<iostream>
#include "CondFormats/GeometryObjects/interface/HGCalParameters.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"

class HGCalDDDConstants {

public:

  HGCalDDDConstants(const HGCalParameters* hp, const std::string name);
  ~HGCalDDDConstants();

  std::pair<int,int>  assignCell(float x, float y, int lay, int subSec,
				 bool reco) const;
  std::pair<int,int>  assignCell(float x, float y, float h, float bl, float tl,
				 float alpha, float cellSize) const;
  std::pair<int,int>  findCell(int cell, int lay, int subSec, bool reco) const;
  std::pair<int,int>  findCell(int cell, float h, float bl, float tl, 
			       float alpha, float cellSize) const;
  bool                isValid(int lay, int mod, int cell, bool reco) const;
  unsigned int        layers(bool reco) const {return (reco ? hgpar_->depthIndex_.size() : hgpar_->layerIndex_.size());}
  std::pair<float,float> locateCell(int cell, int lay, int subSec,
				    bool reco) const;
  int                 maxCells(bool reco) const;
  int                 maxCells(int lay, bool reco) const;
  int                 maxCells(float h, float bl, float tl, float alpha,
			       float cellSize) const;
  int                 maxRows(int lay, bool reco) const;
  std::pair<int,int>  newCell(int cell, int layer, int sector, int subsector,
			      int incrx, int incry, bool half) const;
  std::pair<int,int>  newCell(int cell, int layer, int subsector, int incrz,
			      bool half) const;
  int                 newCell(int kx, int ky, int lay, int subSec) const;
  std::vector<int>    numberCells(int lay, bool reco) const;
  std::vector<int>    numberCells(float h, float bl, float tl, float alpha,
				  float cellSize) const;
  int                 sectors() const {return hgpar_->nSectors_;}
  std::pair<int,int>  simToReco(int cell, int layer, bool half) const;

  std::vector<HGCalParameters::hgtrap>::const_iterator getFirstModule(bool reco=false) const {return (reco ? hgpar_->moduler_.begin() : hgpar_->modules_.begin());}
  std::vector<HGCalParameters::hgtrap>::const_iterator getLastModule(bool reco=false)  const {return (reco ? hgpar_->moduler_.end() : hgpar_->modules_.end());}
  std::vector<HGCalParameters::hgtrform>::const_iterator getFirstTrForm() const {return hgpar_->trform_.begin();}
  std::vector<HGCalParameters::hgtrform>::const_iterator getLastTrForm()  const {return hgpar_->trform_.end(); }

  const std::vector<HGCalParameters::hgtrap> & getModules() const {return hgpar_->moduler_; }
  const std::vector<HGCalParameters::hgtrform> & getTrForms() const {return hgpar_->trform_; }
  
private:
  std::pair<int,float> getIndex(int lay, bool reco) const;

  const HGCalParameters* hgpar_;
};

#endif
