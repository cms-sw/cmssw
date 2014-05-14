#ifndef HGCalCommonData_HGCalDDDConstants_h
#define HGCalCommonData_HGCalDDDConstants_h

/** \class HGCalDDDConstants
 *
 * this class reads the constant section of
 * the shashlik-numbering xml-file
 *  
 *  $Date: 2014/03/20 00:06:50 $
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include<string>
#include<vector>
#include<iostream>

#include "DetectorDescription/Core/interface/DDsvalues.h"

class DDCompactView;    
class DDFilteredView;

class HGCalDDDConstants {

public:

  struct hgtrap {
  hgtrap(float bl0, float tl0, float h0, float dz0, float alpha0): bl(bl0), tl(tl0), h(h0), dz(dz0), alpha(alpha0), cellSim(0), cellRec(0) {}
    float         bl, tl, h, dz, alpha, cellSim, cellRec;
  };

  HGCalDDDConstants();
  HGCalDDDConstants(const DDCompactView& cpv, std::string & name);
  ~HGCalDDDConstants();

  std::pair<int,int>  assignCell(float x, float y, int lay, int subSec,
				 bool reco) const;
  std::pair<int,int>  assignCell(float x, float y, float h, float bl, float tl,
				 float alpha, float cellSize) const;
  std::pair<int,int>  findCell(int cell, int lay, int subSec, bool reco) const;
  std::pair<int,int>  findCell(int cell, float h, float bl, float tl, 
			       float alpha, float cellSize) const;
  void                initialize(const DDCompactView& cpv, std::string name);
  unsigned int        layers(bool reco) const {return (reco ? depthIndex.size() : layerIndex.size());}
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
  int                 sectors() const {return nSectors;}
  std::pair<int,int>  simToReco(int cell, int layer, bool half) const;
       
  std::vector<hgtrap> getModules()  const { return modules_; }
  
private:
  void                checkInitialized() const;
  void                loadGeometry(const DDFilteredView& fv, std::string& tag);
  void                loadSpecPars(const DDFilteredView& fv);
  std::vector<double> getDDDArray(const std::string &, 
                                  const DDsvalues_type &, int &) const;
  std::pair<int,float> getIndex(int lay, bool reco) const;

  bool                tobeInitialized;
  int                 nCells, nSectors, nLayers;
  std::vector<double> cellSize_;
  std::vector<hgtrap> modules_;
  std::vector<int>    layer_, layerIndex;
  std::vector<int>    layerGroup_, cellFactor_, depth_, depthIndex;
};

#endif
