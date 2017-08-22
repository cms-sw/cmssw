#ifndef Geometry_HcalTowerAlgo_HcalLayerDepthMap_h
#define Geometry_HcalTowerAlgo_HcalLayerDepthMap_h

/** \class HcalLayerDepthMap
 *
 * this class stores the map of layer to depth for special phi sections
 *  
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include<iostream>
#include<iomanip>
#include<map>
#include<string>
#include<vector>

class HcalLayerDepthMap {

public:

  HcalLayerDepthMap();
  ~HcalLayerDepthMap();
  void initialize(int subdet, int ietaMax, int dep16C, 
		  int dep29C, double wtl0C,
		  std::vector<int> const& iphi, std::vector<int> const& ieta,
		  std::vector<int> const& layer,std::vector<int> const& depth);
  int    getSubdet() const {return subdet_;}
  int    getDepth(int subdet, int ieta, int iphi, 
		  int zside, int layer) const;
  int    getDepth16(int subdet, int iphi, int zside) const;
  int    getDepthMin(int subdet, int iphi, int zside) const;
  int    getDepthMax(int subdet, int iphi, int zside) const;
  int    getDepthMax(int subdet, int ieta, int iphi, 
		     int zside) const;
  std::pair<int,int> getDepths(int eta) const;
  int    getLayerFront(int subdet, int ieta, int iphi,
		       int zside, int depth) const;
  int    getLayerBack(int subdet, int ieta, int iphi,
		      int zside, int depth) const;
  void   getLayerDepth(int subdet, int ieta, int iphi,
		       int zside, std::map<int,int>& layers) const;
  void   getLayerDepth(int ieta, std::map<int,int>& layers) const;
  double getLayer0Wt(int subdet, int iphi, int zside) const;
  int    getMaxDepthLastHE(int subdet, int iphi,
			   int zside) const;
  const std::vector<int> & getPhis() const {return iphi_;}
  bool   isValid(int det, int phi, int zside) const;
  int    validDet(std::vector<int>& phis) const;
  std::pair<int,int> validEta() const {return std::pair<int,int>(ietaMin_,ietaMax_);}
       
private:
  static const int                 maxLayers_ = 18;
  int                              subdet_;       // Subdet (HB=1, HE=2)
  int                              ietaMin_;      // Minimum eta value
  int                              ietaMax_;      // Maximum eta value
  int                              depthMin_;     // Minimum depth
  int                              depthMax_;     // Maximum depth
  int                              dep16C_;       // Max/Min layer # for HB/HE (ieta=16)
  int                              dep29C_;       // Max Depth of the last HE
  double                           wtl0C_;        // Layer 0 weight   
  std::vector<int>                 iphi_;         // phi*zside values
  std::map<std::pair<int,int>,int> layer2Depth_;  // Layer to depth map
  std::map<std::pair<int,int>,int> depth2LayerF_; // Depth to front layer map
  std::map<std::pair<int,int>,int> depth2LayerB_; // Depth to back  layer map
  std::map<int,std::pair<int,int>> depthsEta_;    // Depth range for each eta
};

#endif
