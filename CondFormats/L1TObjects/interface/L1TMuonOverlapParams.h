#ifndef L1TMTFOverlapParams_h
#define L1TMTFOverlapParams_h

#include <memory>
#include <iostream>
#include <vector>
#include <cmath>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/LUT.h"

///////////////////////////////////////
///////////////////////////////////////
class L1TMuonOverlapParams {
  
 public:
  
  class Node {
  public:
    std::string type_;
    unsigned version_;
    l1t::LUT LUT_;
    std::vector<double> dparams_;
    std::vector<unsigned> uparams_;
    std::vector<int> iparams_;
    std::vector<std::string> sparams_;
    Node(){ type_="unspecified"; version_=0; }
    COND_SERIALIZABLE;
  };

  class LayerMapNode {
  public:
    ///short layer number used within OMTF emulator
    unsigned int hwNumber;
    
    ///logic numer of the layer
    unsigned int logicNumber;

    ///Is this a bending layers?
    bool bendingLayer;

    ///Login number of layer to which this layer is tied.
    ///I.e both layers have to fire to account a hit
    unsigned int connectedToLayer;

    COND_SERIALIZABLE;
  };


  class RefLayerMapNode{

  public:

    ///Reference layer number 
    unsigned int refLayer;

    ///Corresponding logical layer number
    unsigned int logicNumber; 

    COND_SERIALIZABLE;
  };


  class RefHitNode{

  public:

    unsigned int iInput;
    int  iPhiMin, iPhiMax;
    unsigned int iRefHit;
    unsigned int iRefLayer;
    unsigned int iRegion;
    
    COND_SERIALIZABLE;
  };

  class LayerInputNode{

  public:

    unsigned int iFirstInput;
    unsigned int iLayer;
    unsigned int nInputs;

    COND_SERIALIZABLE;
  };
  
  
  
  enum { Version = 1 };
  
  // DO NOT ADD ENTRIES ANYWHERE BUT DIRECTLY BEFORE "NUM_OMTFPARAMNODES"
  enum { CHARGE=0, ETA=1, PT=2, PDF=3, MEANDISTPHI=4,
	 GENERAL = 5, SECTORS_START=6, SECTORS_END=7,
	 NUM_OMTFPARAMNODES=8};

  // General configuration parameters indexes
  enum {GENERAL_ADDRBITS=0, GENERAL_VALBITS=1, GENERAL_HITSPERLAYER=2, GENERAL_PHIBITS=3, GENERAL_PHIBINS=4, GENERAL_NREFHITS=5, GENERAL_NTESTREFHITS=6,
	GENERAL_NPROCESSORS=7, GENERAL_NLOGIC_REGIONS=8, GENERAL_NINPUTS=9, GENERAL_NLAYERS=10, GENERAL_NREFLAYERS=11, GENERAL_NGOLDENPATTERNS=12,
	GENERAL_NCONFIG=13
  };
	
  
  L1TMuonOverlapParams() { fwVersion_=Version; pnodes_.resize(NUM_OMTFPARAMNODES); }
  ~L1TMuonOverlapParams() {}

  // Firmware version
  unsigned fwVersion() const { return fwVersion_; }
  void setFwVersion(unsigned fwVersion) { fwVersion_ = fwVersion; }

  ///General definitions
  const std::vector<int>* generalParams()   const     { return &pnodes_[GENERAL].iparams_; }
  void setGeneralParams (const std::vector<int> & paramsVec) { pnodes_[GENERAL].type_ = "INT"; pnodes_[GENERAL].iparams_ = paramsVec;}

  ///Access to specific general settings.
  int nPdfAddrBits() const { return pnodes_[GENERAL].iparams_[GENERAL_ADDRBITS];};

  int nPdfValBits() const { return pnodes_[GENERAL].iparams_[GENERAL_VALBITS];};

  int nHitsPerLayer() const { return pnodes_[GENERAL].iparams_[GENERAL_HITSPERLAYER];};

  int nPhiBits() const { return pnodes_[GENERAL].iparams_[GENERAL_PHIBITS];};

  int nPhiBins() const { return pnodes_[GENERAL].iparams_[GENERAL_PHIBINS];};

  int nRefHits() const { return pnodes_[GENERAL].iparams_[GENERAL_NREFHITS];};
    
  int nTestRefHits() const { return pnodes_[GENERAL].iparams_[GENERAL_NTESTREFHITS];};

  int nProcessors() const { return pnodes_[GENERAL].iparams_[GENERAL_NPROCESSORS];};

  int nLogicRegions() const { return pnodes_[GENERAL].iparams_[GENERAL_NLOGIC_REGIONS];};

  int nInputs() const { return pnodes_[GENERAL].iparams_[GENERAL_NINPUTS];};

  int nLayers() const { return pnodes_[GENERAL].iparams_[GENERAL_NLAYERS];};

  int nRefLayers() const { return pnodes_[GENERAL].iparams_[GENERAL_NREFLAYERS];};

  int nGoldenPatterns() const { return pnodes_[GENERAL].iparams_[GENERAL_NGOLDENPATTERNS];};
    
  ///Connections definitions
  void setLayerMap(const  std::vector<LayerMapNode> &aVector) { layerMap_ = aVector;}

  void setRefLayerMap(const  std::vector<RefLayerMapNode> &aVector) { refLayerMap_ = aVector;}

  void setRefHitMap(const std::vector<RefHitNode> &aVector) {refHitMap_ = aVector;};

  void setGlobalPhiStartMap(const std::vector<int> &aVector) {globalPhiStart_ = aVector;};

  void setLayerInputMap(const std::vector<LayerInputNode> &aVector) {layerInputMap_ = aVector;};

  void setConnectedSectorsStart(const std::vector<int> &aVector){pnodes_[SECTORS_START].type_ = "INT"; pnodes_[SECTORS_START].iparams_ = aVector;};
  
  void setConnectedSectorsEnd(const std::vector<int> &aVector){pnodes_[SECTORS_END].type_ = "INT"; pnodes_[SECTORS_END].iparams_ = aVector;};
  
  const std::vector<LayerMapNode> * layerMap() const { return &layerMap_;};

  const std::vector<RefLayerMapNode> * refLayerMap() const { return &refLayerMap_;};

  const std::vector<RefHitNode> * refHitMap() const {return &refHitMap_;};
  
  const std::vector<int> * globalPhiStartMap() const { return &globalPhiStart_;};

  const std::vector<LayerInputNode> * layerInputMap() const { return &layerInputMap_;};

  const std::vector<int> * connectedSectorsStart() const { return &pnodes_[SECTORS_START].iparams_;};
  
  const std::vector<int> * connectedSectorsEnd() const { return &pnodes_[SECTORS_END].iparams_;};

 
  ///Golden Patterns definitions
  const l1t::LUT* chargeLUT() const { return &pnodes_[CHARGE].LUT_; }
  const l1t::LUT* etaLUT() const { return &pnodes_[ETA].LUT_; }
  const l1t::LUT* ptLUT()  const { return &pnodes_[PT].LUT_; }
  const l1t::LUT* pdfLUT() const { return &pnodes_[PDF].LUT_; }
  const l1t::LUT* meanDistPhiLUT() const { return &pnodes_[MEANDISTPHI].LUT_; }

  void setChargeLUT (const l1t::LUT & lut) { pnodes_[CHARGE].type_ = "LUT"; pnodes_[CHARGE].LUT_ = lut; }
  void setEtaLUT (const l1t::LUT & lut) { pnodes_[ETA].type_ = "LUT"; pnodes_[ETA].LUT_ = lut; }
  void setPtLUT (const l1t::LUT & lut) { pnodes_[PT].type_ = "LUT"; pnodes_[PT].LUT_ = lut; }
  void setPdfLUT (const l1t::LUT & lut) { pnodes_[PDF].type_ = "LUT"; pnodes_[PDF].LUT_ = lut; }
  void setMeanDistPhiLUT (const l1t::LUT & lut) { pnodes_[MEANDISTPHI].type_ = "LUT"; pnodes_[MEANDISTPHI].LUT_ = lut; }
  
  
 private:

  ///Version of firmware configuration
  unsigned fwVersion_;
    
  ///vector of LUT like parameters
  std::vector<Node> pnodes_;

  ///Vector of structs representing definitions of measurement layers.
  std::vector<LayerMapNode> layerMap_;

  ///Vector of structs representing definitins of reference layers
  ///in terms of logic measurement layers numbers.
  std::vector<RefLayerMapNode> refLayerMap_;

  ///Vector of RefHitNode defining assignenemt of
  ///reference hits to logical regions.
  ///definitions for all processor are serialized in a single vector.
  std::vector<RefHitNode> refHitMap_;

  ///Vector of global phi of processor beggining in each reference layer.
  ///All processors are serialized in a single vector.
  std::vector<int> globalPhiStart_;

  ///Vector of all definitions of input ranges for given
  ///logic region.
  ///All processors and all regions are serialized in a single vector.
  std::vector<LayerInputNode> layerInputMap_;
  
  COND_SERIALIZABLE;
};    
#endif
