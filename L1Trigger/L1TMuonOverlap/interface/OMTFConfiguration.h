#ifndef OMTF_OMTFConfiguration_H
#define OMTF_OMTFConfiguration_H

#include <map>
#include <set>
#include <vector>
#include <ostream>
#include <memory>

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

namespace edm{
  class ParameterSet;
}

class RefHitDef{

 public:

  //FIXME: default values should be sonnected to configuration values
  RefHitDef(unsigned int aInput=15, 
	    int aPhiMin=5760, 
	    int aPhiMax=5760,
	    unsigned int aRegion=99,
	    unsigned int aRefLayer=99);


 public:

  bool fitsRange(int iPhi) const;
  
  ///Hit input number within a cone
  unsigned int iInput;

  ///Region number assigned to this referecne hit
  unsigned int iRegion;

  ///Reference layer logic number (0-7)
  unsigned int iRefLayer;

  ///Local to processor phi range.
  ///Hit has to fit into this range to be assigned to this iRegion;
  std::pair<int, int> range;

  friend std::ostream & operator << (std::ostream &out, const RefHitDef & aRefHitDef);

};

class OMTFConfiguration{

 public:

  typedef std::vector< std::pair<unsigned int, unsigned int> > vector1D_pair;
  typedef std::vector<vector1D_pair > vector2D_pair;
  typedef std::vector<vector2D_pair > vector3D_pair;

  typedef std::vector<int> vector1D;
  typedef std::vector<vector1D > vector2D;
  typedef std::vector<vector2D > vector3D;
  typedef std::vector<vector3D > vector4D;

  OMTFConfiguration(){;};

  void configure(const L1TMuonOverlapParams* omtfParams);

  void initCounterMatrices();
  
   ///Find logic region number using first input number
  ///and then local phi value. The input and phi
  ///ranges are taken from DB. 
  unsigned int getRegionNumberFromMap(unsigned int iInput,
				      unsigned int iRefLayer,					     
				      int iPhi) const;
  
  ///Check if given referecne hit is
  ///in phi range for some logic cone.
  ///Care is needed arounf +Pi and +2Pi points
  bool isInRegionRange(int iPhiStart,
		       unsigned int coneSize,
		       int iPhi) const;

  ///Return global phi for beggining of given processor
  ///Uses minim phi over all reference layers.
  int globalPhiStart(unsigned int iProcessor) const;

  ///Return layer number encoding subsystem, and
  ///station number in a simple formula:
  /// aLayer+100*detId.subdetId()
  ///where aLayer is a layer number counting from vertex
  uint32_t getLayerNumber(uint32_t rawId) const;

  unsigned int fwVersion() const {return rawParams.fwVersion();};
  float minPdfVal() const {return 0.001;};
  unsigned int nLayers() const {return rawParams.nLayers();};
  unsigned int nHitsPerLayer() const {return rawParams.nHitsPerLayer();};
  unsigned int nRefLayers() const {return rawParams.nRefLayers();};
  unsigned int nPhiBits() const {return rawParams.nPhiBits();};
  unsigned int nPdfAddrBits() const {return rawParams.nPdfAddrBits();};
  unsigned int nPdfValBits() const {return rawParams.nPdfValBits();};
  unsigned int nPhiBins() const {return rawParams.nPhiBins();};
  unsigned int nRefHits() const {return rawParams.nRefHits();};
  unsigned int nTestRefHits() const {return rawParams.nTestRefHits();};
  unsigned int nProcessors() const {return rawParams.nProcessors();};
  unsigned int nLogicRegions() const {return rawParams.nLogicRegions();};
  unsigned int nInputs() const {return rawParams.nInputs();};
  unsigned int nGoldenPatterns() const {return rawParams.nGoldenPatterns();};

  const std::map<int,int>& getHwToLogicLayer() const {return hwToLogicLayer;}
  const std::map<int,int>& getLogicToHwLayer() const {return logicToHwLayer;}
  const std::map<int,int>& getLogicToLogic() const {return logicToLogic;}
  const std::set<int>& getBendingLayers() const {return bendingLayers;}
  const std::vector<int>& getRefToLogicNumber() const {return refToLogicNumber;}

  const std::vector<unsigned int>& getBarrelMin() const {return barrelMin;}
  const std::vector<unsigned int>& getBarrelMax() const {return barrelMax;}
  const std::vector<unsigned int>& getEndcap10DegMin() const {return endcap10DegMin;}
  const std::vector<unsigned int>& getEndcap10DegMax() const {return endcap10DegMax;}
  const std::vector<unsigned int>& getEndcap20DegMin() const {return endcap20DegMin;}
  const std::vector<unsigned int>& getEndcap20DegMax() const {return endcap20DegMax;}

  const std::vector<std::vector<int> >& getProcessorPhiVsRefLayer() const {return processorPhiVsRefLayer;}
  const std::vector<std::vector<std::vector<std::pair<int,int> > > >& getRegionPhisVsRefLayerVsInput() const {return regionPhisVsRefLayerVsInput;}
  const std::vector<std::vector<RefHitDef> >& getRefHitsDefs() const {return refHitsDefs;}

  const vector3D_pair & getConnections() const {return connections;};

  vector4D & getMeasurements4D() {return measurements4D;}
  vector4D & getMeasurements4Dref() {return measurements4Dref;}

  const vector4D & getMeasurements4D() const {return measurements4D;}
  const vector4D & getMeasurements4Dref() const {return measurements4Dref;}
  
  friend std::ostream & operator << (std::ostream &out, const OMTFConfiguration & aConfig);

 private:

  L1TMuonOverlapParams rawParams;
     
  std::map<int,int> hwToLogicLayer;
  std::map<int,int> logicToHwLayer;
  std::map<int,int> logicToLogic;
  std::set<int> bendingLayers;
  std::vector<int> refToLogicNumber;

  ///Starting and final sectors connected to
  ///processors.
  ///Index: processor number
   std::vector<unsigned int> barrelMin;
   std::vector<unsigned int> barrelMax;
   std::vector<unsigned int> endcap10DegMin;
   std::vector<unsigned int> endcap10DegMax;
   std::vector<unsigned int> endcap20DegMin;
   std::vector<unsigned int> endcap20DegMax;
    
  ///Starting iPhi for each processor and each referecne layer    
  ///Global phi scale is used
  ///First index: processor number
  ///Second index: referecne layer number
  std::vector<std::vector<int> > processorPhiVsRefLayer;
  
  ///Begin and end local phi for each logis region 
  ///First index: input number
  ///Second index: reference layer number
  ///Third index: region
  ///pair.first: starting phi of region (inclusive)
  ///pair.second: ending phi of region (inclusive)
  std::vector<std::vector<std::vector<std::pair<int,int> > > >regionPhisVsRefLayerVsInput;

  ///Vector with definitions of reference hits
  ///Vector has fixed size of nRefHits
  ///Order of elements defines priority order
  ///First index: processor number (0-5)
  ///Second index: ref hit number (0-127)
  std::vector<std::vector<RefHitDef> > refHitsDefs;

  ///Map of connections
  vector3D_pair connections;

  ///4D matrices used during creation of the connections tables.
  vector4D measurements4D;
  vector4D measurements4Dref;

};


#endif
 
