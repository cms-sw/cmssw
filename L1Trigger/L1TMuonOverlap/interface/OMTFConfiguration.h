#ifndef OMTF_OMTFConfiguration_H
#define OMTF_OMTFConfiguration_H

#include <map>
#include <set>
#include <vector>
#include <ostream>
#include <memory>

class L1TMuonOverlapParams;
class XMLConfigReader;

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

  static const OMTFConfiguration * instance(){ return latest_instance_; }

  OMTFConfiguration(const edm::ParameterSet & cfg);

  void configure(XMLConfigReader *aReader);

  void configure(const L1TMuonOverlapParams* omtfParams);

  void initCounterMatrices();
  
  friend std::ostream & operator << (std::ostream &out, const OMTFConfiguration & aConfig);

  unsigned int fwVersion;
  float minPdfVal;  
  unsigned int nLayers;
  unsigned int nHitsPerLayer;
  unsigned int nRefLayers;
  unsigned int nPhiBits;
  unsigned int nPdfAddrBits;
  unsigned int nPdfValBits;
  unsigned int nPhiBins;
  unsigned int nRefHits;
  unsigned int nTestRefHits;
  unsigned int nProcessors;
  unsigned int nLogicRegions;
  unsigned int nInputs;
  unsigned int nGoldenPatterns;
    
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
  typedef std::vector< std::pair<unsigned int, unsigned int> > vector1D_A;
  typedef std::vector<vector1D_A > vector2D_A;
  typedef std::vector<vector2D_A > vector3D_A;
  vector3D_A connections;

  ///Temporary hack to pass data from deep inside class
  ///Matrices are used during creation of the connections tables.
  typedef std::vector<int> vector1D;
  typedef std::vector<vector1D > vector2D;
  typedef std::vector<vector2D > vector3D;
  typedef std::vector<vector3D > vector4D;
  vector4D measurements4D;
  vector4D measurements4Dref;

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

  static OMTFConfiguration * latest_instance_;
};


#endif
 
