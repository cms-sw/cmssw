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

  OMTFConfiguration(const edm::ParameterSet & cfg);

  void configure(XMLConfigReader *aReader);

  void configure(std::shared_ptr<L1TMuonOverlapParams> omtfParams);

  void initCounterMatrices();
  
  friend std::ostream & operator << (std::ostream &out, const OMTFConfiguration & aConfig);

  static float minPdfVal;  
  static unsigned int nLayers;
  static unsigned int nHitsPerLayer;
  static unsigned int nRefLayers;
  static unsigned int nPhiBits;
  static unsigned int nPdfAddrBits;
  static unsigned int nPdfValBits;
  static unsigned int nPhiBins;
  static unsigned int nRefHits;
  static unsigned int nTestRefHits;
  static unsigned int nProcessors;
  static unsigned int nLogicRegions;
  static unsigned int nInputs;
  static unsigned int nGoldenPatterns;
    
  static std::map<int,int> hwToLogicLayer;
  static std::map<int,int> logicToHwLayer;
  static std::map<int,int> logicToLogic;
  static std::set<int> bendingLayers;
  static std::vector<int> refToLogicNumber;

  ///Starting and final sectors connected to
  ///processors.
  ///Index: processor number
  static std::vector<unsigned int> barrelMin;
  static std::vector<unsigned int> barrelMax;
  static std::vector<unsigned int> endcap10DegMin;
  static std::vector<unsigned int> endcap10DegMax;
  static std::vector<unsigned int> endcap20DegMin;
  static std::vector<unsigned int> endcap20DegMax;
    
  ///Starting iPhi for each processor and each referecne layer    
  ///Global phi scale is used
  ///First index: processor number
  ///Second index: referecne layer number
  static std::vector<std::vector<int> > processorPhiVsRefLayer;

  ///Begin and end local phi for each processor and each reference layer    
  ///First index: processor number
  ///Second index: reference layer number
  ///Third index: region
  ///pair.first: starting phi of region (inclusive)
  ///pair.second: ending phi of region (inclusive)
  static std::vector<std::vector<std::vector<std::pair<int,int> > > >regionPhisVsRefLayerVsProcessor;

  ///Vector with definitions of reference hits
  ///Vector has fixed size of nRefHits
  ///Order of elements defines priority order
  ///First index: processor number (0-5)
  ///Second index: ref hit number (0-79)
  static std::vector<std::vector<RefHitDef> > refHitsDefs;

  ///Map of connections
  typedef std::vector< std::pair<unsigned int, unsigned int> > vector1D_A;
  typedef std::vector<vector1D_A > vector2D_A;
  typedef std::vector<vector2D_A > vector3D_A;
  static vector3D_A connections;

  ///Temporary hack to pass data from deep inside class
  ///Matrices are used during creation of the connections tables.
  typedef std::vector<int> vector1D;
  typedef std::vector<vector1D > vector2D;
  typedef std::vector<vector2D > vector3D;
  typedef std::vector<vector3D > vector4D;
  static vector4D measurements4D;
  static vector4D measurements4Dref;


  ///Find number of logic region within a given processor.
  ///Number is calculated assuming 10 deg wide logic regions
  ///Global phi scale is assumed at input.
  static unsigned int getRegionNumber(unsigned int iProcessor,
				    unsigned int iRefLayer,
				    int iPhi);

  ///Find logic region number using shifted, 10 bit
  ///phi values, and commection maps
  static unsigned int getRegionNumberFromMap(unsigned int iProcessor,
					     unsigned int iRefLayer,
					     int iPhi);
  
  ///Check if given referecne hit is
  ///in phi range for some logic cone.
  ///Care is needed arounf +Pi and +2Pi points
  static bool isInRegionRange(int iPhiStart,
			    unsigned int coneSize,
			    int iPhi);


  ///Return global phi for beggining of given processor
  ///Uses minim phi over all reference layers.
  static int globalPhiStart(unsigned int iProcessor);

  ///Return layer number encoding subsystem, and
  ///station number in a simple formula:
  /// aLayer+100*detId.subdetId()
  ///where aLayer is a layer number counting from vertex
  static uint32_t getLayerNumber(uint32_t rawId);

};


#endif
 
