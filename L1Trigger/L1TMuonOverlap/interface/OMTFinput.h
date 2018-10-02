#ifndef OMTF_OMTFinput_H
#define OMTF_OMTFinput_H

#include <vector>
#include <ostream>
#include <bitset>

class XMLConfigReader;
class OMTFConfiguration;

class OMTFinput{

 public:

  typedef std::vector<int> vector1D;
  typedef std::vector<vector1D> vector2D;

  OMTFinput(const OMTFConfiguration*);

  ///Add hit to given layer.
  ///iInput marks input number (max 14 per layer)
  bool addLayerHit(unsigned int iLayer,
		   unsigned int iInput,
		   int iPhi, int iEta, bool allowOverwrite=true);

  ///Reset vectors with data.
  void clear();

  ///Read data from a XML file
  void readData(XMLConfigReader *aReader, 
		unsigned int iEvent=0,
		unsigned int iProcessor=0);

  ///Apply shift to all data
  void shiftMyPhi(int phiShift);

  ///Merge data of two input objects.
  ///Method used in DiMuon studies.
  void mergeData(const OMTFinput *aInput);

  const OMTFinput::vector1D & getLayerData(unsigned int iLayer, bool giveEta=false) const;

  std::bitset<128> getRefHits(unsigned int iProcessor) const;

  friend std::ostream & operator << (std::ostream &out, const OMTFinput & aInput);

  OMTFinput & operator += (const OMTFinput & iInput) { mergeData(&iInput); return *this; }

 private:

  ///Phi measurements in logic layers
  ///First index: layer number
  ///Second index: measurement number within layer
  vector2D measurementsPhi;

  ///Eta measurements in logic layers
  ///First index: layer number
  ///Second index: measurement number within layer
  vector2D measurementsEta; 

  ///RefHitsEta
  mutable vector1D refHitsEta;

  const OMTFConfiguration* myOmtfConfig;

};


#endif
