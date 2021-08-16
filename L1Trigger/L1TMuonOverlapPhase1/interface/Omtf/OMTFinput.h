#ifndef OMTF_OMTFinput_H
#define OMTF_OMTFinput_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStubsInput.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"

#include <vector>
#include <ostream>
#include <bitset>

class XMLConfigReader;
class OMTFConfiguration;

class OMTFinput : public MuonStubsInput {
public:
  //typedef std::vector<int> vector1D;
  //typedef std::vector<vector1D> vector2D;

  OMTFinput(const OMTFConfiguration *);

  ~OMTFinput() override {}

  ///Add hit to given layer.
  ///iInput marks input number (max 14 per layer)
  /*
  bool addLayerHit(unsigned int iLayer,
		   unsigned int iInput,
		   int iPhi, int iEta, bool allowOverwrite=true);
*/

  ///Reset vectors with data.
  //void clear();

  ///Read data from a XML file
  void readData(XMLConfigReader *aReader, unsigned int iEvent = 0, unsigned int iProcessor = 0);

  ///Apply shift to all data
  void shiftMyPhi(int phiShift);

  ///Merge data of two input objects.
  ///Method used in DiMuon studies.
  void mergeData(const OMTFinput *aInput);

  const MuonStubPtr getMuonStub(unsigned int iLayer, unsigned int iInput) const {
    return muonStubsInLayers.at(iLayer).at(iInput);
  }

  //if the layer is banding layer, the phiB from the iLayer -1 is returned
  int getPhiHw(unsigned int iLayer, unsigned int iInput) const override;

  //if the layer is banding layer, the eta from the iLayer -1 is returned
  const int getHitEta(unsigned int iLayer, unsigned int iInput) const;

  std::bitset<128> getRefHits(unsigned int iProcessor) const;

  friend std::ostream &operator<<(std::ostream &out, const OMTFinput &aInput);

  //OMTFinput & operator += (const OMTFinput & iInput) { mergeData(&iInput); return *this; }

private:
  //friend class OMTFinputMaker; //to allow the OMTFinputMaker see and modify the muonStubsInLayers, which for other classes should be const

  ///Phi measurements in logic layers
  ///First index: layer number
  ///Second index: measurement number within layer
  //vector2D measurementsPhi;

  ///Eta measurements in logic layers
  ///First index: layer number
  ///Second index: measurement number within layer
  //vector2D measurementsEta;

  ///RefHitsEta
  //mutable vector1D refHitsEta;

  const OMTFConfiguration *myOmtfConfig = nullptr;
};

#endif
