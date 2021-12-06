#ifndef L1T_OmtfP1_OMTFinput_H
#define L1T_OmtfP1_OMTFinput_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStubsInput.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"

#include <vector>
#include <ostream>
#include <bitset>

class XMLConfigReader;
class OMTFConfiguration;

class OMTFinput : public MuonStubsInput {
public:
  OMTFinput(const OMTFConfiguration *);

  ~OMTFinput() override {}

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

  //if the layer is bending layer, the phiB from the iLayer -1 is returned
  int getPhiHw(unsigned int iLayer, unsigned int iInput) const override;

  //if the layer is bending layer, the eta from the iLayer -1 is returned
  const int getHitEta(unsigned int iLayer, unsigned int iInput) const;

  std::bitset<128> getRefHits(unsigned int iProcessor) const;

  friend std::ostream &operator<<(std::ostream &out, const OMTFinput &aInput);

private:
  const OMTFConfiguration *myOmtfConfig = nullptr;
};

#endif
