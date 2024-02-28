/*
 * PtAssignmentNN.cc
 *
 *  Created on: May 8, 2020
 *      Author: kbunkow
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "L1Trigger/L1TMuonOverlapPhase2/interface/PtAssignmentNNRegression.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <sstream>
#include <fstream>

namespace lutNN {
  static const int input_I = 10;
  static const int input_F = 4;
  static const std::size_t networkInputSize = 18;

  static const int layer1_neurons = 16;
  static const int layer1_lut_I = 3;
  static const int layer1_lut_F = 13;

  static const int layer1_output_I = 4;
  //4 bits are for the count of the noHit layers which goes to the input of the layer2
  static const int layer2_input_I = 8;

  static const int layer2_neurons = 9;
  static const int layer2_lut_I = 5;
  static const int layer2_lut_F = 11;

  static const int layer3_input_I = 5;

  static const int layer3_0_inputCnt = 8;
  static const int layer3_0_lut_I = 5;
  static const int layer3_0_lut_F = 11;
  static const int output0_I = 8;
  static const int output0_F = 2;

  static const int layer3_1_inputCnt = 1;
  static const int layer3_1_lut_I = 4;
  static const int layer3_1_lut_F = 11;
  static const int output1_I = 8;
  static const int output1_F = 0;  //Does not matter in principle - it is not used

  typedef LutNetworkFixedPointRegression2Outputs<input_I,
                                                 input_F,
                                                 networkInputSize,
                                                 layer1_lut_I,
                                                 layer1_lut_F,
                                                 layer1_neurons,  //layer1_lutSize = 2 ^ input_I
                                                 layer1_output_I,
                                                 layer2_input_I,
                                                 layer2_lut_I,
                                                 layer2_lut_F,
                                                 layer2_neurons,
                                                 layer3_input_I,
                                                 layer3_0_inputCnt,
                                                 layer3_0_lut_I,
                                                 layer3_0_lut_F,
                                                 output0_I,
                                                 output0_F,
                                                 layer3_1_inputCnt,
                                                 layer3_1_lut_I,
                                                 layer3_1_lut_F,
                                                 output1_I,
                                                 output1_F>
      LutNetworkFP;
}  // namespace lutNN

PtAssignmentNNRegression::PtAssignmentNNRegression(const edm::ParameterSet& edmCfg,
                                                   const OMTFConfiguration* omtfConfig,
                                                   std::string networkFile)
    : PtAssignmentBase(omtfConfig), lutNetworkFP(make_unique<lutNN::LutNetworkFP>()) {
  std::ifstream ifs(networkFile);

  edm::LogImportant("OMTFReconstruction")
      << " " << __FUNCTION__ << ":" << __LINE__ << " networkFile " << networkFile << std::endl;

  lutNetworkFP->load(networkFile);

  edm::LogImportant("OMTFReconstruction") << " " << __FUNCTION__ << ":" << __LINE__ << std::endl;
}

struct OmtfHit {
  union {
    unsigned long rawData = 0;

    struct {
      char layer;
      char quality;
      char z;
      char valid;
      short eta;
      short phiDist;
    };
  };

  OmtfHit(unsigned long rawData) : rawData(rawData) {}
};

bool omtfHitToEventInput(OmtfHit& hit, std::vector<float>& inputs, unsigned int omtfRefLayer, bool print) {
  float offset = (omtfRefLayer << 7) + 64;

  if (hit.valid) {
    if ((hit.layer == 1 || hit.layer == 3 || hit.layer == 5) && hit.quality < 4)  ///TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      return false;

    int rangeFactor = 2;  //rangeFactor scales the hit.phiDist such that the event->inputs is smaller then 63
    if (hit.layer == 1) {
      rangeFactor = 8;
    }
    /*else if(hit.layer == 8 || hit.layer == 17) {
            rangeFactor = 4;
        }*/
    else if (hit.layer == 3) {
      rangeFactor = 4;
    } else if (hit.layer == 9) {
      rangeFactor = 1;
    }
    /*else {
            rangeFactor = 2;
        }
		 */

    rangeFactor *= 2;  //TODO !!!!!!!!!!!!!!!!!!!

    if (std::abs(hit.phiDist) >= (63 * rangeFactor)) {
      edm::LogImportant("OMTFReconstruction")  //<<" muonPt "<<omtfEvent.muonPt<<" omtfPt "<<omtfEvent.omtfPt
          << " RefLayer " << omtfRefLayer << " layer " << int(hit.layer) << " hit.phiDist " << hit.phiDist << " valid "
          << ((short)hit.valid) << " !!!!!!!!!!!!!!!!!!!!!!!!" << endl;
      hit.phiDist = copysign(63 * rangeFactor, hit.phiDist);
    }

    inputs.at(hit.layer) = (float)hit.phiDist / (float)rangeFactor + offset;

    if (inputs.at(hit.layer) >=
        1022)  //the last address i.e. 1023 is reserved for the no-hit value, so interpolation between the 1022 and 1023 has no sense
      inputs.at(hit.layer) = 1022;

    if (print || inputs.at(hit.layer) < 0) {
      edm::LogImportant("OMTFReconstruction")  //<<"rawData "<<hex<<setw(16)<<hit.rawData
          << " layer " << dec << int(hit.layer);
      edm::LogImportant("OMTFReconstruction")
          << " phiDist " << hit.phiDist << " inputVal " << inputs.at(hit.layer) << " hit.z " << int(hit.z) << " valid "
          << ((short)hit.valid) << " quality " << (short)hit.quality << " omtfRefLayer " << omtfRefLayer;
      if (inputs.at(hit.layer) < 0)
        edm::LogImportant("OMTFReconstruction") << " event->inputs.at(hit.layer) < 0 !!!!!!!!!!!!!!!!!" << endl;
      edm::LogImportant("OMTFReconstruction") << endl;
    }

    if (inputs[hit.layer] >= 1024) {  //TODO should be the size of the LUT of the first layer
      edm::LogImportant("OMTFReconstruction") << " event->inputs[hit.layer] >= 1024 !!!!!!!!!!!!!!!!!" << endl;
    }
    return true;
  }

  return false;
}

std::vector<float> PtAssignmentNNRegression::getPts(AlgoMuons::value_type& algoMuon,
                                                    std::vector<std::unique_ptr<IOMTFEmulationObserver>>& observers) {
  LogTrace("l1tOmtfEventPrint") << " " << __FUNCTION__ << ":" << __LINE__ << std::endl;
  auto& gpResult = algoMuon->getGpResultConstr();
  //int pdfMiddle = 1<<(omtfConfig->nPdfAddrBits()-1);

  LogTrace("l1tOmtfEventPrint") << " " << __FUNCTION__ << ":" << __LINE__ << std::endl;
  /*
  edm::LogVerbatim("l1tOmtfEventPrint")<<"DataROOTDumper2:;observeEventEnd muonPt "<<event.muonPt<<" muonCharge "<<event.muonCharge
      <<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer<<" omtfPtCont "<<event.omtfPtCont
      <<std::endl;
*/

  unsigned int inputCnt = 18;  //TDOO!!!!!
  unsigned int outputCnt = 2;
  const float noHitVal = 1023.;

  //edm::LogImportant("OMTFReconstruction") <<"\n----------------------"<<endl;
  //edm::LogImportant("OMTFReconstruction") <<(*algoMuon)<<std::endl;

  std::vector<float> inputs(inputCnt, noHitVal);

  for (unsigned int iLogicLayer = 0; iLogicLayer < gpResult.getStubResults().size(); ++iLogicLayer) {
    auto& stubResult = gpResult.getStubResults()[iLogicLayer];
    if (stubResult.getMuonStub()) {  //&& stubResult.getValid() //TODO!!!!!!!!!!!!!!!!1
      int hitPhi = stubResult.getMuonStub()->phiHw;
      unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[algoMuon->getRefLayer()];
      int phiRefHit = gpResult.getStubResults()[refLayerLogicNum].getMuonStub()->phiHw;

      if (omtfConfig->isBendingLayer(iLogicLayer)) {
        hitPhi = stubResult.getMuonStub()->phiBHw;
        phiRefHit = 0;  //phi ref hit for the banding layer set to 0, since it should not be included in the phiDist
      }

      OmtfHit hit(0);

      hit.layer = iLogicLayer;
      hit.quality = stubResult.getMuonStub()->qualityHw;
      hit.eta = stubResult.getMuonStub()->etaHw;  //in which scale?
      hit.valid = stubResult.getValid();

      //phiDist = hitPhi - phiRefHit;
      hit.phiDist = hitPhi - phiRefHit;

      /*
      LogTrace("l1tOmtfEventPrint") <<" muonPt "<<event.muonPt<<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer
          <<" layer "<<int(hit.layer)<<" PdfBin "<<stubResult.getPdfBin()<<" hit.phiDist "<<hit.phiDist<<" valid "<<stubResult.getValid()<<" " //<<" phiDist "<<phiDist
          <<" getDistPhiBitShift "<<omtfCand->getGoldenPatern()->getDistPhiBitShift(iLogicLayer, omtfCand->getRefLayer())
          <<" meanDistPhiValue   "<<omtfCand->getGoldenPatern()->meanDistPhiValue(iLogicLayer, omtfCand->getRefLayer())//<<(phiDist != hit.phiDist? "!!!!!!!<<<<<" : "")
          <<endl;*/

      if (hit.phiDist > 504 || hit.phiDist < -512) {
        edm::LogVerbatim("l1tOmtfEventPrint")
            //<<" muonPt "<<event.muonPt<<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer
            << " layer " << int(hit.layer) << " hit.phiDist " << hit.phiDist << " valid " << stubResult.getValid()
            << " !!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
      }

      DetId detId(stubResult.getMuonStub()->detId);
      if (detId.subdetId() == MuonSubdetId::CSC) {
        CSCDetId cscId(detId);
        hit.z = cscId.chamber() % 2;
      }

      LogTrace("l1tOmtfEventPrint") << "hit: layer " << (int)hit.layer << " quality " << (int)hit.quality << " eta "
                                    << (int)hit.eta << " valid " << (int)hit.valid << " phiDist " << (int)hit.phiDist
                                    << " z " << (int)hit.z << std::endl;

      omtfHitToEventInput(hit, inputs, algoMuon->getRefLayer(), false);
    }
  }

  LogTrace("l1tOmtfEventPrint") << " " << __FUNCTION__ << ":" << __LINE__ << std::endl;

  std::vector<double> nnResult(outputCnt);
  lutNetworkFP->run(inputs, noHitVal, nnResult);

  LogTrace("l1tOmtfEventPrint") << " " << __FUNCTION__ << ":" << __LINE__ << std::endl;

  double pt = std::copysign(nnResult.at(0), nnResult.at(1));

  LogTrace("l1tOmtfEventPrint") << " " << __FUNCTION__ << ":" << __LINE__ << " nnResult.at(0) " << nnResult.at(0)
                                << " nnResult.at(1) " << nnResult.at(1) << " pt " << pt << std::endl;

  std::vector<float> pts;
  pts.emplace_back(pt);

  //algoMuon->setPtNN(omtfConfig->ptGevToHw(nnResult.at(0)));
  auto calibratedHwPt = lutNetworkFP->getCalibratedHwPt();
  algoMuon->setPtNNConstr(calibratedHwPt);

  algoMuon->setChargeNNConstr(nnResult[1] >= 0 ? 1 : -1);

  //TODO add some if here, such that the property_tree is filled only when needed
  boost::property_tree::ptree procDataTree;
  for (unsigned int i = 0; i < inputs.size(); i++) {
    auto& inputTree = procDataTree.add("input", "");
    inputTree.add("<xmlattr>.num", i);
    inputTree.add("<xmlattr>.val", inputs[i]);
  }

  std::ostringstream ostr;
  ostr << std::fixed << std::setprecision(19) << nnResult.at(0);
  procDataTree.add("output0.<xmlattr>.val", ostr.str());

  ostr.str("");
  ostr << std::fixed << std::setprecision(19) << nnResult.at(1);
  procDataTree.add("output1.<xmlattr>.val", ostr.str());

  procDataTree.add("calibratedHwPt.<xmlattr>.val", calibratedHwPt);

  procDataTree.add("hwSign.<xmlattr>.val", algoMuon->getChargeNNConstr() < 0 ? 1 : 0);

  for (auto& obs : observers)
    obs->addProcesorData("regressionNN", procDataTree);

  return pts;

  //event.print();
  /*
  std::vector<float> pts(classifierToRegressions.size(), 0);

  unsigned int i =0;
  for(auto& classifierToRegression : classifierToRegressions) {
    auto orgValue = classifierToRegression->getValue(&event);
    auto absOrgValue = std::abs(orgValue);
    pts.at(i) = classifierToRegression->getCalibratedValue(absOrgValue);
    pts.at(i) = std::copysign(pts.at(i), orgValue);

    LogTrace("OMTFReconstruction") <<" "<<__FUNCTION__<<":"<<__LINE__<<" orgValue "<<orgValue<<" pts["<<i<<"] "<<pts[i]<<std::endl;
    //std::cout<<"nn pts["<<i<<"] "<<pts[i]<< std::endl;
    i++;
  }

  return pts;*/
}
