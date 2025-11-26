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

#include "L1Trigger/L1TMuonOverlapPhase2/interface/LutNetworkFixedPointRegressionMultipleOutputs.h"
#include "DataFormats/L1TMuonPhase2/interface/Constants.h"
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <L1Trigger/L1TMuonOverlapPhase2/interface/NNRegression.h>

#include <sstream>
#include <fstream>

namespace lutNN {
  const int input_I = 10;
  const int input_F = 7;
  const std::size_t networkInputSize = 18;

  const int layer1_neurons = 16;
  const int layer1_lut_I = 3;
  const int layer1_lut_F = 9;

  const int layer1_output_I = 4;
  const int layer1_output_F = layer1_lut_F + 2;

  //4 bits are for the count of the noHit layers which goes to the input of the layer2
  const int layer2_input_I = layer1_output_I + 4;

  const int layer2_neurons = 8 * 2 + 1 + 3;  //8 neurons for pt0, pt1 (each), 1 for charge, 3s for p_displ
  const int layer2_lut_I = 5;
  const int layer2_lut_F = 7;

  const int layer3_input_I = 5;
  const int layer3_input_F = layer2_lut_F + 2;

  const int layer3_0_inputCnt = 8;
  const int layer3_0_lut_I = 8;
  const int layer3_0_lut_F = 5 + 5;
  const int output0_I = 8;
  const int output0_F = 5 + 5;  //layer3_0_lut_F + 2;
  const int layer3_0_multiplicity = 2;

  const int layer3_1_inputCnt = 1;
  const int layer3_1_lut_I = 8;  //TODO it should be smaller than 4 bits
  const int layer3_1_lut_F = 5 + 5;

  const int output1_I = 8;
  const int output1_F = 5 + 5;
  const int layer3_1_multiplicity = 4;

  typedef LutNetworkFixedPointRegressionMultipleOutputs<input_I,
                                                        input_F,
                                                        networkInputSize,
                                                        layer1_lut_I,
                                                        layer1_lut_F,
                                                        layer1_neurons,  //layer1_lutSize = 2 ^ input_I
                                                        layer1_output_I,
                                                        layer1_output_F,
                                                        layer2_input_I,
                                                        layer2_lut_I,
                                                        layer2_lut_F,
                                                        layer2_neurons,
                                                        layer3_input_I,
                                                        layer3_input_F,
                                                        layer3_0_inputCnt,
                                                        layer3_0_lut_I,
                                                        layer3_0_lut_F,
                                                        output0_I,
                                                        output0_F,
                                                        layer3_0_multiplicity,
                                                        layer3_1_inputCnt,
                                                        layer3_1_lut_I,
                                                        layer3_1_lut_F,
                                                        output1_I,
                                                        output1_F,
                                                        layer3_1_multiplicity>
      LutNetworkFP;
}  // namespace lutNN

NNRegression::NNRegression(const edm::ParameterSet& edmCfg,
                           const OMTFConfiguration* omtfConfig,
                           std::string networkFile)
    : MlModelBase(omtfConfig), lutNetworkFP(make_unique<lutNN::LutNetworkFP>()) {
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
      short deltaR;
      short phiDist;
    };
  };

  OmtfHit(unsigned long rawData) : rawData(rawData) {}
};

bool omtfHitWithQualAndRToEventInput(OmtfHit& hit, std::vector<float>& inputs, unsigned int omtfRefLayer, bool print) {
  int lustSize = 1024;  //TODO change it if needed
  int refLayers = 8;
  float rangeSize = lustSize / (refLayers * 2);
  //float offset = (omtfRefLayer<<7) + rangeMiddle;

  //two ranges for each omtfRefLayer, so that two qualites can be used for each omtfRefLayer
  float offset = omtfRefLayer * rangeSize * 2 + rangeSize / 2;
  int rangeFactor = 2;  //rangeFactor scales the hit.phiDist such that the event->inputs is smaller then 63

  //if(!hit.valid)
  //    return false; ///TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  if (hit.layer <= 5) {  //DT hits
    rangeFactor = 2;     //rangeFactor scales the hit.phiDist such that the event->inputs is smaller then 63
    //two ranges for each omtfRefLayer, so that two qualites can be used for each omtfRefLayer
    offset = omtfRefLayer * rangeSize * 2 + rangeSize / 2;
    if ((hit.layer == 1 || hit.layer == 3 || hit.layer == 5)) {  //phiB
      //if(!hit.valid)
      //    return false; ///TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

      if (hit.quality < 2)  ///TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        return false;

      if (hit.layer == 1) {
        rangeFactor = 8 * 2;
      } else if (hit.layer == 3) {
        rangeFactor = 8 * 2;
      }

      else if (hit.layer == 5) {
        rangeFactor = 8 * 2;
      }
    } else {  //phi
      rangeFactor *= 4;
    }

    if (hit.quality >= 4) {
      offset += rangeSize;
      //rangeFactor *= 4;
    }

  } else if ((hit.layer >= 6 && hit.layer <= 9) || (hit.layer >= 15)) {  //CSC hits and RPCe hits
    int rBins = 16;
    rangeSize = lustSize / (refLayers * rBins);
    rangeFactor = 4;
    int rBin = std::abs(hit.deltaR) >> 4;
    if (rBin >= rBins) {
      //cout<<"rBin "<<rBin<<" hit.eta "<<hit.eta<<" hit.layer "<<(int)hit.layer<<" omtfRefLayer "<<omtfRefLayer<<endl;
      rBin = rBins - 1;
    }

    offset = (omtfRefLayer << 7) + (rBin << 3) + rangeSize / 2;

    if (hit.layer == 9)
      rangeFactor = 4;

    rangeFactor = rangeFactor * rBins;

  } else if (hit.layer >= 10 || hit.layer <= 14) {  //RPCb hits
    rangeFactor *= 4;
  }

  rangeFactor *= 2;  //TODO !!!!!!!!!!!!!!!!!!!

  if (abs(hit.phiDist) >= ((rangeSize / 2 - 1) * rangeFactor)) {
    if (hit.valid)
      cout  //<<" muonPt "<<omtfEvent.muonPt<<" omtfPt "<<omtfEvent.omtfPt
          << " RefLayer " << omtfRefLayer << " layer " << int(hit.layer) << " hit.phiDist " << hit.phiDist << " valid "
          << ((short)hit.valid) << " quality " << ((short)hit.quality)
          << " hit.phiDist outside the range !!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    hit.phiDist = copysign((rangeSize / 2 - 1) * rangeFactor, hit.phiDist);
  }

  inputs.at(hit.layer) = (float)hit.phiDist / (float)rangeFactor + offset;

  //the last address i.e. 1023 is reserved for the no-hit value, so interpolation between the 1022 and 1023 has no sense
  if (inputs.at(hit.layer) >= lustSize - 2)
    inputs.at(hit.layer) = lustSize - 2;

  if (print || inputs.at(hit.layer) < 0) {
    cout  //<<"rawData "<<hex<<setw(16)<<hit.rawData
        << " layer " << dec << int(hit.layer);
    cout << " phiDist " << hit.phiDist << " inputVal " << inputs.at(hit.layer) << " hit.z " << int(hit.z) << " valid "
         << ((short)hit.valid) << " quality " << (short)hit.quality << " omtfRefLayer " << omtfRefLayer << " offset "
         << offset;
    if (inputs.at(hit.layer) < 0)
      cout << " event->inputs.at(hit.layer) < 0 !!!!!!!!!!!!!!!!!" << endl;
    cout << endl;
  }

  if (inputs[hit.layer] >= lustSize) {  //TODO should be the size of the LUT of the first layer
    cout << " event->inputs[hit.layer] >= " << lustSize << " !!!!!!!!!!!!!!!!!" << endl;
  }
  return true;
}

bool omtfHitWithQualToEventInput(OmtfHit& hit, std::vector<float>& inputs, unsigned int omtfRefLayer, bool print) {
  float rangeMiddle = 64 / 2;
  float offset = (omtfRefLayer << 7) + rangeMiddle;

  int rangeFactor = 2;  //rangeFactor scales the hit.phiDist such that the event->inputs is smaller then 63

  if ((hit.layer == 1 || hit.layer == 3 || hit.layer == 5)) {
    //if (!hit.valid)
    //  return false;

    if (hit.quality < 2)  ///TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      return false;

    if (hit.layer == 1) {
      rangeFactor = 8;
    } else if (hit.layer == 3) {
      rangeFactor = 8;
    } else if (hit.layer == 5) {
      rangeFactor = 8;
    }

    if (hit.quality >= 4) {
      offset += (1 << 6);
      //rangeFactor *= 4;
    }
  } else {
    //if (!hit.valid)
    //  return false;

    if ((hit.layer == 0 || hit.layer == 2 || hit.layer == 3)) {
      if (hit.quality >= 4) {
        offset += (1 << 6);
        //rangeFactor *= 4;
      }
    }

    /*else if(hit.layer == 8 || hit.layer == 17) {
        rangeFactor = 4;
        }*/
    if (hit.layer == 9) {
      rangeFactor = 1;
    }
    /*else {
        rangeFactor = 2;
        }*/

    /*if(hit.valid) {
            offset += (1 << 6);
            rangeFactor *= 4;
        }*/
  }

  rangeFactor *= 4;  //TODO !!!!!!!!!!!!!!!!!!!

  if (abs(hit.phiDist) >= ((rangeMiddle - 1) * rangeFactor)) {
    cout  //<<" muonPt "<<omtfEvent.muonPt<<" omtfPt "<<omtfEvent.omtfPt
        << " RefLayer " << omtfRefLayer << " layer " << int(hit.layer) << " hit.phiDist " << hit.phiDist << " valid "
        << ((short)hit.valid) << " quality " << ((short)hit.quality)
        << " hit.phiDist outside the range !!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    hit.phiDist = copysign((rangeMiddle - 1) * rangeFactor, hit.phiDist);
  }

  inputs.at(hit.layer) = (float)hit.phiDist / (float)rangeFactor + offset;

  if (inputs.at(hit.layer) >=
      1022)  //the last address i.e. 1023 is reserved for the no-hit value, so interpolation between the 1022 and 1023 has no sense
    inputs.at(hit.layer) = 1022;

  if (print || inputs.at(hit.layer) < 0) {
    cout  //<<"rawData "<<hex<<setw(16)<<hit.rawData
        << " layer " << dec << int(hit.layer);
    cout << " phiDist " << hit.phiDist << " inputVal " << inputs.at(hit.layer) << " hit.z " << int(hit.z) << " valid "
         << ((short)hit.valid) << " quality " << (short)hit.quality << " omtfRefLayer " << omtfRefLayer << " offset "
         << offset;
    if (inputs.at(hit.layer) < 0)
      cout << " event->inputs.at(hit.layer) < 0 !!!!!!!!!!!!!!!!!" << endl;
    cout << endl;
  }

  if (inputs[hit.layer] >= 1024) {  //TODO should be the size of the LUT of the first layer
    cout << " event->inputs[hit.layer] >= 1024 !!!!!!!!!!!!!!!!!" << endl;
  }
  return true;
}

bool omtfHitToEventInput(OmtfHit& hit, std::vector<float>& inputs, unsigned int omtfRefLayer, bool print) {
  float offset = (omtfRefLayer << 7) + 64;

  if (hit.valid) {
    if ((hit.layer == 1 || hit.layer == 3 || hit.layer == 5) && hit.quality < 4)  ///TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      return false;

    int rangeFactor = 2;  //rangeFactor scales the hit.phiDist such that the event->inputs is smaller then 63
    if (hit.layer == 1) {
      rangeFactor = 8;
    } else if (hit.layer == 3) {
      rangeFactor = 4;
    } else if (hit.layer == 5) {
      //rangeFactor = 4;
    } else if (hit.layer == 9) {
      rangeFactor = 1;
    }

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

void NNRegression::run(AlgoMuons::value_type& algoMuon,
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
  //unsigned int minHitsCnt = 3;
  const unsigned int maxHitCnt = 18;  //layer cnt

  const unsigned int inputCnt = maxHitCnt;
  const unsigned int outputCnt = 6;
  const float noHitVal = 1023.;

  //edm::LogImportant("OMTFReconstruction") <<"\n----------------------"<<endl;
  //edm::LogImportant("OMTFReconstruction") <<(*algoMuon)<<std::endl;

  std::vector<float> inputs(inputCnt, noHitVal);
  int hitCnt = 0;
  unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[algoMuon->getRefLayer()];
  for (unsigned int iLogicLayer = 0; iLogicLayer < gpResult.getStubResults().size(); ++iLogicLayer) {
    auto& stubResult = gpResult.getStubResults()[iLogicLayer];
    if (stubResult.getMuonStub()) {  //&& stubResult.getValid() //TODO!!!!!!!!!!!!!!!!1
      OmtfHit hit(0);
      hit.layer = iLogicLayer;
      hit.quality = stubResult.getMuonStub()->qualityHw;
      //hit.eta = stubResult.getMuonStub()->etaHw;  //in which scale?
      if (refLayerLogicNum == iLogicLayer)
        hit.deltaR = stubResult.getMuonStub()->r - 413;  //r of the ref hit - r of RB1in
      else
        hit.deltaR = stubResult.getMuonStub()->r - gpResult.getStubResults()[refLayerLogicNum].getMuonStub()->r;

      hit.valid = stubResult.getValid();

      //TODO the hit.phiDist should be set in the same way as in DataROOTDumper2, for the root files used for the NN training
      //so either hit.phiDist = hitPhi - phiRefHit; or hit.phiDist = stubResult.getDeltaPhi();
      /*
      int hitPhi = stubResult.getMuonStub()->phiHw;
      unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[algoMuon->getRefLayer()];
      int phiRefHit = gpResult.getStubResults()[refLayerLogicNum].getMuonStub()->phiHw;

      if (omtfConfig->isBendingLayer(iLogicLayer)) {
        hitPhi = stubResult.getMuonStub()->phiBHw;
        phiRefHit = 0;  //phi ref hit for the banding layer set to 0, since it should not be included in the phiDist
      }

      hit.phiDist = hitPhi - phiRefHit;*/

      hit.phiDist = stubResult.getDeltaPhi();

      /*
      LogTrace("l1tOmtfEventPrint") <<" muonPt "<<event.muonPt<<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer
          <<" layer "<<int(hit.layer)<<" PdfBin "<<stubResult.getPdfBin()<<" hit.phiDist "<<hit.phiDist<<" valid "<<stubResult.getValid()<<" " //<<" phiDist "<<phiDist
          <<" getDistPhiBitShift "<<omtfCand->getGoldenPatern()->getDistPhiBitShift(iLogicLayer, omtfCand->getRefLayer())
          <<" meanDistPhiValue   "<<omtfCand->getGoldenPatern()->meanDistPhiValue(iLogicLayer, omtfCand->getRefLayer())//<<(phiDist != hit.phiDist? "!!!!!!!<<<<<" : "")
          <<endl;*/

      /* if (hit.phiDist > 504 || hit.phiDist < -512) {
        edm::LogVerbatim("l1tOmtfEventPrint")
            //<<" muonPt "<<event.muonPt<<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer
            << " layer " << int(hit.layer) << " hit.phiDist " << hit.phiDist << " valid " << stubResult.getValid()
            << " !!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
      } */

      hitCnt += omtfHitWithQualAndRToEventInput(hit, inputs, algoMuon->getRefLayer(), false);
    }
  }

  //noHitCnt input is calculated in the LutNetworkFixedPointRegression2Outputs::run
  //so here hitCnt is used only for debug
  LogTrace("l1tOmtfEventPrint") << " " << __FUNCTION__ << ":" << __LINE__ << " hitCnt " << hitCnt << std::endl;

  std::vector<double> nnResult(outputCnt);
  lutNetworkFP->run(inputs, noHitVal, nnResult);

  LogTrace("l1tOmtfEventPrint") << " " << __FUNCTION__ << ":" << __LINE__ << std::endl;

  int charge = nnResult[2] >= 0 ? 1 : -1;

  LogTrace("l1tOmtfEventPrint") << " " << __FUNCTION__ << ":" << __LINE__ << " nnResult.at(0) " << nnResult.at(0)
                                << " nnResult.at(1) " << nnResult.at(1) << std::endl;

  //algoMuon->setPtNN(omtfConfig->ptGevToHw(nnResult.at(0)));
  //auto calibratedHwPt = lutNetworkFP->getCalibratedHwPt();
  //pt in the hardware scale, ptGeV = (ptHw -1) / 2

  //algoMuon->setPtNNConstr(omtfConfig->ptGevToHw(calibratedHwPt));

  //here the pts are GeV
  double omtfPt = omtfConfig->hwPtToGev(algoMuon->getPtConstr());
  double nnPt = nnResult.at(0);
  double combinedPt = nnPt;
  if (nnPt < 2.5 || (omtfPt < 0.75 * nnResult[0]))
    combinedPt = omtfPt;

  algoMuon->setPtNNConstr(combinedPt);

  algoMuon->setChargeNNConstr(charge);

  //TODOO uncomment when NN with upt is ready
  //int ptHwUnconstr = round(nnResult.at(??) );
  //algoMuon->setPtNNUnconstr(ptHwUnconstr);

  algoMuon->setNnOutputs(nnResult);

  if (omtfConfig->getDumpResultToXML()) {
    boost::property_tree::ptree procDataTree;
    for (unsigned int i = 0; i < inputs.size(); i++) {
      auto& inputTree = procDataTree.add("input", "");
      inputTree.add("<xmlattr>.num", i);
      inputTree.add("<xmlattr>.val", inputs[i]);
    }

    std::ostringstream ostr;

    for (unsigned int i = 0; i < nnResult.size(); i++) {
      auto& inputTree = procDataTree.add("output", "");
      ostr.str("");
      ostr << std::fixed << std::setprecision(19) << nnResult.at(i);

      inputTree.add("<xmlattr>.num", i);
      inputTree.add("<xmlattr>.val", ostr.str());
    }
    //procDataTree.add("calibratedHwPt.<xmlattr>.val", calibratedHwPt);

    procDataTree.add("hwSign.<xmlattr>.val", algoMuon->getChargeNNConstr() < 0 ? 1 : 0);

    for (auto& obs : observers)
      obs->addProcesorData("regressionNN", procDataTree);
  }

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
