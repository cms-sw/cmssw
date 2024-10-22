// -*- C++ -*-
//
// Package:    TestBtagPayloads
// Class:      TestBtagPayloads
//
/**\class TestBtagPayloads.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michael Segala
//         Created:  Wed Jun 20 02:47:47 CDT 2012
//
//

// system include files
#include <iostream>
#include <memory>
#include <map>
#include <stdio.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoBTag/Records/interface/BTagPerformanceRecord.h"
#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"
#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"

class TestBtagPayloads : public edm::one::EDAnalyzer<> {
public:
  explicit TestBtagPayloads(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) final;

  const edm::ESGetToken<BtagPerformance, BTagPerformanceRecord> ttToken_;
  const edm::ESGetToken<BtagPerformance, BTagPerformanceRecord> muToken_;
  const edm::ESGetToken<BtagPerformance, BTagPerformanceRecord> mistagToken_;

  std::map<std::string, PerformanceResult::ResultType> measureMap_;

  const std::vector<std::string> measureName_;
  const std::vector<std::string> measureType_;

  std::vector<edm::ESGetToken<BtagPerformance, BTagPerformanceRecord>> measureTokens_;
};

namespace {
  //Possible algorithms: TTBARDISCRIMBTAGCSV, TTBARDISCRIMBTAGJP, TTBARDISCRIMBTAGTCHP
  constexpr char const* const ttName = "TTBARDISCRIMBTAGCSV";

  //Possible algorithms: MUJETSWPBTAGCSVL,  MUJETSWPBTAGCSVM,   MUJETSWPBTAGCSVT
  //                     MUJETSWPBTAGJPL,   MUJETSWPBTAGJPM,    MUJETSWPBTAGJPT
  //                                                            MUJETSWPBTAGTCHPT
  constexpr char const* const muName = "MUJETSWPBTAGCSVL";

  //Possible algorithms: MISTAGCSVLAB,  MISTAGCSVMAB,   MISTAGCSVTAB
  //                     MISTAGJPLAB,   MISTAGJPMAB,    MISTAGJPTAB                // Data period 2012 AB
  //                                                    MISTAGTCHPTAB
  //
  //                     MISTAGCSVLABCD,  MISTAGCSVMABCD,   MISTAGCSVTABCD
  //                     MISTAGJPLABCD,   MISTAGJPMABCD,    MISTAGJPTABCD          // Data period 2012 ABCD
  //                                                        MISTAGTCHPTABCD
  //
  //                     MISTAGCSVLC,  MISTAGCSVMC,   MISTAGCSVTC
  //                     MISTAGJPLC,   MISTAGJPMC,    MISTAGJPTC                   // Data period 2012 C
  //                                                  MISTAGTCHPTC
  //
  //                     MISTAGCSVLD,  MISTAGCSVMD,   MISTAGCSVTD
  //                     MISTAGJPLD,   MISTAGJPMD,    MISTAGJPTD                   // Data period 2012 D
  //                                                  MISTAGTCHPTD
  constexpr char const* const mistagName = "MISTAGTCHPTD";
}  // namespace
TestBtagPayloads::TestBtagPayloads(const edm::ParameterSet& iConfig)
    : ttToken_(esConsumes(edm::ESInputTag("", ttName))),
      muToken_(esConsumes(edm::ESInputTag("", muName))),
      mistagToken_(esConsumes(edm::ESInputTag("", mistagName))),
      //Possible algorithms: TTBARWPBTAGCSVL,  TTBARWPBTAGCSVM,   TTBARWPBTAGCSVT
      //                     TTBARWPBTAGJPL,   TTBARWPBTAGJPM,    TTBARWPBTAGJPT
      //                                                          TTBARWPBTAGTCHPT
      measureName_({"TTBARWPBTAGCSVL",
                    "TTBARWPBTAGCSVL",
                    "TTBARWPBTAGCSVL",
                    "TTBARWPBTAGCSVL",
                    "TTBARWPBTAGJPT",
                    "TTBARWPBTAGJPT",
                    "TTBARWPBTAGJPT",
                    "TTBARWPBTAGJPT"}),
      measureType_({"BTAGBEFFCORR",
                    "BTAGBERRCORR",
                    "BTAGCEFFCORR",
                    "BTAGCERRCORR",
                    "BTAGBEFFCORR",
                    "BTAGBERRCORR",
                    "BTAGCEFFCORR",
                    "BTAGCERRCORR"}) {
  if (measureName_.size() != measureType_.size()) {
    std::cout << "measureName_, measureType_ size mismatch!" << std::endl;
    exit(-1);
  }

  measureMap_["BTAGBEFFCORR"] = PerformanceResult::BTAGBEFFCORR;
  measureMap_["BTAGBERRCORR"] = PerformanceResult::BTAGBERRCORR;
  measureMap_["BTAGCEFFCORR"] = PerformanceResult::BTAGCEFFCORR;
  measureMap_["BTAGCERRCORR"] = PerformanceResult::BTAGCERRCORR;

  measureTokens_.reserve(measureName_.size());

  //only call esConsumes if the name changes, else reuse the token
  std::map<std::string, edm::ESGetToken<BtagPerformance, BTagPerformanceRecord>> tokens;
  for (auto const& n : measureName_) {
    auto insert = tokens.insert({n, {}});
    if (insert.second) {
      insert.first->second = esConsumes(edm::ESInputTag("", n));
    }
    measureTokens_.push_back(insert.first->second);
  }
}

//
// member functions
//

// ------------ method called for each event  ------------
void TestBtagPayloads::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  edm::ESHandle<BtagPerformance> perfH;
  BinningPointByMap p;

  //
  //++++++++++++------  TESTING FOR TTBAR SF's and efficiencies using CONTINIOUS DISCRIMINATORS      --------+++++++++++++
  //

  printf("\033[22;31m \n TESTING FOR TTBAR SF's and efficiencies using CONTINIOUS DISCRIMINATORS \n\033[0m");

  std::cout << " Studying performance with label " << ttName << std::endl;
  const BtagPerformance& perf = iSetup.getData(ttToken_);

  std::cout << " My Performance Object is indeed a " << typeid(&perf).name() << std::endl;

  std::cout << " The WP is defined by a cut at " << perf.workingPoint().cut() << std::endl;
  std::cout << " Discriminant is " << perf.workingPoint().discriminantName() << std::endl;
  std::cout << " Is cut based WP " << perf.workingPoint().cutBased() << std::endl;

  p.insert(BinningVariables::JetEta, 0.6);
  p.insert(BinningVariables::Discriminator, 0.23);

  std::cout << " test eta=0.6, discrim = 0.23" << std::endl;
  std::cout << " bSF/bFSerr ?" << perf.isResultOk(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf.isResultOk(PerformanceResult::BTAGBERRCORR, p) << std::endl;
  std::cout << " bSF/bSFerr =" << perf.getResult(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf.getResult(PerformanceResult::BTAGBERRCORR, p) << std::endl;

  std::cout << std::endl;
  std::cout << std::endl;

  //
  //++++++++++++------  TESTING FOR TTBAR WP's   --------+++++++++++++
  //

  printf("\033[22;31m TESTING FOR TTBAR WP's \n\033[0m");

  for (size_t iMeasure = 0; iMeasure < measureName_.size(); iMeasure++) {
    std::cout << "Testing: " << measureName_[iMeasure] << " of type " << measureType_[iMeasure] << std::endl;

    //Setup our measurement
    const BtagPerformance& perf2 = iSetup.getData(measureTokens_[iMeasure]);

    //Working point
    std::cout << "Working point: " << perf2.workingPoint().cut() << std::endl;
    //Setup the point we wish to test!
    BinningPointByMap measurePoint;
    measurePoint.insert(BinningVariables::JetEt, 50);
    measurePoint.insert(BinningVariables::JetEta, 0.6);

    std::cout << "Is it OK? " << perf2.isResultOk(measureMap_[measureType_[iMeasure]], measurePoint)
              << " result at 50 GeV, 0,6 |eta| = " << perf2.getResult(measureMap_[measureType_[iMeasure]], measurePoint)
              << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;

  //
  //++++++++++++------  TESTING FOR Mu+Jets WP's   --------+++++++++++++
  //

  printf("\033[22;31m TESTING FOR Mu+Jets WP's \n\033[0m");

  std::cout << " Studying performance with label " << muName << std::endl;
  const BtagPerformance& perf3 = iSetup.getData(muToken_);

  std::cout << " My Performance Object is indeed a " << typeid(&perf3).name() << std::endl;

  std::cout << " The WP is defined by a cut at " << perf3.workingPoint().cut() << std::endl;
  std::cout << " Discriminant is " << perf3.workingPoint().discriminantName() << std::endl;
  std::cout << " Is cut based WP " << perf3.workingPoint().cutBased() << std::endl;

  p.insert(BinningVariables::JetEta, 0.6);
  p.insert(BinningVariables::JetEt, 50);

  std::cout << " test eta=0.6, et = 50" << std::endl;
  std::cout << " bSF/bFSerr ?" << perf3.isResultOk(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf3.isResultOk(PerformanceResult::BTAGBERRCORR, p) << std::endl;
  std::cout << " bSF/bSFerr =" << perf3.getResult(PerformanceResult::BTAGBEFFCORR, p) << "/"
            << perf3.getResult(PerformanceResult::BTAGBERRCORR, p) << std::endl;

  std::cout << std::endl;
  std::cout << std::endl;

  //
  //++++++++++++------  TESTING FOR Mu+Jets Mistags   --------+++++++++++++
  //

  printf("\033[22;31m TESTING FOR Mu+Jets Mistags \n\033[0m");

  std::cout << " Studying performance with label " << mistagName << std::endl;
  const BtagPerformance& perf4 = iSetup.getData(mistagToken_);

  std::cout << " My Performance Object is indeed a " << typeid(&perf4).name() << std::endl;

  std::cout << " The WP is defined by a cut at " << perf4.workingPoint().cut() << std::endl;
  std::cout << " Discriminant is " << perf4.workingPoint().discriminantName() << std::endl;
  std::cout << " Is cut based WP " << perf4.workingPoint().cutBased() << std::endl;

  p.insert(BinningVariables::JetEta, 0.6);
  p.insert(BinningVariables::JetEt, 50);

  std::cout << " test eta=0.6, et = 50" << std::endl;
  std::cout << " leff ?" << perf4.isResultOk(PerformanceResult::BTAGLEFF, p) << std::endl;
  std::cout << " leff =" << perf4.getResult(PerformanceResult::BTAGLEFF, p) << std::endl;
  std::cout << " bSF/bFSerr ?" << perf4.isResultOk(PerformanceResult::BTAGLEFFCORR, p) << "/"
            << perf4.isResultOk(PerformanceResult::BTAGLERRCORR, p) << std::endl;
  std::cout << " bSF/bSFerr =" << perf4.getResult(PerformanceResult::BTAGLEFFCORR, p) << "/"
            << perf4.getResult(PerformanceResult::BTAGLERRCORR, p) << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestBtagPayloads);
