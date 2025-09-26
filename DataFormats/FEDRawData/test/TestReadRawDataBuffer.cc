// -*- C++ -*-
//
// Package:    DataFormats/FEDRawData
// Class:      TestReadRawDataBuffer
//
/**\class edmtest::TestReadRawDataBuffer
  Description: Used as part of tests that ensure the RawDataBuffer
  data format can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time the RawDataBuffer persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  1 May 2023

#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/FEDRawData/interface/SLinkRocketHeaders.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <vector>

namespace edmtest {

  class TestReadRawDataBuffer : public edm::global::EDAnalyzer<> {
  public:
    TestReadRawDataBuffer(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // Two FEDRawData elements should be enough to verify we can read
    // and write the whole collection. I arbitrarily chose elements
    // 0 and 3 of the Collection. Values are meaningless, we just
    // verify what we read matches what we wrote. For purposes of
    // this test that is enough.
    std::vector<unsigned int> dataPattern1_;
    std::vector<unsigned int> dataPattern2_;
    edm::EDGetTokenT<RawDataBuffer> rawDataBufferToken_;
  };

  TestReadRawDataBuffer::TestReadRawDataBuffer(edm::ParameterSet const& iPSet)
      : dataPattern1_(iPSet.getParameter<std::vector<unsigned int>>("dataPattern1")),
        dataPattern2_(iPSet.getParameter<std::vector<unsigned int>>("dataPattern2")),
        rawDataBufferToken_(consumes(iPSet.getParameter<edm::InputTag>("rawDataBufferTag"))) {}

  void TestReadRawDataBuffer::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& rawDataBuffer = iEvent.get(rawDataBufferToken_);

    auto const& fragData0 = rawDataBuffer.fragmentData(0);
    auto const& fragData1 = rawDataBuffer.fragmentData(1);
    auto const& fragData2 = rawDataBuffer.fragmentData(30);
    auto const& fragDataHigh = rawDataBuffer.fragmentData(298457834);

    assert(fragData0.size());
    assert(fragData1.size());
    assert(fragData2.size());
    assert(fragDataHigh.size());

    auto hdrsize = sizeof(SLinkRocketHeader_v3);
    auto trsize = sizeof(SLinkRocketTrailer_v3);

    auto hdrView0 = makeSLinkRocketHeaderView(fragData0.dataHeader(hdrsize));
    auto trlView0 = makeSLinkRocketTrailerView(fragData0.dataTrailer(trsize), hdrView0->version());

    auto hdrView1 = makeSLinkRocketHeaderView(fragData1.dataHeader(hdrsize));
    auto trlView1 = makeSLinkRocketTrailerView(fragData1.dataTrailer(trsize), hdrView1->version());

    auto hdrView2 = makeSLinkRocketHeaderView(fragData2.dataHeader(hdrsize));
    auto trlView2 = makeSLinkRocketTrailerView(fragData2.dataTrailer(trsize), hdrView2->version());

    auto hdrViewHigh = makeSLinkRocketHeaderView(fragDataHigh.dataHeader(hdrsize));
    auto trlViewHigh = makeSLinkRocketTrailerView(fragDataHigh.dataTrailer(trsize), hdrViewHigh->version());

    auto src0data = fragData0.payload(hdrsize, trsize);
    for (size_t i = 0; i < src0data.size(); i++) {
      if (src0data[i] != dataPattern1_[i % std::size(dataPattern1_)])
        throwWithMessage("data id 0 does not have expected contents");
    }
    auto src1data = fragData1.payload(hdrsize, trsize);
    for (size_t i = 0; i < src1data.size(); i++) {
      if (src1data[i] != dataPattern2_[i % std::size(dataPattern2_)])
        throwWithMessage("data id 3 does not have expected contents");
    }
    auto src2data = fragData2.payload(hdrsize, trsize);
    for (size_t i = 0; i < src2data.size(); i++) {
      if (src2data[i] != dataPattern1_[i % std::size(dataPattern1_)])
        throwWithMessage("data id 30 does not have expected contents");
    }
    auto srcHighdata = fragDataHigh.payload(hdrsize, trsize);
    for (size_t i = 0; i < srcHighdata.size(); i++) {
      if (srcHighdata[i] != (dataPattern2_[i % std::size(dataPattern2_)]))
        throwWithMessage("data id (high) does not have expected contents");
    }

    //todo other two
  }

  void TestReadRawDataBuffer::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadRawDataBuffer::analyze, " << msg;
  }

  void TestReadRawDataBuffer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("dataPattern1");
    desc.add<std::vector<unsigned int>>("dataPattern2");
    desc.add<edm::InputTag>("rawDataBufferTag");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadRawDataBuffer;
DEFINE_FWK_MODULE(TestReadRawDataBuffer);
