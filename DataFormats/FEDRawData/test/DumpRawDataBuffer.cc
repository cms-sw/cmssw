// -*- C++ -*-
//
// Package:    DataFormats/FEDRawData
// Class:      DumpRawDataBuffer
//
/**\class edmtest::DumpRawDataBuffer
  Description: Prints the RawDataBuffer EDProduct 
                (corresponding to the DAQ RAW data),
               either in its entirety, or for a range of S-Link sources.
*/

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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <cmath>
#include <iostream>

namespace edmtest {

  class DumpRawDataBuffer : public edm::global::EDAnalyzer<> {
  public:
    DumpRawDataBuffer(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // Two FEDRawData elements should be enough to verify we can read
    // and write the whole collection. I arbitrarily chose elements
    // 0 and 3 of the Collection. Values are meaningless, we just
    // verify what we read matches what we wrote. For purposes of
    // this test that is enough.
    
    unsigned int minSLinkID_;
    unsigned int maxSLinkID_;
    edm::EDGetTokenT<RawDataBuffer> rawDataBufferToken_;
  };

  DumpRawDataBuffer::DumpRawDataBuffer(edm::ParameterSet const& iPSet)
      : minSLinkID_(iPSet.getParameter<unsigned int>("minSLinkID")),
        maxSLinkID_(iPSet.getParameter<unsigned int>("maxSLinkID")),
        rawDataBufferToken_(consumes(iPSet.getParameter<edm::InputTag>("rawDataBufferTag"))) {}

void DumpRawDataBuffer::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
  auto const& rawDataBuffer = iEvent.get(rawDataBufferToken_);

  edm::LogSystem out("DumpRawDataBuffer");

    auto hdrsize = sizeof(SLinkRocketHeader_v3);
    auto trsize = sizeof(SLinkRocketTrailer_v3);
    out<<"\n      S-Link CMS Header & Trailer size (bytes) = "<<hdrsize<<" "<<trsize<<"\n";
  
  for (unsigned int idSource = minSLinkID_; idSource < maxSLinkID_; idSource++) {
    
    auto const& fragData = rawDataBuffer.fragmentData(idSource);
    if (fragData.size() == 0) continue;

    out<<std::dec<<"\n\n";
    out<<"=== Found SOURCE ID = "<<idSource<<" with size = "<<fragData.size()<<"\n";

    //auto srcdata = fragData.payload(hdrsize, trsize);
    auto srcdata = fragData.payload(0, 0); // Dump entire payload including header & trailer.
    const int bytesPerWord = 16;
    const int nWords = std::ceil(float(srcdata.size())/float(bytesPerWord));
    
    // Print 128b S-Link word as 16 bytes, with LSB at right of each word.
    for (int word = 0; word < nWords;word++) {
      out<<"\n"<<" Word "<<std::dec<<std::setw(6)<<word<<" : ";
      for (int j = bytesPerWord - 1; j >=0; j--) {
        unsigned int addr = static_cast<unsigned int>(word * bytesPerWord + j);
        if (addr < srcdata.size()) {
          unsigned int byte = static_cast<unsigned int>(srcdata[addr]);
          out<<std::hex<<std::setw(2)<<byte<<"  ";
        } else {
          out<<"    ";
        }
      }
    }
  }
  out<<"\n";
}

  void DumpRawDataBuffer::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "DumpRawDataBuffer::analyze, " << msg;
  }

  void DumpRawDataBuffer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<unsigned int>("minSLinkID", 0);
    desc.add<unsigned int>("maxSLinkID", 99999);
    desc.add<edm::InputTag>("rawDataBufferTag");    
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::DumpRawDataBuffer;
DEFINE_FWK_MODULE(DumpRawDataBuffer);
