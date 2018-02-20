#include "DQM/L1TMonitor/interface/L1TStage2uGMT.h"

L1TStage2uGMT::L1TStage2uGMT(const edm::ParameterSet& ps)
    : ugmtMuonToken(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonProducer"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir")),
      emul(ps.getUntrackedParameter<bool>("emulator")),
      verbose(ps.getUntrackedParameter<bool>("verbose")),
      etaScale_(0.010875), // eta scale (CMS DN-2015/017)
      phiScale_(0.010908)  // phi scale (2*pi/576 HW values)
{
  if (!emul) {
    ugmtBMTFToken = consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("bmtfProducer"));
    ugmtOMTFToken = consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("omtfProducer"));
    ugmtEMTFToken = consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emtfProducer"));
  }
}

L1TStage2uGMT::~L1TStage2uGMT() {}

void L1TStage2uGMT::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonProducer")->setComment("uGMT output muons.");;
  desc.add<edm::InputTag>("bmtfProducer")->setComment("RegionalMuonCands from BMTF.");
  desc.add<edm::InputTag>("omtfProducer")->setComment("RegionalMuonCands from OMTF.");
  desc.add<edm::InputTag>("emtfProducer")->setComment("RegionalMuonCands from EMTF.");
  desc.addUntracked<std::string>("monitorDir", "")->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<bool>("emulator", false)->setComment("Create histograms for muonProducer input only. xmtfProducer inputs are ignored.");
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("l1tStage2uGMT", desc);
}

void L1TStage2uGMT::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c, ugmtdqm::Histograms& histograms) const {}

void L1TStage2uGMT::bookHistograms(DQMStore::ConcurrentBooker& booker, const edm::Run&, const edm::EventSetup&, ugmtdqm::Histograms& histograms) const {

  if (!emul) {
    // BMTF Input
    booker.setCurrentFolder(monitorDir + "/BMTFInput");

    histograms.ugmtBMTFBX = booker.book1D("ugmtBMTFBX", "uGMT BMTF Input BX", 7, -3.5, 3.5);
    histograms.ugmtBMTFBX.setAxisTitle("BX", 1);

    histograms.ugmtBMTFnMuons = booker.book1D("ugmtBMTFnMuons", "uGMT BMTF Input Muon Multiplicity", 37, -0.5, 36.5);
    histograms.ugmtBMTFnMuons.setAxisTitle("Muon Multiplicity (BX == 0)", 1);

    histograms.ugmtBMTFhwPt = booker.book1D("ugmtBMTFhwPt", "uGMT BMTF Input p_{T}", 512, -0.5, 511.5);
    histograms.ugmtBMTFhwPt.setAxisTitle("Hardware p_{T}", 1);

    histograms.ugmtBMTFhwEta = booker.book1D("ugmtBMTFhwEta", "uGMT BMTF Input #eta", 201, -100.5, 100.5);
    histograms.ugmtBMTFhwEta.setAxisTitle("Hardware #eta", 1);
    
    histograms.ugmtBMTFhwPhi = booker.book1D("ugmtBMTFhwPhi", "uGMT BMTF Input #phi", 71, -10.5, 60.5);
    histograms.ugmtBMTFhwPhi.setAxisTitle("Hardware #phi", 1);

    histograms.ugmtBMTFglbPhi = booker.book1D("ugmtBMTFglbhwPhi", "uGMT BMTF Input #phi", 576, -0.5, 575.5);
    histograms.ugmtBMTFglbPhi.setAxisTitle("Global Hardware #phi", 1);

    histograms.ugmtBMTFProcvshwPhi = booker.book2D("ugmtBMTFProcvshwPhi", "uGMT BMTF Processor vs #phi", 71, -10.5, 60.5, 12, 0, 12);
    histograms.ugmtBMTFProcvshwPhi.setAxisTitle("Hardware #phi", 1);
    histograms.ugmtBMTFProcvshwPhi.setAxisTitle("Wedge", 2);
    for (int bin = 1; bin <= 12; ++bin) {
      histograms.ugmtBMTFProcvshwPhi.setBinLabel(bin, std::to_string(bin), 2);
    }

    histograms.ugmtBMTFhwSign = booker.book1D("ugmtBMTFhwSign", "uGMT BMTF Input Sign", 2, -0.5, 1.5);
    histograms.ugmtBMTFhwSign.setAxisTitle("Hardware Sign", 1);

    histograms.ugmtBMTFhwSignValid = booker.book1D("ugmtBMTFhwSignValid", "uGMT BMTF Input SignValid", 2, -0.5, 1.5);
    histograms.ugmtBMTFhwSignValid.setAxisTitle("SignValid", 1);

    histograms.ugmtBMTFhwQual = booker.book1D("ugmtBMTFhwQual", "uGMT BMTF Input Quality", 16, -0.5, 15.5);
    histograms.ugmtBMTFhwQual.setAxisTitle("Quality", 1);

    histograms.ugmtBMTFlink = booker.book1D("ugmtBMTFlink", "uGMT BMTF Input Link", 12, 47.5, 59.5);
    histograms.ugmtBMTFlink.setAxisTitle("Link", 1);

    histograms.ugmtBMTFMuMuDEta = booker.book1D("ugmtBMTFMuMuDEta", "uGMT BMTF input muons #Delta#eta between wedges", 100, -0.5, 0.5);
    histograms.ugmtBMTFMuMuDEta.setAxisTitle("#Delta#eta", 1);

    histograms.ugmtBMTFMuMuDPhi = booker.book1D("ugmtBMTFMuMuDPhi", "uGMT BMTF input muons #Delta#phi between wedges", 100, -0.5, 0.5);
    histograms.ugmtBMTFMuMuDPhi.setAxisTitle("#Delta#phi", 1);

    histograms.ugmtBMTFMuMuDR = booker.book1D("ugmtBMTFMuMuDR", "uGMT BMTF input muons #DeltaR between wedges", 50, 0., 0.5);
    histograms.ugmtBMTFMuMuDR.setAxisTitle("#DeltaR", 1);

    // OMTF Input
    booker.setCurrentFolder(monitorDir + "/OMTFInput");

    histograms.ugmtOMTFBX = booker.book1D("ugmtOMTFBX", "uGMT OMTF Input BX", 7, -3.5, 3.5);
    histograms.ugmtOMTFBX.setAxisTitle("BX", 1);

    histograms.ugmtOMTFnMuons = booker.book1D("ugmtOMTFnMuons", "uGMT OMTF Input Muon Multiplicity", 37, -0.5, 36.5);
    histograms.ugmtOMTFnMuons.setAxisTitle("Muon Multiplicity (BX == 0)", 1);

    histograms.ugmtOMTFhwPt = booker.book1D("ugmtOMTFhwPt", "uGMT OMTF Input p_{T}", 512, -0.5, 511.5);
    histograms.ugmtOMTFhwPt.setAxisTitle("Hardware p_{T}", 1);

    histograms.ugmtOMTFhwEta = booker.book1D("ugmtOMTFhwEta", "uGMT OMTF Input #eta", 231, -115.5, 115.5);
    histograms.ugmtOMTFhwEta.setAxisTitle("Hardware #eta", 1);
    
    histograms.ugmtOMTFhwPhiPos = booker.book1D("ugmtOMTFhwPhiPos", "uGMT OMTF Input #phi, Positive Side", 122, -16.5, 105.5);
    histograms.ugmtOMTFhwPhiPos.setAxisTitle("Hardware #phi", 1);

    histograms.ugmtOMTFhwPhiNeg = booker.book1D("ugmtOMTFhwPhiNeg", "uGMT OMTF Input #phi, Negative Side", 122, -16.5, 105.5);
    histograms.ugmtOMTFhwPhiNeg.setAxisTitle("Hardware #phi", 1);

    histograms.ugmtOMTFglbPhiPos = booker.book1D("ugmtOMTFglbhwPhiPos", "uGMT OMTF Input #phi, Positive Side", 576, -0.5, 575.5);
    histograms.ugmtOMTFglbPhiPos.setAxisTitle("Global Hardware #phi", 1);

    histograms.ugmtOMTFglbPhiNeg = booker.book1D("ugmtOMTFglbhwPhiNeg", "uGMT OMTF Input #phi, Negative Side", 576, -0.5, 575.5);
    histograms.ugmtOMTFglbPhiNeg.setAxisTitle("Global Hardware #phi", 1);

    histograms.ugmtOMTFProcvshwPhiPos = booker.book2D("ugmtOMTFProcvshwPhiPos", "uGMT OMTF Processor vs #phi", 122, -16.5, 105.5, 6, 0, 6);
    histograms.ugmtOMTFProcvshwPhiPos.setAxisTitle("Hardware #phi", 1);
    histograms.ugmtOMTFProcvshwPhiPos.setAxisTitle("Sector (Positive Side)", 2);

    histograms.ugmtOMTFProcvshwPhiNeg = booker.book2D("ugmtOMTFProcvshwPhiNeg", "uGMT OMTF Processor vs #phi", 122, -16.5, 105.5, 6, 0, 6);
    histograms.ugmtOMTFProcvshwPhiNeg.setAxisTitle("Hardware #phi", 1);
    histograms.ugmtOMTFProcvshwPhiNeg.setAxisTitle("Sector (Negative Side)", 2);

    for (int bin = 1; bin <= 6; ++bin) {
      histograms.ugmtOMTFProcvshwPhiPos.setBinLabel(bin, std::to_string(bin), 2);
      histograms.ugmtOMTFProcvshwPhiNeg.setBinLabel(bin, std::to_string(bin), 2);
    }

    histograms.ugmtOMTFhwSign = booker.book1D("ugmtOMTFhwSign", "uGMT OMTF Input Sign", 2, -0.5, 1.5);
    histograms.ugmtOMTFhwSign.setAxisTitle("Hardware Sign", 1);

    histograms.ugmtOMTFhwSignValid = booker.book1D("ugmtOMTFhwSignValid", "uGMT OMTF Input SignValid", 2, -0.5, 1.5);
    histograms.ugmtOMTFhwSignValid.setAxisTitle("SignValid", 1);

    histograms.ugmtOMTFhwQual = booker.book1D("ugmtOMTFhwQual", "uGMT OMTF Input Quality", 16, -0.5, 15.5);
    histograms.ugmtOMTFhwQual.setAxisTitle("Quality", 1);

    histograms.ugmtOMTFlink = booker.book1D("ugmtOMTFlink", "uGMT OMTF Input Link", 24, 41.5, 65.5);
    histograms.ugmtOMTFlink.setAxisTitle("Link", 1);

    histograms.ugmtOMTFMuMuDEta = booker.book1D("ugmtOMTFMuMuDEta", "uGMT OMTF input muons #Delta#eta between sectors", 100, -0.5, 0.5);
    histograms.ugmtOMTFMuMuDEta.setAxisTitle("#Delta#eta", 1);

    histograms.ugmtOMTFMuMuDPhi = booker.book1D("ugmtOMTFMuMuDPhi", "uGMT OMTF input muons #Delta#phi between sectors", 100, -0.5, 0.5);
    histograms.ugmtOMTFMuMuDPhi.setAxisTitle("#Delta#phi", 1);

    histograms.ugmtOMTFMuMuDR = booker.book1D("ugmtOMTFMuMuDR", "uGMT OMTF input muons #DeltaR between sectors", 50, 0., 0.5);
    histograms.ugmtOMTFMuMuDR.setAxisTitle("#DeltaR", 1);

    // EMTF Input
    booker.setCurrentFolder(monitorDir + "/EMTFInput");

    histograms.ugmtEMTFBX = booker.book1D("ugmtEMTFBX", "uGMT EMTF Input BX", 7, -3.5, 3.5);
    histograms.ugmtEMTFBX.setAxisTitle("BX", 1);

    histograms.ugmtEMTFnMuons = booker.book1D("ugmtEMTFnMuons", "uGMT EMTF Input Muon Multiplicity", 37, -0.5, 36.5);
    histograms.ugmtEMTFnMuons.setAxisTitle("Muon Multiplicity (BX == 0)", 1);

    histograms.ugmtEMTFhwPt = booker.book1D("ugmtEMTFhwPt", "uGMT EMTF p_{T}", 512, -0.5, 511.5);
    histograms.ugmtEMTFhwPt.setAxisTitle("Hardware p_{T}", 1);

    histograms.ugmtEMTFhwEta = booker.book1D("ugmtEMTFhwEta", "uGMT EMTF #eta", 461, -230.5, 230.5);
    histograms.ugmtEMTFhwEta.setAxisTitle("Hardware #eta", 1);
    
    histograms.ugmtEMTFhwPhiPos = booker.book1D("ugmtEMTFhwPhiPos", "uGMT EMTF #phi, Positive Side", 146, -40.5, 105.5);
    histograms.ugmtEMTFhwPhiPos.setAxisTitle("Hardware #phi", 1);

    histograms.ugmtEMTFhwPhiNeg = booker.book1D("ugmtEMTFhwPhiNeg", "uGMT EMTF #phi, Negative Side", 146, -40.5, 105.5);
    histograms.ugmtEMTFhwPhiNeg.setAxisTitle("Hardware #phi", 1);

    histograms.ugmtEMTFglbPhiPos = booker.book1D("ugmtEMTFglbhwPhiPos", "uGMT EMTF Input Global #phi, Positive Side", 576, -0.5, 575.5);
    histograms.ugmtEMTFglbPhiPos.setAxisTitle("Global Hardware #phi", 1);

    histograms.ugmtEMTFglbPhiNeg = booker.book1D("ugmtEMTFglbhwPhiNeg", "uGMT EMTF Input Global #phi, Negative Side", 576, -0.5, 575.5);
    histograms.ugmtEMTFglbPhiNeg.setAxisTitle("Global Hardware #phi", 1);

    histograms.ugmtEMTFProcvshwPhiPos = booker.book2D("ugmtEMTFProcvshwPhiPos", "uGMT EMTF Processor vs #phi", 146, -40.5, 105.5, 6, 0, 6);
    histograms.ugmtEMTFProcvshwPhiPos.setAxisTitle("Hardware #phi", 1);
    histograms.ugmtEMTFProcvshwPhiPos.setAxisTitle("Sector (Positive Side)", 2);

    histograms.ugmtEMTFProcvshwPhiNeg = booker.book2D("ugmtEMTFProcvshwPhiNeg", "uGMT EMTF Processor vs #phi", 146, -40.5, 105.5, 6, 0, 6);
    histograms.ugmtEMTFProcvshwPhiNeg.setAxisTitle("Hardware #phi", 1);
    histograms.ugmtEMTFProcvshwPhiNeg.setAxisTitle("Sector (Negative Side)", 2);

    for (int bin = 1; bin <= 6; ++bin) {
      histograms.ugmtEMTFProcvshwPhiPos.setBinLabel(bin, std::to_string(bin), 2);
      histograms.ugmtEMTFProcvshwPhiNeg.setBinLabel(bin, std::to_string(bin), 2);
    }

    histograms.ugmtEMTFhwSign = booker.book1D("ugmtEMTFhwSign", "uGMT EMTF Sign", 2, -0.5, 1.5);
    histograms.ugmtEMTFhwSign.setAxisTitle("Hardware Sign", 1);

    histograms.ugmtEMTFhwSignValid = booker.book1D("ugmtEMTFhwSignValid", "uGMT EMTF SignValid", 2, -0.5, 1.5);
    histograms.ugmtEMTFhwSignValid.setAxisTitle("SignValid", 1);

    histograms.ugmtEMTFhwQual = booker.book1D("ugmtEMTFhwQual", "uGMT EMTF Quality", 16, -0.5, 15.5);
    histograms.ugmtEMTFhwQual.setAxisTitle("Quality", 1);

    histograms.ugmtEMTFlink = booker.book1D("ugmtEMTFlink", "uGMT EMTF Link", 36, 35.5, 71.5);
    histograms.ugmtEMTFlink.setAxisTitle("Link", 1);

    histograms.ugmtEMTFMuMuDEta = booker.book1D("ugmtEMTFMuMuDEta", "uGMT EMTF input muons #Delta#eta between sectors", 100, -0.5, 0.5);
    histograms.ugmtEMTFMuMuDEta.setAxisTitle("#Delta#eta", 1);

    histograms.ugmtEMTFMuMuDPhi = booker.book1D("ugmtEMTFMuMuDPhi", "uGMT EMTF input muons #Delta#phi between sectors", 100, -0.5, 0.5);
    histograms.ugmtEMTFMuMuDPhi.setAxisTitle("#Delta#phi", 1);

    histograms.ugmtEMTFMuMuDR = booker.book1D("ugmtEMTFMuMuDR", "uGMT EMTF input muons #DeltaR between sectors", 50, 0., 0.5);
    histograms.ugmtEMTFMuMuDR.setAxisTitle("#DeltaR", 1);

    // inter-TF muon correlations
    booker.setCurrentFolder(monitorDir + "/muon_correlations");

    histograms.ugmtBOMTFposMuMuDEta = booker.book1D("ugmtBOMTFposMuMuDEta", "uGMT input muons #Delta#eta between BMTF and OMTF+", 100, -0.5, 0.5);
    histograms.ugmtBOMTFposMuMuDEta.setAxisTitle("#Delta#eta", 1);

    histograms.ugmtBOMTFposMuMuDPhi = booker.book1D("ugmtBOMTFposMuMuDPhi", "uGMT input muons #Delta#phi between BMTF and OMTF+", 100, -0.5, 0.5);
    histograms.ugmtBOMTFposMuMuDPhi.setAxisTitle("#Delta#phi", 1);

    histograms.ugmtBOMTFposMuMuDR = booker.book1D("ugmtBOMTFposMuMuDR", "uGMT input muons #DeltaR between BMTF and OMTF+", 50, 0., 0.5);
    histograms.ugmtBOMTFposMuMuDR.setAxisTitle("#DeltaR", 1);

    histograms.ugmtBOMTFnegMuMuDEta = booker.book1D("ugmtBOMTFnegMuMuDEta", "uGMT input muons #Delta#eta between BMTF and OMTF-", 100, -0.5, 0.5);
    histograms.ugmtBOMTFnegMuMuDEta.setAxisTitle("#Delta#eta", 1);

    histograms.ugmtBOMTFnegMuMuDPhi = booker.book1D("ugmtBOMTFnegMuMuDPhi", "uGMT input muons #Delta#phi between BMTF and OMTF-", 100, -0.5, 0.5);
    histograms.ugmtBOMTFnegMuMuDPhi.setAxisTitle("#Delta#phi", 1);

    histograms.ugmtBOMTFnegMuMuDR = booker.book1D("ugmtBOMTFnegMuMuDR", "uGMT input muons #DeltaR between BMTF and OMTF-", 50, 0., 0.5);
    histograms.ugmtBOMTFnegMuMuDR.setAxisTitle("#DeltaR", 1);

    histograms.ugmtEOMTFposMuMuDEta = booker.book1D("ugmtEOMTFposMuMuDEta", "uGMT input muons #Delta#eta between EMTF+ and OMTF+", 100, -0.5, 0.5);
    histograms.ugmtEOMTFposMuMuDEta.setAxisTitle("#Delta#eta", 1);

    histograms.ugmtEOMTFposMuMuDPhi = booker.book1D("ugmtEOMTFposMuMuDPhi", "uGMT input muons #Delta#phi between EMTF+ and OMTF+", 100, -0.5, 0.5);
    histograms.ugmtEOMTFposMuMuDPhi.setAxisTitle("#Delta#phi", 1);

    histograms.ugmtEOMTFposMuMuDR = booker.book1D("ugmtEOMTFposMuMuDR", "uGMT input muons #DeltaR between EMTF+ and OMTF+", 50, 0., 0.5);
    histograms.ugmtEOMTFposMuMuDR.setAxisTitle("#DeltaR", 1);

    histograms.ugmtEOMTFnegMuMuDEta = booker.book1D("ugmtEOMTFnegMuMuDEta", "uGMT input muons #Delta#eta between EMTF- and OMTF-", 100, -0.5, 0.5);
    histograms.ugmtEOMTFnegMuMuDEta.setAxisTitle("#Delta#eta", 1);

    histograms.ugmtEOMTFnegMuMuDPhi = booker.book1D("ugmtEOMTFnegMuMuDPhi", "uGMT input muons #Delta#phi between EMTF- and OMTF-", 100, -0.5, 0.5);
    histograms.ugmtEOMTFnegMuMuDPhi.setAxisTitle("#Delta#phi", 1);

    histograms.ugmtEOMTFnegMuMuDR = booker.book1D("ugmtEOMTFnegMuMuDR", "uGMT input muons #DeltaR between EMTF- and OMTF-", 50, 0., 0.5);
    histograms.ugmtEOMTFnegMuMuDR.setAxisTitle("#DeltaR", 1);

  }

  // Subsystem Monitoring and Muon Output
  booker.setCurrentFolder(monitorDir);

  if (!emul) {
    histograms.ugmtBMTFBXvsProcessor = booker.book2D("ugmtBXvsProcessorBMTF", "uGMT BMTF Input BX vs Processor", 12, -0.5, 11.5, 5, -2.5, 2.5);
    histograms.ugmtBMTFBXvsProcessor.setAxisTitle("Wedge", 1);
    for (int bin = 1; bin <= 12; ++bin) {
      histograms.ugmtBMTFBXvsProcessor.setBinLabel(bin, std::to_string(bin), 1);
    }
    histograms.ugmtBMTFBXvsProcessor.setAxisTitle("BX", 2);

    histograms.ugmtOMTFBXvsProcessor = booker.book2D("ugmtBXvsProcessorOMTF", "uGMT OMTF Input BX vs Processor", 12, -0.5, 11.5, 5, -2.5, 2.5);
    histograms.ugmtOMTFBXvsProcessor.setAxisTitle("Sector (Detector Side)", 1);
    for (int bin = 1; bin <= 6; ++bin) {
      histograms.ugmtOMTFBXvsProcessor.setBinLabel(bin, std::to_string(7 - bin) + " (-)", 1);
      histograms.ugmtOMTFBXvsProcessor.setBinLabel(bin + 6, std::to_string(bin) + " (+)", 1);
    }
    histograms.ugmtOMTFBXvsProcessor.setAxisTitle("BX", 2);

    histograms.ugmtEMTFBXvsProcessor = booker.book2D("ugmtBXvsProcessorEMTF", "uGMT EMTF Input BX vs Processor", 12, -0.5, 11.5, 5, -2.5, 2.5);
    histograms.ugmtEMTFBXvsProcessor.setAxisTitle("Sector (Detector Side)", 1);
    for (int bin = 1; bin <= 6; ++bin) {
      histograms.ugmtEMTFBXvsProcessor.setBinLabel(bin, std::to_string(7 - bin) + " (-)", 1);
      histograms.ugmtEMTFBXvsProcessor.setBinLabel(bin + 6, std::to_string(bin) + " (+)", 1);
    }
    histograms.ugmtEMTFBXvsProcessor.setAxisTitle("BX", 2);

    histograms.ugmtBXvsLink = booker.book2D("ugmtBXvsLink", "uGMT BX vs Input Links", 36, 35.5, 71.5, 5, -2.5, 2.5);
    histograms.ugmtBXvsLink.setAxisTitle("Link", 1);
    histograms.ugmtBXvsLink.setAxisTitle("BX", 2);
  }
 
  histograms.ugmtMuonBX = booker.book1D("ugmtMuonBX", "uGMT Muon BX", 7, -3.5, 3.5);
  histograms.ugmtMuonBX.setAxisTitle("BX", 1);

  histograms.ugmtnMuons = booker.book1D("ugmtnMuons", "uGMT Muon Multiplicity", 9, -0.5, 8.5);
  histograms.ugmtnMuons.setAxisTitle("Muon Multiplicity (BX == 0)", 1);

  histograms.ugmtMuonIndex = booker.book1D("ugmtMuonIndex", "uGMT Input Muon Index", 108, -0.5, 107.5);
  histograms.ugmtMuonIndex.setAxisTitle("Index", 1);

  histograms.ugmtMuonhwPt = booker.book1D("ugmtMuonhwPt", "uGMT Muon p_{T}", 512, -0.5, 511.5);
  histograms.ugmtMuonhwPt.setAxisTitle("Hardware p_{T}", 1);

  histograms.ugmtMuonhwEta = booker.book1D("ugmtMuonhwEta", "uGMT Muon #eta", 461, -230.5, 230.5);
  histograms.ugmtMuonhwEta.setAxisTitle("Hardware Eta", 1);

  histograms.ugmtMuonhwPhi = booker.book1D("ugmtMuonhwPhi", "uGMT Muon #phi", 576, -0.5, 575.5);
  histograms.ugmtMuonhwPhi.setAxisTitle("Hardware Phi", 1);

  histograms.ugmtMuonhwEtaAtVtx = booker.book1D("ugmtMuonhwEtaAtVtx", "uGMT Muon #eta at vertex", 461, -230.5, 230.5);
  histograms.ugmtMuonhwEtaAtVtx.setAxisTitle("Hardware Eta at Vertex", 1);

  histograms.ugmtMuonhwPhiAtVtx = booker.book1D("ugmtMuonhwPhiAtVtx", "uGMT Muon #phi at vertex", 576, -0.5, 575.5);
  histograms.ugmtMuonhwPhiAtVtx.setAxisTitle("Hardware Phi at Vertex", 1);

  histograms.ugmtMuonhwCharge = booker.book1D("ugmtMuonhwCharge", "uGMT Muon Charge", 2, -0.5, 1.5);
  histograms.ugmtMuonhwCharge.setAxisTitle("Hardware Charge", 1);

  histograms.ugmtMuonhwChargeValid = booker.book1D("ugmtMuonhwChargeValid", "uGMT Muon ChargeValid", 2, -0.5, 1.5);
  histograms.ugmtMuonhwChargeValid.setAxisTitle("ChargeValid", 1);

  histograms.ugmtMuonhwQual = booker.book1D("ugmtMuonhwQual", "uGMT Muon Quality", 16, -0.5, 15.5);
  histograms.ugmtMuonhwQual.setAxisTitle("Quality", 1);

  histograms.ugmtMuonhwIso = booker.book1D("ugmtMuonhwIso", "uGMT Muon Isolation", 4, -0.5, 3.5);
  histograms.ugmtMuonhwIso.setAxisTitle("Isolation", 1);

  histograms.ugmtMuonPt = booker.book1D("ugmtMuonPt", "uGMT Muon p_{T}", 256, -0.5, 255.5);
  histograms.ugmtMuonPt.setAxisTitle("p_{T} [GeV]", 1);

  histograms.ugmtMuonEta = booker.book1D("ugmtMuonEta", "uGMT Muon #eta", 100, -2.5, 2.5);
  histograms.ugmtMuonEta.setAxisTitle("#eta", 1);

  histograms.ugmtMuonPhi = booker.book1D("ugmtMuonPhi", "uGMT Muon #phi", 126, -3.15, 3.15);
  histograms.ugmtMuonPhi.setAxisTitle("#phi", 1);

  histograms.ugmtMuonEtaAtVtx = booker.book1D("ugmtMuonEtaAtVtx", "uGMT Muon #eta at vertex", 100, -2.5, 2.5);
  histograms.ugmtMuonEtaAtVtx.setAxisTitle("#eta at vertex", 1);

  histograms.ugmtMuonPhiAtVtx = booker.book1D("ugmtMuonPhiAtVtx", "uGMT Muon #phi at vertex", 126, -3.15, 3.15);
  histograms.ugmtMuonPhiAtVtx.setAxisTitle("#phi at vertex", 1);

  histograms.ugmtMuonCharge = booker.book1D("ugmtMuonCharge", "uGMT Muon Charge", 3, -1.5, 1.5);
  histograms.ugmtMuonCharge.setAxisTitle("Charge", 1);

  histograms.ugmtMuonPhiBmtf = booker.book1D("ugmtMuonPhiBmtf", "uGMT Muon #phi for BMTF Inputs", 126, -3.15, 3.15);
  histograms.ugmtMuonPhiBmtf.setAxisTitle("#phi", 1);

  histograms.ugmtMuonPhiOmtf = booker.book1D("ugmtMuonPhiOmtf", "uGMT Muon #phi for OMTF Inputs", 126, -3.15, 3.15);
  histograms.ugmtMuonPhiOmtf.setAxisTitle("#phi", 1);

  histograms.ugmtMuonPhiEmtf = booker.book1D("ugmtMuonPhiEmtf", "uGMT Muon #phi for EMTF Inputs", 126, -3.15, 3.15);
  histograms.ugmtMuonPhiEmtf.setAxisTitle("#phi", 1);

  const float dPhiScale = 4*phiScale_;
  const float dEtaScale = etaScale_;
  histograms.ugmtMuonDEtavsPtBmtf = booker.book2D("ugmtMuonDEtavsPtBmtf", "uGMT Muon from BMTF #eta_{at vertex} - #eta_{at muon system} vs p_{T}", 32, 0, 64, 31, -15.5*dEtaScale, 15.5*dEtaScale);
  histograms.ugmtMuonDEtavsPtBmtf.setAxisTitle("p_{T} [GeV]", 1);
  histograms.ugmtMuonDEtavsPtBmtf.setAxisTitle("#eta_{at vertex} - #eta", 2);

  histograms.ugmtMuonDPhivsPtBmtf = booker.book2D("ugmtMuonDPhivsPtBmtf", "uGMT Muon from BMTF #phi_{at vertex} - #phi_{at muon system} vs p_{T}", 32, 0, 64, 31, -15.5*dPhiScale, 15.5*dPhiScale);
  histograms.ugmtMuonDPhivsPtBmtf.setAxisTitle("p_{T} [GeV]", 1);
  histograms.ugmtMuonDPhivsPtBmtf.setAxisTitle("#phi_{at vertex} - #phi", 2);

  histograms.ugmtMuonDEtavsPtOmtf = booker.book2D("ugmtMuonDEtavsPtOmtf", "uGMT Muon from OMTF #eta_{at vertex} - #eta_{at muon system} vs p_{T}", 32, 0, 64, 31, -15.5*dEtaScale, 15.5*dEtaScale);
  histograms.ugmtMuonDEtavsPtOmtf.setAxisTitle("p_{T} [GeV]", 1);
  histograms.ugmtMuonDEtavsPtOmtf.setAxisTitle("#eta_{at vertex} - #eta", 2);

  histograms.ugmtMuonDPhivsPtOmtf = booker.book2D("ugmtMuonDPhivsPtOmtf", "uGMT Muon from OMTF #phi_{at vertex} - #phi_{at muon system} vs p_{T}", 32, 0, 64, 31, -15.5*dPhiScale, 15.5*dPhiScale);
  histograms.ugmtMuonDPhivsPtOmtf.setAxisTitle("p_{T} [GeV]", 1);
  histograms.ugmtMuonDPhivsPtOmtf.setAxisTitle("#phi_{at vertex} - #phi", 2);

  histograms.ugmtMuonDEtavsPtEmtf = booker.book2D("ugmtMuonDEtavsPtEmtf", "uGMT Muon from EMTF #eta_{at vertex} - #eta_{at muon system} vs p_{T}", 32, 0, 64, 31, -15.5*dEtaScale, 15.5*dEtaScale);
  histograms.ugmtMuonDEtavsPtEmtf.setAxisTitle("p_{T} [GeV]", 1);
  histograms.ugmtMuonDEtavsPtEmtf.setAxisTitle("#eta_{at vertex} - #eta", 2);

  histograms.ugmtMuonDPhivsPtEmtf = booker.book2D("ugmtMuonDPhivsPtEmtf", "uGMT Muon from EMTF #phi_{at vertex} - #phi_{at muon system} vs p_{T}", 32, 0, 64, 31, -15.5*dPhiScale, 15.5*dPhiScale);
  histograms.ugmtMuonDPhivsPtEmtf.setAxisTitle("p_{T} [GeV]", 1);
  histograms.ugmtMuonDPhivsPtEmtf.setAxisTitle("#phi_{at vertex} - #phi", 2);

  histograms.ugmtMuonPtvsEta = booker.book2D("ugmtMuonPtvsEta", "uGMT Muon p_{T} vs #eta", 100, -2.5, 2.5, 256, -0.5, 255.5);
  histograms.ugmtMuonPtvsEta.setAxisTitle("#eta", 1);
  histograms.ugmtMuonPtvsEta.setAxisTitle("p_{T} [GeV]", 2);

  histograms.ugmtMuonPtvsPhi = booker.book2D("ugmtMuonPtvsPhi", "uGMT Muon p_{T} vs #phi", 64, -3.2, 3.2, 256, -0.5, 255.5);
  histograms.ugmtMuonPtvsPhi.setAxisTitle("#phi", 1);
  histograms.ugmtMuonPtvsPhi.setAxisTitle("p_{T} [GeV]", 2);

  histograms.ugmtMuonPhivsEta = booker.book2D("ugmtMuonPhivsEta", "uGMT Muon #phi vs #eta", 100, -2.5, 2.5, 64, -3.2, 3.2);
  histograms.ugmtMuonPhivsEta.setAxisTitle("#eta", 1);
  histograms.ugmtMuonPhivsEta.setAxisTitle("#phi", 2);

  histograms.ugmtMuonPhiAtVtxvsEtaAtVtx = booker.book2D("ugmtMuonPhiAtVtxvsEtaAtVtx", "uGMT Muon #phi at vertex vs #eta at vertex", 100, -2.5, 2.5, 64, -3.2, 3.2);
  histograms.ugmtMuonPhiAtVtxvsEtaAtVtx.setAxisTitle("#eta at vertex", 1);
  histograms.ugmtMuonPhiAtVtxvsEtaAtVtx.setAxisTitle("#phi at vertex", 2);

  histograms.ugmtMuonBXvsLink = booker.book2D("ugmtMuonBXvsLink", "uGMT Muon BX vs Input Links", 36, 35.5, 71.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvsLink.setAxisTitle("Muon Input Links", 1);
  histograms.ugmtMuonBXvsLink.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwPt = booker.book2D("ugmtMuonBXvshwPt", "uGMT Muon BX vs p_{T}", 256, -0.5, 511.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwPt.setAxisTitle("Hardware p_{T}", 1);
  histograms.ugmtMuonBXvshwPt.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwEta = booker.book2D("ugmtMuonBXvshwEta", "uGMT Muon BX vs #eta", 93, -232.5, 232.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwEta.setAxisTitle("Hardware #eta", 1);
  histograms.ugmtMuonBXvshwEta.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwPhi = booker.book2D("ugmtMuonBXvshwPhi", "uGMT Muon BX vs #phi", 116, -2.5, 577.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwPhi.setAxisTitle("Hardware #phi", 1);
  histograms.ugmtMuonBXvshwPhi.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwCharge = booker.book2D("ugmtMuonBXvshwCharge", "uGMT Muon BX vs Charge", 2, -0.5, 1.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwCharge.setAxisTitle("Hardware Charge", 1);
  histograms.ugmtMuonBXvshwCharge.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwChargeValid = booker.book2D("ugmtMuonBXvshwChargeValid", "uGMT Muon BX vs ChargeValid", 2, -0.5, 1.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwChargeValid.setAxisTitle("ChargeValid", 1);
  histograms.ugmtMuonBXvshwChargeValid.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwQual = booker.book2D("ugmtMuonBXvshwQual", "uGMT Muon BX vs Quality", 16, -0.5, 15.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwQual.setAxisTitle("Quality", 1);
  histograms.ugmtMuonBXvshwQual.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwIso = booker.book2D("ugmtMuonBXvshwIso", "uGMT Muon BX vs Isolation", 4, -0.5, 3.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwIso.setAxisTitle("Isolation", 1);
  histograms.ugmtMuonBXvshwIso.setAxisTitle("BX", 2);

  // muon correlations
  booker.setCurrentFolder(monitorDir + "/muon_correlations");

  histograms.ugmtMuMuInvMass = booker.book1D("ugmtMuMuInvMass", "uGMT dimuon invariant mass", 200, 0., 200.);
  histograms.ugmtMuMuInvMass.setAxisTitle("m(#mu#mu) [GeV]", 1);

  histograms.ugmtMuMuInvMassAtVtx = booker.book1D("ugmtMuMuInvMassAtVtx", "uGMT dimuon invariant mass with coordinates at vertex", 200, 0., 200.);
  histograms.ugmtMuMuInvMassAtVtx.setAxisTitle("m(#mu#mu) [GeV]", 1);

  histograms.ugmtMuMuDEta = booker.book1D("ugmtMuMuDEta", "uGMT Muons #Delta#eta", 100, -1., 1.);
  histograms.ugmtMuMuDEta.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhi = booker.book1D("ugmtMuMuDPhi", "uGMT Muons #Delta#phi", 100, -1., 1.);
  histograms.ugmtMuMuDPhi.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDR = booker.book1D("ugmtMuMuDR", "uGMT Muons #DeltaR", 50, 0., 1.);
  histograms.ugmtMuMuDR.setAxisTitle("#DeltaR", 1);

  // barrel - overlap
  histograms.ugmtMuMuDEtaBOpos = booker.book1D("ugmtMuMuDEtaBOpos", "uGMT Muons #Delta#eta barrel-overlap positive side", 100, -1., 1.);
  histograms.ugmtMuMuDEtaBOpos.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhiBOpos = booker.book1D("ugmtMuMuDPhiBOpos", "uGMT Muons #Delta#phi barrel-overlap positive side", 100, -1., 1.);
  histograms.ugmtMuMuDPhiBOpos.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDRBOpos = booker.book1D("ugmtMuMuDRBOpos", "uGMT Muons #DeltaR barrel-overlap positive side", 50, 0., 1.);
  histograms.ugmtMuMuDRBOpos.setAxisTitle("#DeltaR", 1);

  histograms.ugmtMuMuDEtaBOneg = booker.book1D("ugmtMuMuDEtaBOneg", "uGMT Muons #Delta#eta barrel-overlap negative side", 100, -1., 1.);
  histograms.ugmtMuMuDEtaBOneg.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhiBOneg = booker.book1D("ugmtMuMuDPhiBOneg", "uGMT Muons #Delta#phi barrel-overlap negative side", 100, -1., 1.);
  histograms.ugmtMuMuDPhiBOneg.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDRBOneg = booker.book1D("ugmtMuMuDRBOneg", "uGMT Muons #DeltaR barrel-overlap negative side", 50, 0., 1.);
  histograms.ugmtMuMuDRBOneg.setAxisTitle("#DeltaR", 1);

  // endcap - overlap
  histograms.ugmtMuMuDEtaEOpos = booker.book1D("ugmtMuMuDEtaEOpos", "uGMT Muons #Delta#eta endcap-overlap positive side", 100, -1., 1.);
  histograms.ugmtMuMuDEtaEOpos.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhiEOpos = booker.book1D("ugmtMuMuDPhiEOpos", "uGMT Muons #Delta#phi endcap-overlap positive side", 100, -1., 1.);
  histograms.ugmtMuMuDPhiEOpos.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDREOpos = booker.book1D("ugmtMuMuDREOpos", "uGMT Muons #DeltaR endcap-overlap positive side", 50, 0., 1.);
  histograms.ugmtMuMuDREOpos.setAxisTitle("#DeltaR", 1);

  histograms.ugmtMuMuDEtaEOneg = booker.book1D("ugmtMuMuDEtaEOneg", "uGMT Muons #Delta#eta endcap-overlap negative side", 100, -1., 1.);
  histograms.ugmtMuMuDEtaEOneg.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhiEOneg = booker.book1D("ugmtMuMuDPhiEOneg", "uGMT Muons #Delta#phi endcap-overlap negative side", 100, -1., 1.);
  histograms.ugmtMuMuDPhiEOneg.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDREOneg = booker.book1D("ugmtMuMuDREOneg", "uGMT Muons #DeltaR endcap-overlap negative side", 50, 0., 1.);
  histograms.ugmtMuMuDREOneg.setAxisTitle("#DeltaR", 1);

  // barrel wedges
  histograms.ugmtMuMuDEtaB = booker.book1D("ugmtMuMuDEtaB", "uGMT Muons #Delta#eta between barrel wedges", 100, -1., 1.);
  histograms.ugmtMuMuDEtaB.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhiB = booker.book1D("ugmtMuMuDPhiB", "uGMT Muons #Delta#phi between barrel wedges", 100, -1., 1.);
  histograms.ugmtMuMuDPhiB.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDRB = booker.book1D("ugmtMuMuDRB", "uGMT Muons #DeltaR between barrel wedges", 50, 0., 1.);
  histograms.ugmtMuMuDRB.setAxisTitle("#DeltaR", 1);

  // overlap sectors
  histograms.ugmtMuMuDEtaOpos = booker.book1D("ugmtMuMuDEtaOpos", "uGMT Muons #Delta#eta between overlap positive side sectors", 100, -1., 1.);
  histograms.ugmtMuMuDEtaOpos.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhiOpos = booker.book1D("ugmtMuMuDPhiOpos", "uGMT Muons #Delta#phi between overlap positive side sectors", 100, -1., 1.);
  histograms.ugmtMuMuDPhiOpos.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDROpos = booker.book1D("ugmtMuMuDROpos", "uGMT Muons #DeltaR between overlap positive side sectors", 50, 0., 1.);
  histograms.ugmtMuMuDROpos.setAxisTitle("#DeltaR", 1);

  histograms.ugmtMuMuDEtaOneg = booker.book1D("ugmtMuMuDEtaOneg", "uGMT Muons #Delta#eta between overlap negative side sectors", 100, -1., 1.);
  histograms.ugmtMuMuDEtaOneg.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhiOneg = booker.book1D("ugmtMuMuDPhiOneg", "uGMT Muons #Delta#phi between overlap negative side sectors", 100, -1., 1.);
  histograms.ugmtMuMuDPhiOneg.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDROneg = booker.book1D("ugmtMuMuDROneg", "uGMT Muons #DeltaR between overlap negative side sectors", 50, 0., 1.);
  histograms.ugmtMuMuDROneg.setAxisTitle("#DeltaR", 1);

  // endcap sectors
  histograms.ugmtMuMuDEtaEpos = booker.book1D("ugmtMuMuDEtaEpos", "uGMT Muons #Delta#eta between endcap positive side sectors", 100, -1., 1.);
  histograms.ugmtMuMuDEtaEpos.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhiEpos = booker.book1D("ugmtMuMuDPhiEpos", "uGMT Muons #Delta#phi between endcap positive side sectors", 100, -1., 1.);
  histograms.ugmtMuMuDPhiEpos.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDREpos = booker.book1D("ugmtMuMuDREpos", "uGMT Muons #DeltaR between endcap positive side sectors", 50, 0., 1.);
  histograms.ugmtMuMuDREpos.setAxisTitle("#DeltaR", 1);

  histograms.ugmtMuMuDEtaEneg = booker.book1D("ugmtMuMuDEtaEneg", "uGMT Muons #Delta#eta between endcap negative side sectors", 100, -1., 1.);
  histograms.ugmtMuMuDEtaEneg.setAxisTitle("#Delta#eta", 1);

  histograms.ugmtMuMuDPhiEneg = booker.book1D("ugmtMuMuDPhiEneg", "uGMT Muons #Delta#phi between endcap negative side sectors", 100, -1., 1.);
  histograms.ugmtMuMuDPhiEneg.setAxisTitle("#Delta#phi", 1);

  histograms.ugmtMuMuDREneg = booker.book1D("ugmtMuMuDREneg", "uGMT Muons #DeltaR between endcap negative side sectors", 50, 0., 1.);
  histograms.ugmtMuMuDREneg.setAxisTitle("#DeltaR", 1);
}

void L1TStage2uGMT::dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, ugmtdqm::Histograms const& histograms) const {

  if (verbose) edm::LogInfo("L1TStage2uGMT") << "L1TStage2uGMT: analyze..." << std::endl;

  if (!emul) {
    edm::Handle<l1t::RegionalMuonCandBxCollection> BMTFBxCollection;
    e.getByToken(ugmtBMTFToken, BMTFBxCollection);

    histograms.ugmtBMTFnMuons.fill(BMTFBxCollection->size(0));

    for (int itBX = BMTFBxCollection->getFirstBX(); itBX <= BMTFBxCollection->getLastBX(); ++itBX) {
      for (l1t::RegionalMuonCandBxCollection::const_iterator BMTF = BMTFBxCollection->begin(itBX); BMTF != BMTFBxCollection->end(itBX); ++BMTF) {
        histograms.ugmtBMTFBX.fill(itBX);
        histograms.ugmtBMTFhwPt.fill(BMTF->hwPt());
        histograms.ugmtBMTFhwEta.fill(BMTF->hwEta());
        histograms.ugmtBMTFhwPhi.fill(BMTF->hwPhi());
        histograms.ugmtBMTFhwSign.fill(BMTF->hwSign());
        histograms.ugmtBMTFhwSignValid.fill(BMTF->hwSignValid());
        histograms.ugmtBMTFhwQual.fill(BMTF->hwQual());
        histograms.ugmtBMTFlink.fill(BMTF->link());

        int global_hw_phi = l1t::MicroGMTConfiguration::calcGlobalPhi(BMTF->hwPhi(), BMTF->trackFinderType(), BMTF->processor());
        histograms.ugmtBMTFglbPhi.fill(global_hw_phi);

        histograms.ugmtBMTFBXvsProcessor.fill(BMTF->processor(), itBX);
        histograms.ugmtBMTFProcvshwPhi.fill(BMTF->hwPhi(), BMTF->processor());
        histograms.ugmtBXvsLink.fill(BMTF->link(), itBX);

        // Analyse muon correlations
        for (l1t::RegionalMuonCandBxCollection::const_iterator BMTF2 = BMTF+1; BMTF2 != BMTFBxCollection->end(itBX); ++BMTF2) {
          int global_hw_phi2 = l1t::MicroGMTConfiguration::calcGlobalPhi(BMTF2->hwPhi(), BMTF2->trackFinderType(), BMTF2->processor());
          float dEta = (BMTF->hwEta() - BMTF2->hwEta()) * etaScale_;
          float dPhi = (global_hw_phi - global_hw_phi2) * phiScale_;
          float dR = sqrt(dEta*dEta + dPhi*dPhi);

          int dLink = std::abs(BMTF->link() - BMTF2->link());
          if (dLink == 1 || dLink == 11) { // two adjacent wedges and wrap around
            histograms.ugmtBMTFMuMuDEta.fill(dEta);
            histograms.ugmtBMTFMuMuDPhi.fill(dPhi);
            histograms.ugmtBMTFMuMuDR.fill(dR);
          }
        }
      }
    }

    edm::Handle<l1t::RegionalMuonCandBxCollection> OMTFBxCollection;
    e.getByToken(ugmtOMTFToken, OMTFBxCollection);

    histograms.ugmtOMTFnMuons.fill(OMTFBxCollection->size(0));

    for (int itBX = OMTFBxCollection->getFirstBX(); itBX <= OMTFBxCollection->getLastBX(); ++itBX) {
      for (l1t::RegionalMuonCandBxCollection::const_iterator OMTF = OMTFBxCollection->begin(itBX); OMTF != OMTFBxCollection->end(itBX); ++OMTF) {
        histograms.ugmtOMTFBX.fill(itBX);
        histograms.ugmtOMTFhwPt.fill(OMTF->hwPt());
        histograms.ugmtOMTFhwEta.fill(OMTF->hwEta());
        histograms.ugmtOMTFhwSign.fill(OMTF->hwSign());
        histograms.ugmtOMTFhwSignValid.fill(OMTF->hwSignValid());
        histograms.ugmtOMTFhwQual.fill(OMTF->hwQual());
        histograms.ugmtOMTFlink.fill(OMTF->link());

        int global_hw_phi = l1t::MicroGMTConfiguration::calcGlobalPhi(OMTF->hwPhi(), OMTF->trackFinderType(), OMTF->processor());

        l1t::tftype trackFinderType = OMTF->trackFinderType();

        if (trackFinderType == l1t::omtf_neg) {
          histograms.ugmtOMTFBXvsProcessor.fill(5 - OMTF->processor(), itBX);
          histograms.ugmtOMTFhwPhiNeg.fill(OMTF->hwPhi());
          histograms.ugmtOMTFglbPhiNeg.fill(global_hw_phi);
          histograms.ugmtOMTFProcvshwPhiNeg.fill(OMTF->hwPhi(), OMTF->processor());
        } else {
          histograms.ugmtOMTFBXvsProcessor.fill(OMTF->processor() + 6, itBX);
          histograms.ugmtOMTFhwPhiPos.fill(OMTF->hwPhi());
          histograms.ugmtOMTFglbPhiPos.fill(global_hw_phi);
          histograms.ugmtOMTFProcvshwPhiPos.fill(OMTF->hwPhi(), OMTF->processor());
        }

        histograms.ugmtBXvsLink.fill(OMTF->link(), itBX);

        // Analyse muon correlations
        for (l1t::RegionalMuonCandBxCollection::const_iterator OMTF2 = OMTF+1; OMTF2 != OMTFBxCollection->end(itBX); ++OMTF2) {
          int global_hw_phi2 = l1t::MicroGMTConfiguration::calcGlobalPhi(OMTF2->hwPhi(), OMTF2->trackFinderType(), OMTF2->processor());
          float dEta = (OMTF->hwEta() - OMTF2->hwEta()) * etaScale_;
          float dPhi = (global_hw_phi - global_hw_phi2) * phiScale_;
          float dR = sqrt(dEta*dEta + dPhi*dPhi);

          int dLink = std::abs(OMTF->link() - OMTF2->link());
          if (dLink == 1 || dLink == 5) { // two adjacent sectors and wrap around
            histograms.ugmtOMTFMuMuDEta.fill(dEta);
            histograms.ugmtOMTFMuMuDPhi.fill(dPhi);
            histograms.ugmtOMTFMuMuDR.fill(dR);
          }
        }
      }
    }

    edm::Handle<l1t::RegionalMuonCandBxCollection> EMTFBxCollection;
    e.getByToken(ugmtEMTFToken, EMTFBxCollection);

    histograms.ugmtEMTFnMuons.fill(EMTFBxCollection->size(0));

    for (int itBX = EMTFBxCollection->getFirstBX(); itBX <= EMTFBxCollection->getLastBX(); ++itBX) {
      for (l1t::RegionalMuonCandBxCollection::const_iterator EMTF = EMTFBxCollection->begin(itBX); EMTF != EMTFBxCollection->end(itBX); ++EMTF) {
        histograms.ugmtEMTFBX.fill(itBX);
        histograms.ugmtEMTFhwPt.fill(EMTF->hwPt());
        histograms.ugmtEMTFhwEta.fill(EMTF->hwEta());
        histograms.ugmtEMTFhwSign.fill(EMTF->hwSign());
        histograms.ugmtEMTFhwSignValid.fill(EMTF->hwSignValid());
        histograms.ugmtEMTFhwQual.fill(EMTF->hwQual());
        histograms.ugmtEMTFlink.fill(EMTF->link());

        int global_hw_phi = l1t::MicroGMTConfiguration::calcGlobalPhi(EMTF->hwPhi(), EMTF->trackFinderType(), EMTF->processor());

        l1t::tftype trackFinderType = EMTF->trackFinderType();
        
        if (trackFinderType == l1t::emtf_neg) {
          histograms.ugmtEMTFBXvsProcessor.fill(5 - EMTF->processor(), itBX);
          histograms.ugmtEMTFhwPhiNeg.fill(EMTF->hwPhi());
          histograms.ugmtEMTFglbPhiNeg.fill(global_hw_phi);
          histograms.ugmtEMTFProcvshwPhiNeg.fill(EMTF->hwPhi(), EMTF->processor());
        } else {
          histograms.ugmtEMTFBXvsProcessor.fill(EMTF->processor() + 6, itBX);
          histograms.ugmtEMTFhwPhiPos.fill(EMTF->hwPhi());
          histograms.ugmtEMTFglbPhiPos.fill(global_hw_phi);
          histograms.ugmtEMTFProcvshwPhiPos.fill(EMTF->hwPhi(), EMTF->processor());
        }

        histograms.ugmtBXvsLink.fill(EMTF->link(), itBX);

        // Analyse muon correlations
        for (l1t::RegionalMuonCandBxCollection::const_iterator EMTF2 = EMTF+1; EMTF2 != EMTFBxCollection->end(itBX); ++EMTF2) {
          int global_hw_phi2 = l1t::MicroGMTConfiguration::calcGlobalPhi(EMTF2->hwPhi(), EMTF2->trackFinderType(), EMTF2->processor());
          float dEta = (EMTF->hwEta() - EMTF2->hwEta()) * etaScale_;
          float dPhi = (global_hw_phi - global_hw_phi2) * phiScale_;
          float dR = sqrt(dEta*dEta + dPhi*dPhi);

          int dLink = std::abs(EMTF->link() - EMTF2->link());
          if (dLink == 1 || dLink == 5) { // two adjacent sectors and wrap around
            histograms.ugmtEMTFMuMuDEta.fill(dEta);
            histograms.ugmtEMTFMuMuDPhi.fill(dPhi);
            histograms.ugmtEMTFMuMuDR.fill(dR);
          }
        }
      }
    }

    // barrel-overlap muon correlations
    int firstBxBO = (BMTFBxCollection->getFirstBX() < OMTFBxCollection->getFirstBX()) ? OMTFBxCollection->getFirstBX() : BMTFBxCollection->getFirstBX();
    int lastBxBO = (BMTFBxCollection->getLastBX() > OMTFBxCollection->getLastBX()) ? OMTFBxCollection->getLastBX() : BMTFBxCollection->getLastBX();
    for (int itBX = firstBxBO; itBX <= lastBxBO; ++itBX) {
      if (BMTFBxCollection->size(itBX) < 1 || OMTFBxCollection->size(itBX) < 1) {
        continue;
      }
      for (l1t::RegionalMuonCandBxCollection::const_iterator BMTF = BMTFBxCollection->begin(itBX); BMTF != BMTFBxCollection->end(itBX); ++BMTF) {
        int global_hw_phi_bmtf = l1t::MicroGMTConfiguration::calcGlobalPhi(BMTF->hwPhi(), BMTF->trackFinderType(), BMTF->processor());

        for (l1t::RegionalMuonCandBxCollection::const_iterator OMTF = OMTFBxCollection->begin(itBX); OMTF != OMTFBxCollection->end(itBX); ++OMTF) {
          int global_hw_phi_omtf = l1t::MicroGMTConfiguration::calcGlobalPhi(OMTF->hwPhi(), OMTF->trackFinderType(), OMTF->processor());
          float dEta = (BMTF->hwEta() - OMTF->hwEta()) * etaScale_;
          float dPhi = (global_hw_phi_bmtf - global_hw_phi_omtf) * phiScale_;
          float dR = sqrt(dEta*dEta + dPhi*dPhi);
          if (OMTF->trackFinderType() == l1t::omtf_neg) {
            histograms.ugmtBOMTFnegMuMuDEta.fill(dEta);
            histograms.ugmtBOMTFnegMuMuDPhi.fill(dPhi);
            histograms.ugmtBOMTFnegMuMuDR.fill(dR);
          } else {
            histograms.ugmtBOMTFposMuMuDEta.fill(dEta);
            histograms.ugmtBOMTFposMuMuDPhi.fill(dPhi);
            histograms.ugmtBOMTFposMuMuDR.fill(dR);
          }
        }
      }
    }

    // endcap-overlap muon correlations
    int firstBxEO = (EMTFBxCollection->getFirstBX() < OMTFBxCollection->getFirstBX()) ? OMTFBxCollection->getFirstBX() : EMTFBxCollection->getFirstBX();
    int lastBxEO = (EMTFBxCollection->getLastBX() > OMTFBxCollection->getLastBX()) ? OMTFBxCollection->getLastBX() : EMTFBxCollection->getLastBX();
    for (int itBX = firstBxEO; itBX <= lastBxEO; ++itBX) {
      if (EMTFBxCollection->size(itBX) < 1 || OMTFBxCollection->size(itBX) < 1) {
        continue;
      }
      for (l1t::RegionalMuonCandBxCollection::const_iterator EMTF = EMTFBxCollection->begin(itBX); EMTF != EMTFBxCollection->end(itBX); ++EMTF) {
        int global_hw_phi_emtf = l1t::MicroGMTConfiguration::calcGlobalPhi(EMTF->hwPhi(), EMTF->trackFinderType(), EMTF->processor());

        for (l1t::RegionalMuonCandBxCollection::const_iterator OMTF = OMTFBxCollection->begin(itBX); OMTF != OMTFBxCollection->end(itBX); ++OMTF) {
          int global_hw_phi_omtf = l1t::MicroGMTConfiguration::calcGlobalPhi(OMTF->hwPhi(), OMTF->trackFinderType(), OMTF->processor());
          float dEta = (EMTF->hwEta() - OMTF->hwEta()) * etaScale_;
          float dPhi = (global_hw_phi_emtf - global_hw_phi_omtf) * phiScale_;
          float dR = sqrt(dEta*dEta + dPhi*dPhi);
          if (EMTF->trackFinderType() == l1t::emtf_neg && OMTF->trackFinderType() == l1t::omtf_neg) {
            histograms.ugmtEOMTFnegMuMuDEta.fill(dEta);
            histograms.ugmtEOMTFnegMuMuDPhi.fill(dPhi);
            histograms.ugmtEOMTFnegMuMuDR.fill(dR);
          } else if (EMTF->trackFinderType() == l1t::emtf_pos && OMTF->trackFinderType() == l1t::omtf_pos) {
            histograms.ugmtEOMTFposMuMuDEta.fill(dEta);
            histograms.ugmtEOMTFposMuMuDPhi.fill(dPhi);
            histograms.ugmtEOMTFposMuMuDR.fill(dR);
          }
        }
      }
    }
  }

  edm::Handle<l1t::MuonBxCollection> MuonBxCollection;
  e.getByToken(ugmtMuonToken, MuonBxCollection);

  histograms.ugmtnMuons.fill(MuonBxCollection->size(0));

  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
    for (l1t::MuonBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX); Muon != MuonBxCollection->end(itBX); ++Muon) {

      int tfMuonIndex = Muon->tfMuonIndex();

      histograms.ugmtMuonBX.fill(itBX);
      histograms.ugmtMuonIndex.fill(tfMuonIndex);
      histograms.ugmtMuonhwPt.fill(Muon->hwPt());
      histograms.ugmtMuonhwEta.fill(Muon->hwEta());
      histograms.ugmtMuonhwPhi.fill(Muon->hwPhi());
      histograms.ugmtMuonhwEtaAtVtx.fill(Muon->hwEtaAtVtx());
      histograms.ugmtMuonhwPhiAtVtx.fill(Muon->hwPhiAtVtx());
      histograms.ugmtMuonhwCharge.fill(Muon->hwCharge());
      histograms.ugmtMuonhwChargeValid.fill(Muon->hwChargeValid());
      histograms.ugmtMuonhwQual.fill(Muon->hwQual());
      histograms.ugmtMuonhwIso.fill(Muon->hwIso());

      histograms.ugmtMuonPt.fill(Muon->pt());
      histograms.ugmtMuonEta.fill(Muon->eta());
      histograms.ugmtMuonPhi.fill(Muon->phi());
      histograms.ugmtMuonEtaAtVtx.fill(Muon->etaAtVtx());
      histograms.ugmtMuonPhiAtVtx.fill(Muon->phiAtVtx());
      histograms.ugmtMuonCharge.fill(Muon->charge());

      l1t::tftype tfType{getTfOrigin(tfMuonIndex)};
      if (tfType == l1t::emtf_pos || tfType == l1t::emtf_neg) {
        histograms.ugmtMuonPhiEmtf.fill(Muon->phi());
        histograms.ugmtMuonDEtavsPtEmtf.fill(Muon->pt(), Muon->hwDEtaExtra()*etaScale_);
        histograms.ugmtMuonDPhivsPtEmtf.fill(Muon->pt(), Muon->hwDPhiExtra()*phiScale_);
      } else if (tfType == l1t::omtf_pos || tfType == l1t::omtf_neg) {
        histograms.ugmtMuonPhiOmtf.fill(Muon->phi());
        histograms.ugmtMuonDEtavsPtOmtf.fill(Muon->pt(), Muon->hwDEtaExtra()*etaScale_);
        histograms.ugmtMuonDPhivsPtOmtf.fill(Muon->pt(), Muon->hwDPhiExtra()*phiScale_);
      } else if (tfType == l1t::bmtf) {
        histograms.ugmtMuonPhiBmtf.fill(Muon->phi());
        histograms.ugmtMuonDEtavsPtBmtf.fill(Muon->pt(), Muon->hwDEtaExtra()*etaScale_);
        histograms.ugmtMuonDPhivsPtBmtf.fill(Muon->pt(), Muon->hwDPhiExtra()*phiScale_);
      }

      histograms.ugmtMuonPtvsEta.fill(Muon->eta(), Muon->pt());
      histograms.ugmtMuonPtvsPhi.fill(Muon->phi(), Muon->pt());
      histograms.ugmtMuonPhivsEta.fill(Muon->eta(), Muon->phi());

      histograms.ugmtMuonPhiAtVtxvsEtaAtVtx.fill(Muon->etaAtVtx(), Muon->phiAtVtx());

      histograms.ugmtMuonBXvsLink.fill(int(Muon->tfMuonIndex()/3.) + 36, itBX);
      histograms.ugmtMuonBXvshwPt.fill(Muon->hwPt(), itBX);
      histograms.ugmtMuonBXvshwEta.fill(Muon->hwEta(), itBX);
      histograms.ugmtMuonBXvshwPhi.fill(Muon->hwPhi(), itBX);
      histograms.ugmtMuonBXvshwCharge.fill(Muon->hwCharge(), itBX);
      histograms.ugmtMuonBXvshwChargeValid.fill(Muon->hwChargeValid(), itBX);
      histograms.ugmtMuonBXvshwQual.fill(Muon->hwQual(), itBX);
      histograms.ugmtMuonBXvshwIso.fill(Muon->hwIso(), itBX);

      int link = (int)std::floor(tfMuonIndex / 3.);
      reco::Candidate::PolarLorentzVector mu1{Muon->pt(), Muon->eta(), Muon->phi(), 0.106};
      reco::Candidate::PolarLorentzVector muAtVtx1{Muon->pt(), Muon->etaAtVtx(), Muon->phiAtVtx(), 0.106};

      // Analyse multi muon events
      for (l1t::MuonBxCollection::const_iterator Muon2 = Muon+1; Muon2 != MuonBxCollection->end(itBX); ++Muon2) {
        reco::Candidate::PolarLorentzVector mu2{Muon2->pt(), Muon2->eta(), Muon2->phi(), 0.106};
        reco::Candidate::PolarLorentzVector muAtVtx2{Muon2->pt(), Muon2->etaAtVtx(), Muon2->phiAtVtx(), 0.106};
        histograms.ugmtMuMuInvMass.fill((mu1 + mu2).M());
        histograms.ugmtMuMuInvMassAtVtx.fill((muAtVtx1 + muAtVtx2).M());

        float dEta = Muon->eta() - Muon2->eta();
        float dPhi = Muon->phi() - Muon2->phi();
        float dR = sqrt(dEta*dEta + dPhi*dPhi);
        histograms.ugmtMuMuDEta.fill(dEta);
        histograms.ugmtMuMuDPhi.fill(dPhi);
        histograms.ugmtMuMuDR.fill(dR);

        // muon distances between muons from different TFs and from different wedges/sectors of one TF
        int link2 = (int)std::floor(Muon2->tfMuonIndex() / 3.);
        l1t::tftype tfType2{getTfOrigin(Muon2->tfMuonIndex())};
        if ((tfType == l1t::bmtf && tfType2 == l1t::omtf_pos) || (tfType == l1t::omtf_pos && tfType2 == l1t::bmtf)) {
          histograms.ugmtMuMuDEtaBOpos.fill(dEta);
          histograms.ugmtMuMuDPhiBOpos.fill(dPhi);
          histograms.ugmtMuMuDRBOpos.fill(dR);
        } else if ((tfType == l1t::bmtf && tfType2 == l1t::omtf_neg) || (tfType == l1t::omtf_neg && tfType2 == l1t::bmtf)) {
          histograms.ugmtMuMuDEtaBOneg.fill(dEta);
          histograms.ugmtMuMuDPhiBOneg.fill(dPhi);
          histograms.ugmtMuMuDRBOneg.fill(dR);
        } else if ((tfType == l1t::emtf_pos && tfType2 == l1t::omtf_pos) || (tfType == l1t::omtf_pos && tfType2 == l1t::emtf_pos)) {
          histograms.ugmtMuMuDEtaEOpos.fill(dEta);
          histograms.ugmtMuMuDPhiEOpos.fill(dPhi);
          histograms.ugmtMuMuDREOpos.fill(dR);
        } else if ((tfType == l1t::emtf_neg && tfType2 == l1t::omtf_neg) || (tfType == l1t::omtf_neg && tfType2 == l1t::emtf_neg)) {
          histograms.ugmtMuMuDEtaEOneg.fill(dEta);
          histograms.ugmtMuMuDPhiEOneg.fill(dPhi);
          histograms.ugmtMuMuDREOneg.fill(dR);
        } else if (tfType == l1t::bmtf && tfType2 == l1t::bmtf) {
          if (std::abs(link - link2) == 1 || (std::abs(link - link2) == 11)) { // two adjacent wedges and wrap around
            histograms.ugmtMuMuDEtaB.fill(dEta);
            histograms.ugmtMuMuDPhiB.fill(dPhi);
            histograms.ugmtMuMuDRB.fill(dR);
          }
        } else if (tfType == l1t::omtf_pos && tfType2 == l1t::omtf_pos) {
          if (std::abs(link - link2) == 1 || (std::abs(link - link2) == 5)) { // two adjacent sectors and wrap around
            histograms.ugmtMuMuDEtaOpos.fill(dEta);
            histograms.ugmtMuMuDPhiOpos.fill(dPhi);
            histograms.ugmtMuMuDROpos.fill(dR);
          }
        } else if (tfType == l1t::omtf_neg && tfType2 == l1t::omtf_neg) {
          if (std::abs(link - link2) == 1 || (std::abs(link - link2) == 5)) { // two adjacent sectors and wrap around
            histograms.ugmtMuMuDEtaOneg.fill(dEta);
            histograms.ugmtMuMuDPhiOneg.fill(dPhi);
            histograms.ugmtMuMuDROneg.fill(dR);
          }
        } else if (tfType == l1t::emtf_pos && tfType2 == l1t::emtf_pos) {
          if (std::abs(link - link2) == 1 || (std::abs(link - link2) == 5)) { // two adjacent sectors and wrap around
            histograms.ugmtMuMuDEtaEpos.fill(dEta);
            histograms.ugmtMuMuDPhiEpos.fill(dPhi);
            histograms.ugmtMuMuDREpos.fill(dR);
          }
        } else if (tfType == l1t::emtf_neg && tfType2 == l1t::emtf_neg) {
          if (std::abs(link - link2) == 1 || (std::abs(link - link2) == 5)) { // two adjacent sectors and wrap around
            histograms.ugmtMuMuDEtaEneg.fill(dEta);
            histograms.ugmtMuMuDPhiEneg.fill(dPhi);
            histograms.ugmtMuMuDREneg.fill(dR);
          }
        }
      }
    }
  }
}

l1t::tftype L1TStage2uGMT::getTfOrigin(int tfMuonIndex) const
{
  if (tfMuonIndex >= 0 && tfMuonIndex <=17) {
    return l1t::emtf_pos;
  } else if (tfMuonIndex >= 90 && tfMuonIndex <=107) {
    return l1t::emtf_neg;
  } else if (tfMuonIndex >= 18 && tfMuonIndex <=35) {
    return l1t::omtf_pos;
  } else if (tfMuonIndex >= 72 && tfMuonIndex <=89) {
    return l1t::omtf_neg;
  } else {
    return l1t::bmtf;
  }
}

