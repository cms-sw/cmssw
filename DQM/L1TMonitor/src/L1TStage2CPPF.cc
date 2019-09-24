#include "DQM/L1TMonitor/interface/L1TStage2CPPF.h"
#include "DQM/RPCMonitorDigi/interface/utils.h"

using namespace std;
using namespace edm;

L1TStage2CPPF::L1TStage2CPPF(const ParameterSet& ps)
    : rpcdigiSource_(ps.getParameter<InputTag>("rpcdigiSource")),
      rpcdigiSource_token_(consumes<RPCDigiCollection>(ps.getParameter<InputTag>("rpcdigiSource"))),
      cppfdigiSource_token_(consumes<l1t::CPPFDigiCollection>(ps.getParameter<InputTag>("rpcdigiSource"))) {
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if (verbose_)
    cout << "L1TStage2CPPF: constructor...." << endl;

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if (!outputFile_.empty()) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }
}

L1TStage2CPPF::~L1TStage2CPPF() {}

void L1TStage2CPPF::dqmBeginRun(edm::Run const& r, edm::EventSetup const& c) {
  //
}

void L1TStage2CPPF::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  nev_ = 0;

  ibooker.setCurrentFolder("L1T/L1TCPPF/Inputs");

  int numberOfDisks_ = 4;
  int numberOfRings_ = 2;
  std::stringstream histoName;
  std::stringstream os;
  bool useNormalization_ = true;
  bool useRollInfo_ = false;

  rpcdqm::utils rpcUtils;

  for (int d = -numberOfDisks_; d <= numberOfDisks_; d++) {
    if (d == 0)
      continue;

    int offset = numberOfDisks_;
    if (d > 0)
      offset--;  //used to skip case equale to zero

    histoName.str("");

    if (useNormalization_) {
      histoName.str("");
      histoName << "CPPFInput_OccupancyNormByEvents_Disk" << d;
      CPPFInputNormOccupDisk[d + offset] = ibooker.book2D(
          histoName.str().c_str(), histoName.str().c_str(), 36, 0.5, 36.5, numberOfRings_, 0.5, numberOfRings_ + 0.5);

      rpcUtils.labelXAxisSegment(CPPFInputNormOccupDisk[d + offset]);
      rpcUtils.labelYAxisRing(CPPFInputNormOccupDisk[d + offset], numberOfRings_, useRollInfo_);
    }

    for (int ring = 2; ring <= 3; ring++) {
      for (int region = -1; region < 2; region += 2) {
        os.str("");
        os << "CPPFInput_Occupancy_Disk_" << (region * d) << "_Ring_" << ring << "_CH01-CH18";

        meInputDiskRing_1st[os.str()] = ibooker.book2D(os.str(), os.str(), 96, 0.5, 96.5, 18, 0.5, 18.5);
        meInputDiskRing_1st[os.str()]->setAxisTitle("strip", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meInputDiskRing_1st[os.str()]);

        std::stringstream yLabel;
        for (int i = 1; i <= 18; i++) {
          yLabel.str("");
          yLabel << "R" << ring << "_CH" << std::setw(2) << std::setfill('0') << i;
          meInputDiskRing_1st[os.str()]->setBinLabel(i, yLabel.str(), 2);
        }

        for (int i = 1; i <= 96; i++) {
          if (i == 1)
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 16)
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "RollA", 1);
          else if (i == 32)
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "32", 1);
          else if (i == 33)
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 48)
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "RollB", 1);
          else if (i == 64)
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "32", 1);
          else if (i == 65)
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 80)
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "RollC", 1);
          else if (i == 96)
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "32", 1);
          else
            meInputDiskRing_1st[os.str()]->setBinLabel(i, "", 1);
        }

        os.str("");
        os << "CPPFInput_Occupancy_Disk_" << (region * d) << "_Ring_" << ring << "_CH19-CH36";

        meInputDiskRing_2nd[os.str()] = ibooker.book2D(os.str(), os.str(), 96, 0.5, 96.5, 18, 18.5, 36.5);
        meInputDiskRing_2nd[os.str()]->setAxisTitle("strip", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meInputDiskRing_2nd[os.str()]);

        for (int i = 1; i <= 18; i++) {
          yLabel.str("");
          yLabel << "R" << ring << "_CH" << i + 18;
          meInputDiskRing_2nd[os.str()]->setBinLabel(i, yLabel.str(), 2);
        }

        for (int i = 1; i <= 96; i++) {
          if (i == 1)
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 16)
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "RollA", 1);
          else if (i == 32)
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "32", 1);
          else if (i == 33)
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 48)
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "RollB", 1);
          else if (i == 64)
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "32", 1);
          else if (i == 65)
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 80)
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "RollC", 1);
          else if (i == 96)
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "32", 1);
          else
            meInputDiskRing_2nd[os.str()]->setBinLabel(i, "", 1);
        }
      }  //loop on region
    }    //loop ring

  }  //End loop on Endcap disk

  const std::array<std::string, 6> CPPF_name{{"43", "42", "33", "32", "22", "12"}};
  const std::array<std::string, 6> CPPF_label{{"4/3", "4/2", "3/3", "3/2", "2/2", "1/2"}};

  CPPFInput_DiskRing_Vs_BX = ibooker.book2D("CPPFInput_DiskRing_Vs_BX", "CPPF Input BX", 7, -3, 4, 12, 0, 12);
  CPPFInput_DiskRing_Vs_BX->setAxisTitle("BX", 1);

  for (int xbin = 1, xbin_label = -3; xbin <= 7; ++xbin, ++xbin_label) {
    CPPFInput_DiskRing_Vs_BX->setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  for (int ybin = 1; ybin <= 6; ++ybin) {
    CPPFInput_DiskRing_Vs_BX->setBinLabel(ybin, "RE-" + CPPF_label[ybin - 1], 2);
    CPPFInput_DiskRing_Vs_BX->setBinLabel(13 - ybin, "RE+" + CPPF_label[ybin - 1], 2);
  }

  CPPFInput_Occupancy_DiskRing_Vs_Segment = ibooker.book2D(
      "CPPFInput_Occupancy_DiskRing_Vs_Segment", "CPPF Input Chamber Occupancy", 36, 0.5, 36.5, 12, 0, 12);
  CPPFInput_Occupancy_DiskRing_Vs_Segment->setAxisTitle("Segment", 1);
  for (int bin = 1; bin < 7; ++bin) {
    CPPFInput_Occupancy_DiskRing_Vs_Segment->setBinLabel(bin, "RE-" + CPPF_label[bin - 1], 2);
    CPPFInput_Occupancy_DiskRing_Vs_Segment->setBinLabel(13 - bin, "RE+" + CPPF_label[bin - 1], 2);
  }
  rpcUtils.labelXAxisSegment(CPPFInput_Occupancy_DiskRing_Vs_Segment);
  CPPFInput_Occupancy_DiskRing_Vs_Segment->getTH2F()->GetXaxis()->SetCanExtend(
      false);  // Needed to stop multi-thread summing

  CPPFInput_Occupancy_Ring_Vs_Disk = ibooker.book2D("CPPFInput_Occupancy_Ring_Vs_Disk",
                                                    "CPPF Input Occupancy Ring Vs Disk",
                                                    (int)numberOfDisks_ * 2.0,
                                                    0.5,
                                                    ((float)numberOfDisks_ * 2.0) + 0.5,
                                                    2,
                                                    1.5,
                                                    3.5);
  CPPFInput_Occupancy_Ring_Vs_Disk->setAxisTitle("Disk", 1);  // X axis title

  std::stringstream binlabel;
  for (int bin = 1; bin <= numberOfDisks_ * 2; bin++) {
    binlabel.str("");
    if (bin < (numberOfDisks_ + 1)) {  //negative endcap
      binlabel << (bin - (numberOfDisks_ + 1));
    } else {  //positive endcaps
      binlabel << (bin - numberOfDisks_);
    }
    CPPFInput_Occupancy_Ring_Vs_Disk->setBinLabel(bin, binlabel.str(), 1);  //X axis bin label
  }
  CPPFInput_Occupancy_Ring_Vs_Disk->setBinLabel(1, "Ring2", 2);  // Y axis bin label
  CPPFInput_Occupancy_Ring_Vs_Disk->setBinLabel(2, "Ring3", 2);  // Y axis bin label

  ibooker.setCurrentFolder("L1T/L1TCPPF/Outputs");
  for (int d = -numberOfDisks_; d <= numberOfDisks_; d++) {
    if (d == 0)
      continue;

    int offset = numberOfDisks_;
    if (d > 0)
      offset--;  //used to skip case equale to zero

    histoName.str("");

    if (useNormalization_) {
      histoName.str("");
      histoName << "CPPFOutput_OccupancyNormByEvents_Disk" << d;
      CPPFOutputNormOccupDisk[d + offset] = ibooker.book2D(
          histoName.str().c_str(), histoName.str().c_str(), 36, 0.5, 36.5, numberOfRings_, 0.5, numberOfRings_ + 0.5);

      rpcUtils.labelXAxisSegment(CPPFOutputNormOccupDisk[d + offset]);
      rpcUtils.labelYAxisRing(CPPFOutputNormOccupDisk[d + offset], numberOfRings_, useRollInfo_);
    }

    for (int ring = 2; ring <= 3; ring++) {
      for (int region = -1; region < 2; region += 2) {
        /*
        os.str("");
        os<<"CPPFOutput_Disk_"<<(region * d)<<"_Ring_"<<ring<<"_board_Vs_Segment";
      
        meOutputDiskRing_board[os.str()] = ibooker.book2D(os.str(), os.str(), 36, 0.5, 36.5, 11 , -99.5,  10.5);
        meOutputDiskRing_board[os.str()]->setAxisTitle("Segment", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meOutputDiskRing_board[os.str()]);
      
        rpcUtils.labelXAxisSegment(meOutputDiskRing_board[os.str()]);


        os.str("");
        os<<"CPPFOutput_Disk_"<<(region * d)<<"_Ring_"<<ring<<"_channel_Vs_Segment";
      
        meOutputDiskRing_channel[os.str()] = ibooker.book2D(os.str(), os.str(), 36, 0.5, 36.5, 11 , -99.5,  10.5);
        meOutputDiskRing_channel[os.str()]->setAxisTitle("Segment", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meOutputDiskRing_channel[os.str()]);
      
        rpcUtils.labelXAxisSegment(meOutputDiskRing_channel[os.str()]);

        os.str("");
        os<<"CPPFOutput_Disk_"<<(region * d)<<"_Ring_"<<ring<<"_emtf_sector_Vs_Segment";
      
        meOutputDiskRing_emtf_sector[os.str()] = ibooker.book2D(os.str(), os.str(), 36, 0.5, 36.5, 11 , -99.5,  10.5);
        meOutputDiskRing_emtf_sector[os.str()]->setAxisTitle("Segment", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meOutputDiskRing_emtf_sector[os.str()]);
      
        rpcUtils.labelXAxisSegment(meOutputDiskRing_emtf_sector[os.str()]);

        os.str("");
        os<<"CPPFOutput_Disk_"<<(region * d)<<"_Ring_"<<ring<<"_emtf_link_Vs_Segment";
      
        meOutputDiskRing_emtf_link[os.str()] = ibooker.book2D(os.str(), os.str(), 36, 0.5, 36.5, 11 , -99.5,  10.5);
        meOutputDiskRing_emtf_link[os.str()]->setAxisTitle("Segment", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meOutputDiskRing_emtf_link[os.str()]);
      
        rpcUtils.labelXAxisSegment(meOutputDiskRing_emtf_link[os.str()]);
        */
        os.str("");
        os << "CPPFOutput_Disk_" << (region * d) << "_Ring_" << ring << "_Theta_Vs_Segment";

        meOutputDiskRing_theta[os.str()] = ibooker.book2D(os.str(), os.str(), 36, 0.5, 36.5, 32, 0, 32);
        meOutputDiskRing_theta[os.str()]->setAxisTitle("Segment", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meOutputDiskRing_theta[os.str()]);

        rpcUtils.labelXAxisSegment(meOutputDiskRing_theta[os.str()]);

        os.str("");
        os << "CPPFOutput_Disk_" << (region * d) << "_Ring_" << ring << "_Phi_Vs_Segment";

        meOutputDiskRing_phi[os.str()] = ibooker.book2D(os.str(), os.str(), 36, 0.5, 36.5, 1250, 0, 1250);
        meOutputDiskRing_phi[os.str()]->setAxisTitle("Segment", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meOutputDiskRing_phi[os.str()]);

        rpcUtils.labelXAxisSegment(meOutputDiskRing_phi[os.str()]);

      }  //loop on region
    }    //loop ring

  }  //End loop on Endcap disk

  CPPFOutput_DiskRing_Vs_BX = ibooker.book2D("CPPFOutput_DiskRing_Vs_BX", "CPPF Digi BX", 7, -3, 4, 12, 0, 12);
  CPPFOutput_DiskRing_Vs_BX->setAxisTitle("BX", 1);

  for (int xbin = 1, xbin_label = -3; xbin <= 7; ++xbin, ++xbin_label) {
    CPPFOutput_DiskRing_Vs_BX->setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  for (int ybin = 1; ybin <= 6; ++ybin) {
    CPPFOutput_DiskRing_Vs_BX->setBinLabel(ybin, "RE-" + CPPF_label[ybin - 1], 2);
    CPPFOutput_DiskRing_Vs_BX->setBinLabel(13 - ybin, "RE+" + CPPF_label[ybin - 1], 2);
  }

  CPPFOutput_Occupancy_DiskRing_Vs_Segment =
      ibooker.book2D("CPPFOutput_Occupancy_DiskRing_Vs_Segment", "CPPF Chamber Occupancy", 36, 0.5, 36.5, 12, 0, 12);
  CPPFOutput_Occupancy_DiskRing_Vs_Segment->setAxisTitle("Segment", 1);
  for (int bin = 1; bin < 7; ++bin) {
    CPPFOutput_Occupancy_DiskRing_Vs_Segment->setBinLabel(bin, "RE-" + CPPF_label[bin - 1], 2);
    CPPFOutput_Occupancy_DiskRing_Vs_Segment->setBinLabel(13 - bin, "RE+" + CPPF_label[bin - 1], 2);
  }
  rpcUtils.labelXAxisSegment(CPPFOutput_Occupancy_DiskRing_Vs_Segment);
  CPPFOutput_Occupancy_DiskRing_Vs_Segment->getTH2F()->GetXaxis()->SetCanExtend(
      false);  // Needed to stop multi-thread summing

  for (int hist = 0, i = 0; hist < 12; ++hist, i = hist % 6) {
    std::string name, label;
    if (hist < 6) {
      name = "RENeg" + CPPF_name[i];
      label = "RE-" + CPPF_label[i];
    } else {
      name = "REPos" + CPPF_name[5 - i];
      label = "RE+" + CPPF_label[5 - i];
    }
    CPPFOutput_1DPhi[hist] = ibooker.book1D("CPPFOutput_1D_Phi_" + name, "CPPF Digi #phi " + label, 1250, 0, 1250);
    CPPFOutput_1DPhi[hist]->setAxisTitle("#phi", 1);
    CPPFOutput_1DTheta[hist] = ibooker.book1D("CPPFOutput_1D_Theta_" + name, "CPPF Digi #theta " + label, 32, 0, 32);
    CPPFOutput_1DTheta[hist]->setAxisTitle("#theta", 1);
  }

  CPPFOutput_Occupancy_Ring_Vs_Disk = ibooker.book2D("CPPFOutput_Occupancy_Ring_Vs_Disk",
                                                     "CPPF Output Occupancy Ring Vs Disk",
                                                     (int)numberOfDisks_ * 2.0,
                                                     0.5,
                                                     ((float)numberOfDisks_ * 2.0) + 0.5,
                                                     2,
                                                     1.5,
                                                     3.5);
  CPPFOutput_Occupancy_Ring_Vs_Disk->setAxisTitle("Disk", 1);  // X axis title

  for (int bin = 1; bin <= numberOfDisks_ * 2; bin++) {
    binlabel.str("");
    if (bin < (numberOfDisks_ + 1)) {  //negative endcap
      binlabel << (bin - (numberOfDisks_ + 1));
    } else {  //positive endcaps
      binlabel << (bin - numberOfDisks_);
    }
    CPPFOutput_Occupancy_Ring_Vs_Disk->setBinLabel(bin, binlabel.str(), 1);  //X axis bin label
  }

  CPPFOutput_Occupancy_Ring_Vs_Disk->setBinLabel(1, "Ring2", 2);  // Y axis bin label
  CPPFOutput_Occupancy_Ring_Vs_Disk->setBinLabel(2, "Ring3", 2);  // Y axis bin label
}

void L1TStage2CPPF::analyze(const Event& e, const EventSetup& c) {
  nev_++;
  if (verbose_)
    cout << "L1TStage2CPPF: analyze...." << endl;

  /// RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeo;
  c.get<MuonGeometryRecord>().get(rpcGeo);
  if (!rpcGeo.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find RPCGeometry" << endl;
    cout << " **** can't find RPCGeometry " << endl;
    return;
  }

  /// DIGI
  edm::Handle<RPCDigiCollection> rpcdigis;
  e.getByToken(rpcdigiSource_token_, rpcdigis);

  if (!rpcdigis.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find RPCDigiCollection with label " << rpcdigiSource_ << endl;
    cout << " **** can't find RPCDigiCollection with label " << rpcdigiSource_ << endl;
    //return; //attention
  }

  /// DIGI
  edm::Handle<l1t::CPPFDigiCollection> cppfdigis;
  e.getByToken(cppfdigiSource_token_, cppfdigis);

  if (!cppfdigis.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find CPPFDigiCollection with label " << rpcdigiSource_ << endl;
    cout << " **** can't find CPPFDigiCollection with label " << rpcdigiSource_ << endl;
    //return; //attention
  }

  const std::map<std::pair<int, int>, int> histIndexCPPF = {
      {{4, 3}, 0}, {{4, 2}, 1}, {{3, 3}, 2}, {{3, 2}, 3}, {{2, 2}, 4}, {{1, 2}, 5}};

  bool useRollInfo_ = false;
  std::stringstream os;

  RPCDigiCollection::DigiRangeIterator collectionItr;
  for (collectionItr = rpcdigis->begin(); collectionItr != rpcdigis->end(); ++collectionItr) {
    const RPCDetId& detId = (*collectionItr).first;

    rpcdqm::utils rpcUtils;

    RPCGeomServ geoServ(detId);
    std::string nameRoll = "";

    if (useRollInfo_)
      nameRoll = geoServ.name();
    else
      nameRoll = geoServ.chambername();

    int n_digi = 0;
    RPCDigiCollection::const_iterator digiItr;
    for (digiItr = ((*collectionItr).second).first; digiItr != ((*collectionItr).second).second; ++digiItr, ++n_digi) {
      int region = (int)detId.region();
      int wheelOrDiskNumber;
      std::string wheelOrDiskType;
      int ring = 0;
      int station = detId.station();
      wheelOrDiskType = "Disk";
      wheelOrDiskNumber = region * (int)detId.station();
      ring = detId.ring();
      // strips is a list of hit strips (regardless of bx) for this roll
      int strip = (*digiItr).strip();
      int bx = (*digiItr).bx();

      int xBin, yBin;
      xBin = geoServ.segment();
      //yBin = (detId.ring()-1)*3-roll+1;
      yBin = detId.ring() - 1;

      if (region != 0) {
        os.str("");
        if (geoServ.segment() > 0 && geoServ.segment() < 19) {
          os << "CPPFInput_Occupancy_" << wheelOrDiskType << "_" << wheelOrDiskNumber << "_Ring_" << ring
             << "_CH01-CH18";
          if (meInputDiskRing_1st[os.str()]) {
            meInputDiskRing_1st[os.str()]->Fill(strip + 32 * (detId.roll() - 1), geoServ.segment());
          }
        } else if (geoServ.segment() > 18) {
          os << "CPPFInput_Occupancy_" << wheelOrDiskType << "_" << wheelOrDiskNumber << "_Ring_" << ring
             << "_CH19-CH36";
          if (meInputDiskRing_2nd[os.str()]) {
            meInputDiskRing_2nd[os.str()]->Fill(strip + 32 * (detId.roll() - 1), geoServ.segment());
          }
        }

        if (wheelOrDiskNumber > 0)
          wheelOrDiskNumber--;
        CPPFInputNormOccupDisk[wheelOrDiskNumber + 4]->Fill(xBin, yBin, 1);
        int hist_index = histIndexCPPF.at({station, ring});
        if (region > 0)
          hist_index = 11 - hist_index;

        CPPFInput_DiskRing_Vs_BX->Fill(bx, hist_index);
        CPPFInput_Occupancy_DiskRing_Vs_Segment->Fill(geoServ.segment(), hist_index + 0.5);

        int Ring_Vs_Disk_xbin = wheelOrDiskNumber + 4 + 1;
        CPPFInput_Occupancy_Ring_Vs_Disk->Fill(Ring_Vs_Disk_xbin, ring);
      }
    }
  }

  int n_cppfdigi = 0;
  l1t::CPPFDigiCollection::const_iterator cppfdigiItr;
  for (cppfdigiItr = cppfdigis->begin(); cppfdigiItr != cppfdigis->end(); ++cppfdigiItr, ++n_cppfdigi) {
    const RPCDetId& detId = (*cppfdigiItr).rpcId();
    int region = (int)detId.region();

    int wheelOrDiskNumber;
    std::string wheelOrDiskType;

    RPCGeomServ geoServ(detId);
    std::string nameRoll = "";

    if (useRollInfo_)
      nameRoll = geoServ.name();
    else
      nameRoll = geoServ.chambername();

    int xBin, yBin;
    xBin = geoServ.segment();
    //yBin = (detId.ring()-1)*3-detId.roll()+1;
    //yBin = (detId.ring()-1)*3-3+1;  // no roll information
    yBin = detId.ring() - 1;
    int ring = detId.ring();

    wheelOrDiskType = "Disk";
    wheelOrDiskNumber = region * (int)detId.station();

    if (region != 0) {
      /*
      os.str("");
      os<<"CPPFOutput_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Ring_"<<ring<<"_board_Vs_Segment";
      if(meOutputDiskRing_board[os.str()]){
        meOutputDiskRing_board[os.str()]->Fill( xBin, (*cppfdigiItr).board()  );
      }

      os.str("");
      os<<"CPPFOutput_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Ring_"<<ring<<"_channel_Vs_Segment";
      if(meOutputDiskRing_channel[os.str()]){
        meOutputDiskRing_channel[os.str()]->Fill( xBin, (*cppfdigiItr).channel());
      }

      os.str("");
      os<<"CPPFOutput_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Ring_"<<ring<<"_emtf_sector_Vs_Segment";
      if(meOutputDiskRing_emtf_sector[os.str()]){
        meOutputDiskRing_emtf_sector[os.str()]->Fill(xBin, (*cppfdigiItr).emtf_sector() );
      }

      os.str("");
      os<<"CPPFOutput_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Ring_"<<ring<<"_emtf_link_Vs_Segment";
      if(meOutputDiskRing_emtf_link[os.str()]){
        meOutputDiskRing_emtf_link[os.str()]->Fill(xBin, (*cppfdigiItr).emtf_link() );
      }
      */
      os.str("");
      os << "CPPFOutput_" << wheelOrDiskType << "_" << wheelOrDiskNumber << "_Ring_" << ring << "_Theta_Vs_Segment";
      if (meOutputDiskRing_theta[os.str()]) {
        meOutputDiskRing_theta[os.str()]->Fill(xBin, (*cppfdigiItr).theta_int());
      }

      os.str("");
      os << "CPPFOutput_" << wheelOrDiskType << "_" << wheelOrDiskNumber << "_Ring_" << ring << "_Phi_Vs_Segment";
      if (meOutputDiskRing_phi[os.str()]) {
        meOutputDiskRing_phi[os.str()]->Fill(xBin, (*cppfdigiItr).phi_int());
      }

      if (wheelOrDiskNumber > 0)
        wheelOrDiskNumber--;
      CPPFOutputNormOccupDisk[wheelOrDiskNumber + 4]->Fill(xBin, yBin, 1);
    }

    int bx = (*cppfdigiItr).bx();

    int station = detId.station();

    int hist_index = histIndexCPPF.at({station, ring});
    if (region > 0)
      hist_index = 11 - hist_index;

    CPPFOutput_DiskRing_Vs_BX->Fill(bx, hist_index);
    CPPFOutput_Occupancy_DiskRing_Vs_Segment->Fill(geoServ.segment(), hist_index + 0.5);
    CPPFOutput_1DPhi[hist_index]->Fill((*cppfdigiItr).phi_int());
    CPPFOutput_1DTheta[hist_index]->Fill((*cppfdigiItr).theta_int());

    int Ring_Vs_Disk_xbin = wheelOrDiskNumber + 4 + 1;
    CPPFOutput_Occupancy_Ring_Vs_Disk->Fill(Ring_Vs_Disk_xbin, ring);
  }

  if (verbose_)
    cout << "L1TStage2CPPF: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events";
}
