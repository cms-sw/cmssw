#include "L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFConfigOnlineProd.h"
#include <cstdio>
#include <string>

std::unique_ptr<L1MuCSCTFConfiguration> CSCTFConfigOnlineProd::newObject(const std::string& objectKey) {
  edm::LogInfo("L1-O2O: CSCTFConfigOnlineProd") << "Producing "
                                                << "L1MuCSCTFConfiguration "
                                                << "with key CSCTF_KEY=" << objectKey;

  std::string csctfreg[12];

  // loop over the 12 SPs forming the CSCTF crate
  for (int iSP = 1; iSP < 13; iSP++) {
    std::string spkey = objectKey + "00";
    if (iSP < 10)
      spkey += "0";
    spkey += std::to_string(iSP);

    //  SELECT Multiple columns  FROM TABLE with correct key:
    std::vector<std::string> columns;
    columns.push_back("STATIC_CONFIG");
    columns.push_back("ETA_CONFIG");
    columns.push_back("FIRMWARE");

    //SELECT * FROM CMS_CSC_TF.CSCTF_SP_CONF WHERE CSCTF_SP_CONF.SP_KEY = spkey
    l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
        columns, "CMS_CSC_TF", "CSCTF_SP_CONF", "CSCTF_SP_CONF.SP_KEY", m_omdsReader.singleAttribute(spkey));

    if (results.queryFailed())  // check if query was successful
    {
      edm::LogError("L1-O2O") << "Problem with L1CSCTFParameters key.";
      // return empty configuration
      return std::make_unique<L1MuCSCTFConfiguration>();
    }

    std::string conf_stat, conf_eta, conf_firmware;
    results.fillVariable("STATIC_CONFIG", conf_stat);
    results.fillVariable("ETA_CONFIG", conf_eta);
    results.fillVariable("FIRMWARE", conf_firmware);

    LogDebug("L1-O2O: CSCTFConfigOnlineProd:") << "conf_stat queried: " << conf_stat << "conf_eta queried:" << conf_eta
                                               << "conf_firmware queried:" << conf_firmware;

    for (size_t pos = conf_stat.find("\\n"); pos != std::string::npos; pos = conf_stat.find("\\n", pos)) {
      conf_stat[pos] = ' ';
      conf_stat[pos + 1] = '\n';
    }

    for (size_t pos = conf_eta.find("\\n"); pos != std::string::npos; pos = conf_eta.find("\\n", pos)) {
      conf_eta[pos] = ' ';
      conf_eta[pos + 1] = '\n';
    }

    for (size_t pos = conf_firmware.find("\\n"); pos != std::string::npos; pos = conf_firmware.find("\\n", pos)) {
      conf_firmware[pos] = ' ';
      conf_firmware[pos + 1] = '\n';
    }

    LogDebug("L1-O2O: CSCTFConfigOnlineProd") << "\nSP KEY: " << spkey << "\n\nSTATIC CONFIGURATION:\n"
                                              << conf_stat << "\nDAT_ETA CONFIGURATION:\n"
                                              << conf_eta << "\nFIRMWARE VERSIONS:\n"
                                              << conf_firmware;

    // The CSCTF firmware needs a bit more manipulation
    // The firmware is written in the DBS as SP SP day/month/year, where the real year is 2000+year
    // For easy handling when retrieving the configuration I prefer to write is as
    // FIRMWARE SP SP yearmonthday
    // e.g. SP SP 26/06/09 -> FIRMWARE SP SP 20090626

    std::string conf_firmware_sp;

    std::stringstream conf(conf_firmware);
    while (!conf.eof()) {
      char buff[1024];
      conf.getline(buff, 1024);
      std::stringstream line(buff);

      std::string register_ = "FIRMWARE";
      std::string chip_;
      line >> chip_;
      std::string muon_;
      line >> muon_;
      std::string writeValue_;
      line >> writeValue_;

      size_t pos;
      pos = writeValue_.find('/');

      std::string day;
      day.push_back(writeValue_[pos - 2]);
      day.push_back(writeValue_[pos - 1]);

      std::string month;
      month.push_back(writeValue_[pos + 1]);
      month.push_back(writeValue_[pos + 2]);

      std::string year("20");
      year.push_back(writeValue_[pos + 4]);
      year.push_back(writeValue_[pos + 5]);

      //std::cout << "day "   << day   <<std::endl;
      //std::cout << "month " << month <<std::endl;
      //std::cout << "year "  << year  <<std::endl;

      std::string date = year + month + day;
      // std::cout << register_  << " -- "
      // 	   << chip_      << " -- "
      // 	   << muon_      << " -- "
      // 	   << date       << " -- "
      // 	   << std::endl;

      // for the CSCTF emulator there is no need of the other firmware (CCB and MS)
      if (chip_ == "SP")
        conf_firmware_sp += register_ + " " + chip_ + " " + muon_ + " " + date + "\n";
    }

    edm::LogInfo("L1-O2O: CSCTFConfigOnlineProd") << "\nSP KEY: " << spkey << "\n\nSTATIC CONFIGURATION:\n"
                                                  << conf_stat << "\nDAT_ETA CONFIGURATION:\n"
                                                  << conf_eta << "\nFIRMWARE VERSIONS:\n"
                                                  << conf_firmware_sp;

    std::string conf_read = conf_eta + conf_stat + conf_firmware_sp;
    // write all registers for a given SP
    csctfreg[iSP - 1] = conf_read;
  }

  // return the final object with the configuration for all CSCTF
  return std::make_unique<L1MuCSCTFConfiguration>(csctfreg);
}
