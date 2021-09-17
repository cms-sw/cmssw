
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripDbParams::SiStripDbParams()
    : usingDb_(false),
      confdb_(null_),
      user_(null_),
      passwd_(null_),
      path_(null_),
      usingDbCache_(false),
      sharedMemory_(""),
      tnsAdmin_(""),
      partitions_(),
      outputModuleXml_("/tmp/module.xml"),
      outputDcuInfoXml_("/tmp/dcuinfo.xml"),
      outputFecXml_("/tmp/fec.xml"),
      outputFedXml_("/tmp/fed.xml") {
  ;
}

// -----------------------------------------------------------------------------
//
SiStripDbParams::SiStripDbParams(const SiStripDbParams& input)
    : usingDb_(input.usingDb()),
      confdb_(input.confdb()),
      user_(input.user()),
      passwd_(input.passwd()),
      path_(input.path()),
      usingDbCache_(input.usingDbCache()),
      sharedMemory_(input.sharedMemory()),
      tnsAdmin_(input.tnsAdmin()),
      partitions_(input.partitions().begin(), input.partitions().end()),
      outputModuleXml_(input.outputModuleXml()),
      outputDcuInfoXml_(input.outputDcuInfoXml()),
      outputFecXml_(input.outputFecXml()),
      outputFedXml_(input.outputFedXml()) {
  ;
}

// -----------------------------------------------------------------------------
//
SiStripDbParams& SiStripDbParams::operator=(const SiStripDbParams& input) {
  usingDb_ = input.usingDb();
  confdb_ = input.confdb();
  user_ = input.user();
  passwd_ = input.passwd();
  path_ = input.path();
  confdb(confdb_);
  usingDbCache_ = input.usingDbCache();
  sharedMemory_ = input.sharedMemory();
  tnsAdmin_ = input.tnsAdmin();
  partitions_.clear();
  partitions_ = SiStripPartitions(input.partitions().begin(), input.partitions().end());
  outputModuleXml_ = input.outputModuleXml();
  outputDcuInfoXml_ = input.outputDcuInfoXml();
  outputFecXml_ = input.outputFecXml();
  outputFedXml_ = input.outputFedXml();
  return *this;
}

// -----------------------------------------------------------------------------
//
bool SiStripDbParams::operator==(const SiStripDbParams& input) const {
  if (!(usingDb_ == input.usingDb() && confdb_ == input.confdb() && user_ == input.user() &&
        passwd_ == input.passwd() && path_ == input.path() && usingDbCache_ == input.usingDbCache() &&
        sharedMemory_ == input.sharedMemory() && tnsAdmin_ == input.tnsAdmin() &&
        outputModuleXml_ == input.outputModuleXml() && outputDcuInfoXml_ == input.outputDcuInfoXml() &&
        outputFecXml_ == input.outputFecXml() && outputFedXml_ == input.outputFedXml())) {
    return false;
  }
  if (partitionsSize() != input.partitionsSize()) {
    return false;
  }
  //   SiStripPartitions::const_iterator ii = input.partitions().begin();
  //   SiStripPartitions::const_iterator jj = input.partitions().end();
  //   SiStripPartitions::const_iterator iter = partitions_.begin();
  //   for ( ; ii != jj; ++ii ) {
  //     if ( ii->first != iter->first || ii->second != iter->second ) { return false; }
  //     iter++;
  //   }
  //   return true;
  return partitions_ == input.partitions_;
}

// -----------------------------------------------------------------------------
//
bool SiStripDbParams::operator!=(const SiStripDbParams& input) const { return !(*this == input); }

// -----------------------------------------------------------------------------
//
SiStripDbParams::~SiStripDbParams() { reset(); }

// -----------------------------------------------------------------------------
//
void SiStripDbParams::reset() {
  usingDb_ = false;
  confdb_ = null_;
  user_ = null_;
  passwd_ = null_;
  path_ = null_;
  usingDbCache_ = false;
  sharedMemory_ = "";
  tnsAdmin_ = "";
  partitions_.clear();
  confdb(confdb_);
  outputModuleXml_ = "/tmp/module.xml";
  outputDcuInfoXml_ = "/tmp/dcuinfo.xml";
  outputFecXml_ = "/tmp/fec.xml";
  outputFedXml_ = "/tmp/fed.xml";
}

// -----------------------------------------------------------------------------
//
void SiStripDbParams::addPartition(const SiStripPartition& in) {
  if (in.partitionName().empty()) {
    std::stringstream ss;
    ss << "[SiStripDbParams::" << __func__ << "]"
       << " Attempting to add partition with null name!";
    //edm::LogWarning(mlConfigDb_) << ss.str();
  }

  SiStripPartitions::const_iterator iter = partitions_.find(in.partitionName());
  if (iter == partitions_.end()) {
    partitions_[in.partitionName()] = in;
    if (in.partitionName() != SiStripPartition::defaultPartitionName_) {
      std::stringstream ss;
      ss << "[SiStripDbParams::" << __func__ << "]"
         << " Added new partition with name \"" << in.partitionName() << "\"";
      //ss << std::endl << partitions_[in.partitionName()];
      ss << " (Currently have " << partitions_.size() << " partitions in cache...)";
      LogTrace(mlConfigDb_) << ss.str();
    }
  } else {
    std::stringstream ss;
    ss << "[SiStripDbParams::" << __func__ << "]"
       << " Partition with name \"" << in.partitionName() << "\" already found!"
       << " Not adding to cache...";
    edm::LogWarning(mlConfigDb_) << ss.str();
  }
}

// -----------------------------------------------------------------------------
//
void SiStripDbParams::pset(const edm::ParameterSet& cfg) {
  // "Common" configurables
  usingDb_ = cfg.getUntrackedParameter<bool>("UsingDb", false);
  usingDbCache_ = cfg.getUntrackedParameter<bool>("UsingDbCache", false);
  sharedMemory_ = cfg.getUntrackedParameter<std::string>("SharedMemory", "");
  tnsAdmin_ = cfg.getUntrackedParameter<std::string>("TNS_ADMIN", "");
  confdb(cfg.getUntrackedParameter<std::string>("ConfDb", ""));

  // Check if top-level PSet (containing partition-level Psets) exists
  std::string partitions = "Partitions";
  std::vector<std::string> str = cfg.getParameterNamesForType<edm::ParameterSet>(false);
  std::vector<std::string>::const_iterator istr = std::find(str.begin(), str.end(), partitions);
  if (istr != str.end()) {
    // Retrieve top-level PSet containing partition-level Psets
    edm::ParameterSet psets = cfg.getUntrackedParameter<edm::ParameterSet>(partitions);

    // Extract names of partition-level PSets
    std::vector<std::string> names = psets.getParameterNamesForType<edm::ParameterSet>(false);

    // Iterator through PSets names, retrieve PSet for each partition and extract info
    std::vector<std::string>::iterator iname = names.begin();
    std::vector<std::string>::iterator jname = names.end();
    for (; iname != jname; ++iname) {
      edm::ParameterSet pset = psets.getUntrackedParameter<edm::ParameterSet>(*iname);
      SiStripPartition tmp;
      tmp.pset(pset);
      addPartition(tmp);
    }
  }

  // Ensure at least one "default" partition
  if (partitions_.empty()) {
    addPartition(SiStripPartition());
  }

  // Set output XML files
  outputModuleXml_ = cfg.getUntrackedParameter<std::string>("OutputModuleXml", "/tmp/module.xml");
  outputDcuInfoXml_ = cfg.getUntrackedParameter<std::string>("OutputDcuInfoXml", "/tmp/dcuinfo.xml");
  outputFecXml_ = cfg.getUntrackedParameter<std::string>("OutputFecXml", "/tmp/fec.xml");
  outputFedXml_ = cfg.getUntrackedParameter<std::string>("OutputFedXml", "/tmp/fed.xml");
}

// -----------------------------------------------------------------------------
//
void SiStripDbParams::confdb(const std::string& confdb) {
  confdb_ = confdb;
  size_t ipass = confdb.find('/');
  size_t ipath = confdb.find('@');
  if (ipass != std::string::npos && ipath != std::string::npos) {
    user_ = confdb.substr(0, ipass);
    passwd_ = confdb.substr(ipass + 1, ipath - ipass - 1);
    path_ = confdb.substr(ipath + 1, confdb.size());
  } else {
    user_ = null_;
    passwd_ = null_;
    path_ = null_;
  }
}

// -----------------------------------------------------------------------------
//
void SiStripDbParams::confdb(const std::string& user, const std::string& passwd, const std::string& path) {
  if (!user.empty() && !passwd.empty() && !path.empty() && user != null_ && passwd != null_ && path != null_) {
    user_ = user;
    passwd_ = passwd;
    path_ = path;
  } else {
    user_ = null_;
    passwd_ = null_;
    path_ = null_;
  }
  confdb_ = user_ + "/" + passwd_ + "@" + path_;
}

// -----------------------------------------------------------------------------
//
SiStripDbParams::SiStripPartitions::const_iterator SiStripDbParams::partition(std::string partition_name) const {
  SiStripDbParams::SiStripPartitions::const_iterator ii = partitions().begin();
  SiStripDbParams::SiStripPartitions::const_iterator jj = partitions().end();
  for (; ii != jj; ++ii) {
    if (partition_name == ii->second.partitionName()) {
      return ii;
    }
  }
  return partitions().end();
}

// -----------------------------------------------------------------------------
//
SiStripDbParams::SiStripPartitions::iterator SiStripDbParams::partition(std::string partition_name) {
  SiStripDbParams::SiStripPartitions::iterator ii = partitions().begin();
  SiStripDbParams::SiStripPartitions::iterator jj = partitions().end();
  for (; ii != jj; ++ii) {
    if (partition_name == ii->second.partitionName()) {
      return ii;
    }
  }
  return partitions().end();
}

// -----------------------------------------------------------------------------
//
std::vector<std::string> SiStripDbParams::partitionNames() const {
  std::vector<std::string> partitions;
  SiStripPartitions::const_iterator ii = partitions_.begin();
  SiStripPartitions::const_iterator jj = partitions_.end();
  for (; ii != jj; ++ii) {
    if (std::find(partitions.begin(), partitions.end(), ii->second.partitionName()) == partitions.end()) {
      if (!ii->second.partitionName().empty()) {
        partitions.push_back(ii->second.partitionName());
      }
    } else {
      edm::LogWarning(mlConfigDb_) << "[SiStripConfigDb::" << __func__ << "]"
                                   << " Partition " << ii->second.partitionName()
                                   << " already found! Not adding to vector...";
    }
  }
  return partitions;
}

// -----------------------------------------------------------------------------
//
std::vector<std::string> SiStripDbParams::partitionNames(std::string input) const {
  std::istringstream ss(input);
  std::vector<std::string> partitions;
  std::string delimiter = ":";
  std::string token;
  while (getline(ss, token, ':')) {
    if (!token.empty()) {
      partitions.push_back(token);
    }
  }
  return partitions;
}

// -----------------------------------------------------------------------------
//
std::string SiStripDbParams::partitionNames(const std::vector<std::string>& partitions) const {
  std::stringstream ss;
  std::vector<std::string>::const_iterator ii = partitions.begin();
  std::vector<std::string>::const_iterator jj = partitions.end();
  bool first = true;
  for (; ii != jj; ++ii) {
    if (!ii->empty()) {
      first ? ss << *ii : ss << ":" << *ii;
      first = false;
    }
  }
  return ss.str();
}

// -----------------------------------------------------------------------------
//
void SiStripDbParams::print(std::stringstream& ss) const {
  ss << " Using database account     : " << std::boolalpha << usingDb_ << std::noboolalpha << std::endl;
  ss << " Using XML files            : " << std::boolalpha << !usingDb_ << std::noboolalpha << std::endl;
  ss << " Using database cache       : " << std::boolalpha << usingDbCache_ << std::noboolalpha << std::endl;
  if (usingDbCache_) {
    ss << " Shared memory name         : " << std::boolalpha << sharedMemory_ << std::noboolalpha << std::endl;
  }

  if (!usingDbCache_) {
    if (usingDb_) {
      ss << " Database account (ConfDb)  : " << user_ + "/******@" + path_ << std::endl;
    }

    ss << " Number of partitions       : " << partitions_.size();
    if (partitions_.empty()) {
      if (!usingDbCache_) {
        ss << " (Empty!)";
      } else {
        ss << " (Using database cache!)";
      }
    }
    ss << std::endl;

    uint16_t cntr = 0;
    SiStripPartitions::const_iterator ii = partitions_.begin();
    SiStripPartitions::const_iterator jj = partitions_.end();
    for (; ii != jj; ++ii) {
      cntr++;
      ss << " Partition #" << cntr << " (out of " << partitions_.size() << "):" << std::endl;
      ii->second.print(ss, usingDb_);
    }

    if (!usingDb_) {
      ss << " Output \"module.xml\" file   : " << outputModuleXml_ << std::endl
         << " Output \"dcuinfo.xml\" file  : " << outputDcuInfoXml_ << std::endl
         << " Output \"fec.xml\" file(s)   : " << outputFecXml_ << std::endl
         << " Output \"fed.xml\" file(s)   : " << outputFedXml_ << std::endl;
    }
  }
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<<(std::ostream& os, const SiStripDbParams& params) {
  std::stringstream ss;
  params.print(ss);
  os << ss.str();
  return os;
}

// -----------------------------------------------------------------------------
//
std::vector<std::string> SiStripDbParams::inputModuleXmlFiles() const {
  std::vector<std::string> files;
  SiStripPartitions::const_iterator ii = partitions_.begin();
  SiStripPartitions::const_iterator jj = partitions_.end();
  for (; ii != jj; ++ii) {
    files.insert(files.end(), ii->second.inputModuleXml());
  }
  return files;
}

// -----------------------------------------------------------------------------
//
std::vector<std::string> SiStripDbParams::inputDcuInfoXmlFiles() const {
  std::vector<std::string> files;
  SiStripPartitions::const_iterator ii = partitions_.begin();
  SiStripPartitions::const_iterator jj = partitions_.end();
  for (; ii != jj; ++ii) {
    files.insert(files.end(), ii->second.inputDcuInfoXml());
  }
  return files;
}

// -----------------------------------------------------------------------------
//
std::vector<std::string> SiStripDbParams::inputFecXmlFiles() const {
  std::vector<std::string> files;
  SiStripPartitions::const_iterator ii = partitions_.begin();
  SiStripPartitions::const_iterator jj = partitions_.end();
  for (; ii != jj; ++ii) {
    files.insert(files.end(), ii->second.inputFecXml().begin(), ii->second.inputFecXml().end());
  }
  return files;
}

// -----------------------------------------------------------------------------
//
std::vector<std::string> SiStripDbParams::inputFedXmlFiles() const {
  std::vector<std::string> files;
  SiStripPartitions::const_iterator ii = partitions_.begin();
  SiStripPartitions::const_iterator jj = partitions_.end();
  for (; ii != jj; ++ii) {
    files.insert(files.end(), ii->second.inputFedXml().begin(), ii->second.inputFedXml().end());
  }
  return files;
}
