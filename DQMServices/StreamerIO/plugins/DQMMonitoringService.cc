#include "DQMMonitoringService.h"

#include <boost/algorithm/string.hpp>

#include <ctime>

/*
 * This service is very similar to the FastMonitoringService in the HLT,
 * except that it is used for monitoring online DQM applications
 */

namespace dqmservices {

namespace fs = boost::filesystem;

#define MAX_LOG_SIZE 64*1024

DQMMonitoringService::DQMMonitoringService(const edm::ParameterSet &pset, edm::ActivityRegistry&) {
  json_path_ = pset.getUntrackedParameter<std::string>("jsonPath");

  char host[128];
  if (gethostname(host ,sizeof(host)) == -1) {
    throw cms::Exception("DQMMonitoringService")
          << "Internal error, cannot get host name";
  }

  hostname_ = host;
  fseq_ = 0;
  tag_ = "";
  nevents_ = 0;
  last_report_nevents_ = 0;
  last_report_time_ = std::chrono::high_resolution_clock::now();

  try {
    fillProcessInfoCmdline();
  } catch (...) {
    // pass
  }
}

DQMMonitoringService::~DQMMonitoringService() {
}

void DQMMonitoringService::registerExtra(std::string name, ptree data) {
  extra_.put_child(name, data);
}

void DQMMonitoringService::reportLumiSection(int run, int lumi) {
  try {
    reportLumiSectionUnsafe(run, lumi);
  } catch (...) {
    // pass
  }
}

void DQMMonitoringService::reportEvents(int nevts) {
  nevents_ += nevts;
}

void DQMMonitoringService::reportLumiSectionUnsafe(int run, int lumi) {
  using std::chrono::duration_cast;
  using std::chrono::milliseconds;

  int pid = getpid();
  ++fseq_;

  if (! fs::is_directory(json_path_)) {
    extra_.clear();
    return; // no directory present, quit
  }

  auto now = std::chrono::high_resolution_clock::now();

  // document unique id
  std::string id =
    str(boost::format("dqm-source-state-run%06d-host%s-pid%06d") % run % hostname_ % pid);

  // output jsn file
  std::string path_id;

  // check for debug fn
  if (fs::exists(json_path_ / ".debug")) {
    path_id = str(boost::format("%d.%08d+%s.jsn") % std::time(NULL) % fseq_ % id);
  } else {
    path_id = id + ".jsn";
  }

  std::string tmp_path = (json_path_ / (path_id + ".tmp")).string();
  std::string final_path = (json_path_ / path_id).string();

  float rate = (nevents_ - last_report_nevents_);
  rate = rate / duration_cast<milliseconds>(now - last_report_time_).count();
  rate = rate / 100;

  ptree pt;
  pt.put("_id", id);
  pt.put("pid", pid);
  pt.put("tag", tag_);
  pt.put("hostname", hostname_); 
  pt.put("sequence", fseq_);
  pt.put("type", "dqm-source-state");
  pt.put("run", run);
  pt.put("lumi", lumi);

  pt.put("events_total", nevents_);
  pt.put("events_rate", rate);

  // add some additional per-lumi information
  std::string log = hackoutTheStdErr();
  pt.put("stderr", log);

  fillProcessInfoStatus();

  // these are predefined
  pt.add_child("extra", extra_);
  pt.add_child("ps_info", ps_info_);

  std::ofstream file(tmp_path);
  write_json(file, pt, true);
  file.close();

  last_report_time_ = now; 
  last_report_nevents_ = nevents_;

  rename(tmp_path.c_str(), final_path.c_str());
}

void DQMMonitoringService::fillProcessInfoCmdline() {
  int fd = open("/proc/self/cmdline", O_RDONLY);
  ptree cmdline;

  if (fd != -1) {
    unsigned char buf[1024];
    int nbytesread = read(fd, buf, 1024);

    // make last character zero
    // in case we have read less than buf
    if (nbytesread > 0)
      buf[nbytesread-1] = 0;

    unsigned char *end = buf + nbytesread;
    for (unsigned char *p = buf; p < end; /**/) {
      std::string token((char *)p);
      ptree child;
      child.put("", token);
      cmdline.push_back(std::make_pair("", child));

      if ((tag_.size() == 0) &&
          (token.find(".py") != std::string::npos)) {

        // a hack to set the tag until we figure
        // out how to set it properly
        tag_ = token;
        boost::replace_last(tag_, ".py", "");
        boost::replace_last(tag_, "_cfg", "");

        size_t pos = tag_.rfind("/");
        if (pos != std::string::npos) {
          tag_ = tag_.substr(pos + 1);
        }
      }

      while (*p++); // skip until start of next 0-terminated section
    }
    close(fd);
  }

  ps_info_.put_child("cmdline", cmdline);
}

void DQMMonitoringService::fillProcessInfoStatus() {
  ptree data;

  std::ifstream in("/proc/self/status");
  std::string line;

  if (in) {
    while (std::getline(in, line)) {
      size_t pos = line.find(':');
      if (pos == std::string::npos)
        continue;

      std::string value = line.substr(pos+1);
      boost::trim(value); // value
      line.resize(pos); // key

      data.put(line, value);
    }

    in.close();
  }

  ps_info_.put_child("status", data);
}

std::string DQMMonitoringService::hackoutTheStdErr() {
  // magic
  char buf[MAX_LOG_SIZE + 1];
  ssize_t ret = readlink("/proc/self/fd/2", buf, MAX_LOG_SIZE); 
  if (ret > 0) {
    buf[ret] = 0;
  } else {
    return "error: can't read the stderr link.";
  }

  if (strstr(buf, "/dev/") != NULL) {
    // can't read this weird file
    return "error: stderr is a special file.";
  }

  // try to open
  FILE *sr = fopen(buf , "rb");
  if (sr == NULL)
    return "error: can't open the stderr (deleted?).";

  // try to get the last position
  // if this is an ordinary it will succeed
  fseek(sr, 0, SEEK_END);
  long size = ftell(sr);
  if (size > 0) {
    long from = size - (MAX_LOG_SIZE);
    if (from < 0)
      from = 0;

    fseek(sr, from, SEEK_SET);
    size_t read = fread(buf, 1, MAX_LOG_SIZE, sr);
    buf[read] = 0;

    // If "from" was not zero, discard the first line.
    // Since it will be corrupted anyway.
    char *start = buf;

    if (from != 0) {
      start = strchr(start, '\n');
      if (start == NULL) {
        // should not happen
        // return an empty string
        start = buf + read;
      } else {
        start = start + 1;
      }
    }

    return std::string(start);
  }

  fclose(sr);
  return "error: stderr is not a seek-able file.";
}

} // end-of-namespace

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using dqmservices::DQMMonitoringService;
DEFINE_FWK_SERVICE(DQMMonitoringService);
