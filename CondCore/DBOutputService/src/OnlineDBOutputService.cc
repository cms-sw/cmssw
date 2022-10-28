#include "CondCore/DBOutputService/interface/OnlineDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <curl/curl.h>
//

static size_t getHtmlCallback(void* contents, size_t size, size_t nmemb, void* ptr) {
  // Cast ptr to std::string pointer and append contents to that string
  ((std::string*)ptr)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

bool getInfoFromService(const std::string& urlString, std::string& info) {
  CURL* curl;
  CURLcode res;
  std::string htmlBuffer;
  char errbuf[CURL_ERROR_SIZE];

  curl = curl_easy_init();
  bool ret = false;
  if (curl) {
    struct curl_slist* chunk = nullptr;
    chunk = curl_slist_append(chunk, "content-type:document/plain");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, chunk);
    curl_easy_setopt(curl, CURLOPT_URL, urlString.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, getHtmlCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &htmlBuffer);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);
    res = curl_easy_perform(curl);
    if (CURLE_OK == res) {
      info = htmlBuffer;
      ret = true;
    } else {
      size_t len = strlen(errbuf);
      fprintf(stderr, "\nlibcurl: (%d) ", res);
      if (len)
        fprintf(stderr, "%s%s", errbuf, ((errbuf[len - 1] != '\n') ? "\n" : ""));
      else
        fprintf(stderr, "%s\n", curl_easy_strerror(res));
    }
    curl_easy_cleanup(curl);
  }
  return ret;
}

namespace cond {

  cond::Time_t getLatestLumiFromFile(const std::string& fileName) {
    cond::Time_t lastLumiProcessed = cond::time::MIN_VAL;
    std::ifstream lastLumiFile(fileName);
    if (lastLumiFile) {
      lastLumiFile >> lastLumiProcessed;
    } else {
      throw Exception(std::string("Can't access lastLumi file ") + fileName);
    }
    return lastLumiProcessed;
  }

  cond::Time_t getLastLumiFromOMS(const std::string& omsServiceUrl) {
    cond::Time_t lastLumiProcessed = cond::time::MIN_VAL;
    std::string info("");
    if (!getInfoFromService(omsServiceUrl, info))
      throw Exception("Can't get data from OMS Service.");
    std::istringstream sinfo(info);
    std::string srun;
    if (!std::getline(sinfo, srun, ',')) {
      throw Exception("Can't get run runmber info from OMS Service.");
    }
    std::string slumi;
    if (!std::getline(sinfo, slumi, ',')) {
      throw Exception("Can't get lumi id from OMS Service.");
    }
    unsigned int run = std::stoul(srun);
    unsigned int lumi = std::stoul(slumi);
    lastLumiProcessed = cond::time::lumiTime(run, lumi);
    return lastLumiProcessed;
  }

}  // namespace cond

cond::service::OnlineDBOutputService::OnlineDBOutputService(const edm::ParameterSet& iConfig,
                                                            edm::ActivityRegistry& iAR)
    : PoolDBOutputService(iConfig, iAR),
      m_runNumber(iConfig.getUntrackedParameter<unsigned long long>("runNumber", 1)),
      m_latencyInLumisections(iConfig.getUntrackedParameter<unsigned int>("latency", 1)),
      m_omsServiceUrl(iConfig.getUntrackedParameter<std::string>("omsServiceUrl", "")),
      m_preLoadConnectionString(iConfig.getUntrackedParameter<std::string>("preLoadConnectionString", "")),
      m_frontierKey(iConfig.getUntrackedParameter<std::string>("frontierKey", "")),
      m_debug(iConfig.getUntrackedParameter<bool>("debugLogging", false)) {
  if (m_omsServiceUrl.empty()) {
    m_lastLumiFile = iConfig.getUntrackedParameter<std::string>("lastLumiFile", "");
  }
}

cond::service::OnlineDBOutputService::~OnlineDBOutputService() {}

cond::Time_t cond::service::OnlineDBOutputService::getLastLumiProcessed() {
  cond::Time_t lastLumiProcessed = cond::time::MIN_VAL;
  std::string info("");
  if (!m_omsServiceUrl.empty()) {
    lastLumiProcessed = cond::getLastLumiFromOMS(m_omsServiceUrl);
    logger().logInfo() << "Last lumi: " << lastLumiProcessed
                       << " Current run: " << cond::time::unpack(lastLumiProcessed).first
                       << " lumi id:" << cond::time::unpack(lastLumiProcessed).second;
  } else {
    if (!m_lastLumiFile.empty()) {
      lastLumiProcessed = cond::getLatestLumiFromFile(m_lastLumiFile);
      auto upkTime = cond::time::unpack(lastLumiProcessed);
      logger().logInfo() << "Last lumi: " << lastLumiProcessed << " Current run: " << upkTime.first
                         << " lumi id:" << upkTime.second;
    } else {
      lastLumiProcessed = cond::time::lumiTime(m_runNumber, 1);
    }
  }
  return lastLumiProcessed;
}

cond::Iov_t cond::service::OnlineDBOutputService::preLoadIov(const PoolDBOutputService::Record& record,
                                                             cond::Time_t targetTime) {
  auto transId = cond::time::transactionIdForLumiTime(targetTime, record.m_refreshTime, m_frontierKey);
  cond::persistency::Session session = PoolDBOutputService::newReadOnlySession(m_preLoadConnectionString, transId);
  cond::persistency::TransactionScope transaction(session.transaction());
  transaction.start(true);
  cond::persistency::IOVProxy proxy = session.readIov(record.m_tag);
  auto iov = proxy.getInterval(targetTime);
  transaction.commit();
  return iov;
}
