#include "CondCore/DBOutputService/interface/OnlineDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <curl/curl.h>
//

cond::service::OnlineDBOutputService::OnlineDBOutputService(const edm::ParameterSet& iConfig,
                                                            edm::ActivityRegistry& iAR)
    : PoolDBOutputService(iConfig, iAR),
      m_runNumber(iConfig.getUntrackedParameter<unsigned long long>("runNumber", 0)),
      m_latencyInLumisections(iConfig.getUntrackedParameter<unsigned int>("latency", 1)),
      m_lastLumiUrl(iConfig.getUntrackedParameter<std::string>("lastLumiUrl", "")),
      m_preLoadConnectionString(iConfig.getUntrackedParameter<std::string>("preLoadConnectionString", "")),
      m_debug(iConfig.getUntrackedParameter<bool>("debugLogging", false)) {
  if (!m_lastLumiUrl.empty()) {
    startTransaction();
    auto lastRun = PoolDBOutputService::session().getLastRun();
    if (lastRun.isOnGoing()) {
      m_runNumber = lastRun.run;
    }
  } else {
    m_lastLumiFile = iConfig.getUntrackedParameter<std::string>("lastLumiFile", "");
  }
}

cond::service::OnlineDBOutputService::~OnlineDBOutputService() {}

static size_t getHtmlCallback(void* contents, size_t size, size_t nmemb, void* ptr) {
  // Cast ptr to std::string pointer and append contents to that string
  ((std::string*)ptr)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

bool getLatestLumiFromDAQ(const std::string& urlString, std::string& info) {
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

cond::Time_t cond::service::OnlineDBOutputService::getLastLumiProcessed() {
  cond::Time_t lastLumiProcessed = cond::time::MIN_VAL;
  unsigned int lastL = 0;
  if (!m_lastLumiUrl.empty()) {
    std::string info("");
    if (!getLatestLumiFromDAQ(m_lastLumiUrl, info))
      throw Exception("Can't get last Lumisection from DAQ.");
    lastL = boost::lexical_cast<unsigned int>(info);
    lastLumiProcessed = cond::time::lumiTime(m_runNumber, lastL);
    edm::LogInfo(MSGSOURCE) << "Last lumi: " << lastLumiProcessed << " Current run: " << m_runNumber
                            << " lumi id:" << lastL;
  } else {
    if (m_lastLumiFile.empty()) {
      //auto t1 = std::chrono::steady_clock::now();
      //auto deltat = std::chrono::duration_cast<std::chrono::seconds>( t1 - m_startRunTime ).count();
      //lastL = (unsigned int)(deltat/23);
      //lastLumiProcessed = cond::time::lumiTime( m_runNumber, lastL );
      //edm::LogInfo( MSGSOURCE ) << "Last lumi: "<<lastLumiProcessed<<" Current run: "<<m_runNumber<<" lumi id:"<<lastL;
      throw Exception("File name for last lumi has not been provided.");
    } else {
      lastLumiProcessed = cond::getLatestLumiFromFile(m_lastLumiFile);
      auto upkTime = cond::time::unpack(lastLumiProcessed);
      edm::LogInfo(MSGSOURCE) << "Last lumi: " << lastLumiProcessed << " Current run: " << upkTime.first
                              << " lumi id:" << upkTime.second;
    }
  }
  return lastLumiProcessed;
}

cond::Iov_t cond::service::OnlineDBOutputService::preLoadIov(const std::string& recordName, cond::Time_t targetTime) {
  cond::persistency::Session session = getReadOnlyCache(targetTime);
  cond::persistency::TransactionScope transaction(session.transaction());
  transaction.start(true);
  cond::persistency::IOVProxy proxy = session.readIov(PoolDBOutputService::tag(recordName));
  auto iov = proxy.getInterval(targetTime);
  transaction.commit();
  return iov;
}

cond::persistency::Session cond::service::OnlineDBOutputService::getReadOnlyCache(cond::Time_t targetTime) {
  return PoolDBOutputService::newReadOnlySession(m_preLoadConnectionString, std::to_string(targetTime));
}
