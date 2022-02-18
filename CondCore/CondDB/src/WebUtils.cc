#include "CondCore/CondDB/interface/WebUtils.h"
//
#include <curl/curl.h>
#include <cstdio>
#include <cstring>

namespace cond {

  // callback to obtain the Get result
  static size_t getBodyCallback(void* contents, size_t size, size_t nmemb, void* ptr) {
    // Cast ptr to std::string pointer and append contents to that string
    ((std::string*)ptr)->append((char*)contents, size * nmemb);
    return size * nmemb;
  }

  unsigned long httpGet(const std::string& urlString, std::string& info) {
    CURL* curl;
    CURLcode res;
    std::string body;
    char errbuf[CURL_ERROR_SIZE];

    curl = curl_easy_init();
    unsigned long ret = false;
    if (curl) {
      struct curl_slist* chunk = nullptr;
      chunk = curl_slist_append(chunk, "content-type:application/json");
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, chunk);
      curl_easy_setopt(curl, CURLOPT_URL, urlString.c_str());
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, getBodyCallback);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
      curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);
      res = curl_easy_perform(curl);
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &ret);
      if (CURLE_OK == res) {
        info = body;
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

}  // namespace cond
