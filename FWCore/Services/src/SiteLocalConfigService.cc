///////////////////////////////////////
//
// data catalogs are filled in "parse"
//
///////////////////////////////////////

//<<<<<< INCLUDES                                                       >>>>>>

#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "tinyxml2.h"
#include <sstream>
#include <memory>
#include <boost/algorithm/string.hpp>
//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

namespace {

  // concatenate all the XML node attribute/value pairs into a
  // paren-separated string (for use by CORAL and frontier_client)
  inline std::string _toParenString(tinyxml2::XMLElement const &nodeToConvert) {
    std::ostringstream oss;

    for (auto child = nodeToConvert.FirstChildElement(); child != nullptr; child = child->NextSiblingElement()) {
      for (auto attribute = child->FirstAttribute(); attribute != nullptr; attribute = attribute->Next()) {
        oss << "(" << child->Name() << attribute->Name() << "=" << attribute->Value() << ")";
      }
    }
    return oss.str();
  }

  template <typename T>
  static void overrideFromPSet(char const *iName, edm::ParameterSet const &iPSet, T &iHolder, T const *&iPointer) {
    if (iPSet.exists(iName)) {
      iHolder = iPSet.getUntrackedParameter<T>(iName);
      iPointer = &iHolder;
    }
  }

  constexpr char const *const kEmptyString = "";
  const char *safe(const char *iCheck) {
    if (iCheck == nullptr) {
      return kEmptyString;
    }
    return iCheck;
  }

  std::string defaultURL() {
    std::string returnValue;
    const char *tmp = std::getenv("SITECONFIG_PATH");
    if (tmp) {
      returnValue = tmp;
    }
    returnValue += "/JobConfig/site-local-config.xml";
    return returnValue;
  }

}  // namespace

namespace edm {
  namespace service {

    const std::string SiteLocalConfigService::m_statisticsDefaultPort = "3334";

    SiteLocalConfigService::SiteLocalConfigService(ParameterSet const &pset)
        : m_url(pset.getUntrackedParameter<std::string>("siteLocalConfigFileUrl", defaultURL())),
          m_trivialDataCatalogs(),
          m_dataCatalogs(),
          m_frontierConnect(),
          m_rfioType("castor"),
          m_connected(false),
          m_cacheTempDir(),
          m_cacheTempDirPtr(nullptr),
          m_cacheMinFree(),
          m_cacheMinFreePtr(nullptr),
          m_cacheHint(),
          m_cacheHintPtr(nullptr),
          m_cloneCacheHint(),
          m_cloneCacheHintPtr(nullptr),
          m_readHint(),
          m_readHintPtr(nullptr),
          m_ttreeCacheSize(0U),
          m_ttreeCacheSizePtr(nullptr),
          m_timeout(0U),
          m_timeoutPtr(nullptr),
          m_debugLevel(0U),
          m_enablePrefetching(false),
          m_enablePrefetchingPtr(nullptr),
          m_nativeProtocols(),
          m_nativeProtocolsPtr(nullptr),
          m_statisticsDestination(),
          m_statisticsAddrInfo(nullptr),
          m_statisticsInfoAvail(false),
          m_siteName(),
          m_subSiteName() {
      this->parse(m_url);

      //apply overrides
      overrideFromPSet("overrideSourceCacheTempDir", pset, m_cacheTempDir, m_cacheTempDirPtr);
      overrideFromPSet("overrideSourceCacheMinFree", pset, m_cacheMinFree, m_cacheMinFreePtr);
      overrideFromPSet("overrideSourceCacheHintDir", pset, m_cacheHint, m_cacheHintPtr);
      overrideFromPSet("overrideSourceCloneCacheHintDir", pset, m_cloneCacheHint, m_cloneCacheHintPtr);
      overrideFromPSet("overrideSourceReadHint", pset, m_readHint, m_readHintPtr);
      overrideFromPSet("overrideSourceNativeProtocols", pset, m_nativeProtocols, m_nativeProtocolsPtr);
      overrideFromPSet("overrideSourceTTreeCacheSize", pset, m_ttreeCacheSize, m_ttreeCacheSizePtr);
      overrideFromPSet("overrideSourceTimeout", pset, m_timeout, m_timeoutPtr);
      overrideFromPSet("overridePrefetching", pset, m_enablePrefetching, m_enablePrefetchingPtr);
      const std::string *tmpStringPtr = nullptr;
      overrideFromPSet("overrideStatisticsDestination", pset, m_statisticsDestination, tmpStringPtr);
      this->computeStatisticsDestination();
      std::vector<std::string> tmpStatisticsInfo;
      std::vector<std::string> const *tmpStatisticsInfoPtr = nullptr;
      overrideFromPSet("overrideStatisticsInfo", pset, tmpStatisticsInfo, tmpStatisticsInfoPtr);
      if (tmpStatisticsInfoPtr) {
        m_statisticsInfoAvail = true;
        m_statisticsInfo.clear();
        for (auto &entry : tmpStatisticsInfo) {
          m_statisticsInfo.insert(std::move(entry));
        }
      }

      if (pset.exists("debugLevel")) {
        m_debugLevel = pset.getUntrackedParameter<unsigned int>("debugLevel");
      }
      if (pset.exists("overrideUseLocalConnectString")) {
        m_useLocalConnectString = pset.getUntrackedParameter<bool>("overrideUseLocalConnectString");
      }
      if (pset.exists("overrideLocalConnectPrefix")) {
        m_localConnectPrefix = pset.getUntrackedParameter<std::string>("overrideLocalConnectPrefix");
      }
      if (pset.exists("overrideLocalConnectSuffix")) {
        m_localConnectSuffix = pset.getUntrackedParameter<std::string>("overrideLocalConnectSuffix");
      }
    }

    SiteLocalConfigService::~SiteLocalConfigService() {
      if (m_statisticsAddrInfo) {
        freeaddrinfo(m_statisticsAddrInfo);
        m_statisticsAddrInfo = nullptr;
      }
    }

    std::vector<std::string> const &SiteLocalConfigService::trivialDataCatalogs() const {
      if (!m_connected) {
        static std::vector<std::string> const tmp{"file:PoolFileCatalog.xml"};
        return tmp;
      }

      if (m_trivialDataCatalogs.empty()) {
        cms::Exception ex("SiteLocalConfigService");
        ex << "Did not find catalogs in event-data section in " << m_url;
        ex.addContext("edm::SiteLocalConfigService::trivialDataCatalogs()");
        throw ex;
      }

      return m_trivialDataCatalogs;
    }

    std::vector<edm::CatalogAttributes> const &SiteLocalConfigService::dataCatalogs() const {
      if (!m_connected) {
        cms::Exception ex("SiteLocalConfigService");
        ex << "Incomplete configuration. Valid site-local-config not found at " << m_url;
        ex.addContext("edm::SiteLocalConfigService::dataCatalogs()");
        throw ex;
      }
      if (m_dataCatalogs.empty()) {
        cms::Exception ex("SiteLocalConfigService");
        ex << "Did not find catalogs in data-access section in " << m_url;
        ex.addContext("edm::SiteLocalConfigService::dataCatalogs()");
        throw ex;
      }
      return m_dataCatalogs;
    }

    std::filesystem::path const SiteLocalConfigService::storageDescriptionPath(
        edm::CatalogAttributes const &aDataCatalog) const {
      std::string siteconfig_path = std::string(std::getenv("SITECONFIG_PATH"));
      std::filesystem::path filename_storage;
      //not a cross site use local path given in SITECONFIG_PATH
      if (aDataCatalog.site == aDataCatalog.storageSite) {
        //it is a site (no defined subSite), use local path given in SITECONFIG_PATH
        if (aDataCatalog.subSite.empty())
          filename_storage = siteconfig_path;
        //it is a subsite, move one level up
        else
          filename_storage = siteconfig_path + "/..";
      } else {  //cross site
        //it is a site (no defined subSite), move one level up
        if (aDataCatalog.subSite.empty())
          filename_storage = siteconfig_path + "/../" + aDataCatalog.storageSite;
        //it is a subsite, move two levels up
        else
          filename_storage = siteconfig_path + "/../../" + aDataCatalog.storageSite;
      }
      filename_storage /= "storage.json";
      try {
        filename_storage = std::filesystem::canonical(filename_storage);
      } catch (std::exception &e) {
        cms::Exception ex("SiteLocalConfigService");
        ex << "Fail to convert path to the storage description, " << filename_storage.string()
           << ", to the canonical absolute path"
           << ". Path exists?";
        ex.addContext("edm::SiteLocalConfigService::storageDescriptionPath()");
        throw ex;
      }
      return filename_storage;
    }

    std::string const SiteLocalConfigService::frontierConnect(std::string const &servlet) const {
      if (!m_connected) {
        throw cms::Exception("Incomplete configuration") << "Valid site-local-config not found at " << m_url;
      }

      if (m_frontierConnect.empty()) {
        throw cms::Exception("Incomplete configuration")
            << "Did not find frontier-connect in calib-data section in " << m_url;
      }

      if (servlet.empty()) {
        return m_frontierConnect;
      }

      // Replace the last component of every "serverurl=" piece (up to the
      //   next close-paren) with the servlet
      std::string::size_type nextparen = 0;
      std::string::size_type serverurl, lastslash;
      std::string complexstr = "";
      while ((serverurl = m_frontierConnect.find("(serverurl=", nextparen)) != std::string::npos) {
        complexstr.append(m_frontierConnect, nextparen, serverurl - nextparen);
        nextparen = m_frontierConnect.find(')', serverurl);
        lastslash = m_frontierConnect.rfind('/', nextparen);
        complexstr.append(m_frontierConnect, serverurl, lastslash - serverurl + 1);
        complexstr.append(servlet);
      }
      complexstr.append(m_frontierConnect, nextparen, m_frontierConnect.length() - nextparen);

      return complexstr;
    }

    std::string const SiteLocalConfigService::lookupCalibConnect(std::string const &input) const {
      static std::string const proto = "frontier://";

      if (input.substr(0, proto.length()) == proto) {
        // Replace the part after the frontier:// and before either an open-
        //  parentheses (which indicates user-supplied options) or the last
        //  slash (which indicates start of the schema) with the complex
        //  parenthesized string returned from frontierConnect() (which
        //  contains all the information needed to connect to frontier),
        //  if that part is a simple servlet name (non-empty and not
        //  containing special characters)
        // Example connect strings where servlet is replaced:
        //  frontier://cms_conditions_data/CMS_COND_ECAL
        //  frontier://FrontierInt/CMS_COND_ECAL
        //  frontier://FrontierInt(retrieve-ziplevel=0)/CMS_COND_ECAL
        // Example connect strings left untouched:
        //  frontier://cmsfrontier.cern.ch:8000/FrontierInt/CMS_COND_ECAL
        //  frontier://(serverurl=cmsfrontier.cern.ch:8000/FrontierInt)/CMS_COND_ECAL
        std::string::size_type startservlet = proto.length();
        // if user supplied extra parenthesized options, stop servlet there
        std::string::size_type endservlet = input.find('(', startservlet);
        if (endservlet == std::string::npos) {
          endservlet = input.rfind('/', input.length());
        }
        std::string servlet = input.substr(startservlet, endservlet - startservlet);
        if ((!servlet.empty()) && (servlet.find_first_of(":/)[]") == std::string::npos)) {
          if (servlet == "cms_conditions_data") {
            // use the default servlet from site-local-config.xml
            servlet = "";
          }
          return proto + frontierConnect(servlet) + input.substr(endservlet);
        }
      }
      return input;
    }

    std::string const SiteLocalConfigService::rfioType(void) const { return m_rfioType; }

    std::string const *SiteLocalConfigService::sourceCacheTempDir() const { return m_cacheTempDirPtr; }

    double const *SiteLocalConfigService::sourceCacheMinFree() const { return m_cacheMinFreePtr; }

    std::string const *SiteLocalConfigService::sourceCacheHint() const { return m_cacheHintPtr; }

    std::string const *SiteLocalConfigService::sourceCloneCacheHint() const { return m_cloneCacheHintPtr; }

    std::string const *SiteLocalConfigService::sourceReadHint() const { return m_readHintPtr; }

    unsigned int const *SiteLocalConfigService::sourceTTreeCacheSize() const { return m_ttreeCacheSizePtr; }

    unsigned int const *SiteLocalConfigService::sourceTimeout() const { return m_timeoutPtr; }

    bool SiteLocalConfigService::enablePrefetching() const {
      return m_enablePrefetchingPtr ? *m_enablePrefetchingPtr : false;
    }

    unsigned int SiteLocalConfigService::debugLevel() const { return m_debugLevel; }

    std::vector<std::string> const *SiteLocalConfigService::sourceNativeProtocols() const {
      return m_nativeProtocolsPtr;
    }

    struct addrinfo const *SiteLocalConfigService::statisticsDestination() const {
      return m_statisticsAddrInfo;
    }

    std::set<std::string> const *SiteLocalConfigService::statisticsInfo() const {
      return m_statisticsInfoAvail ? &m_statisticsInfo : nullptr;
    }

    std::string const &SiteLocalConfigService::siteName() const { return m_siteName; }
    std::string const &SiteLocalConfigService::subSiteName() const { return m_subSiteName; }
    bool SiteLocalConfigService::useLocalConnectString() const { return m_useLocalConnectString; }
    std::string const &SiteLocalConfigService::localConnectPrefix() const { return m_localConnectPrefix; }
    std::string const &SiteLocalConfigService::localConnectSuffix() const { return m_localConnectSuffix; }

    void SiteLocalConfigService::getCatalog(tinyxml2::XMLElement const &cat, std::string site, std::string subSite) {
      edm::CatalogAttributes aCatalog;
      aCatalog.site = site;
      aCatalog.subSite = subSite;
      auto tmp_site = std::string(safe(cat.Attribute("site")));
      //no site attribute in the data catalog defined in <data-access>, so storage site is from <site> block of site_local_config.xml, which is the input parameter "site" of this method
      if (tmp_site.empty())
        aCatalog.storageSite = site;
      //now storage site is explicitly defined in <data-access>
      else
        aCatalog.storageSite = tmp_site;
      aCatalog.volume = std::string(safe(cat.Attribute("volume")));
      aCatalog.protocol = std::string(safe(cat.Attribute("protocol")));
      m_dataCatalogs.push_back(aCatalog);
    }

    void SiteLocalConfigService::parse(std::string const &url) {
      tinyxml2::XMLDocument doc;
      auto loadErr = doc.LoadFile(url.c_str());
      if (loadErr != tinyxml2::XML_SUCCESS) {
        return;
      }

      // The Site Config has the following format
      // <site-local-config>
      // <site name="FNAL">
      //   <subsite name="FNAL_SUBSITE"/>
      //   <event-data>
      //     <catalog url="trivialcatalog_file:/x/y/z.xml"/>
      //     <rfiotype value="castor"/>
      //   </event-data>
      //   <calib-data>
      //     <catalog url="trivialcatalog_file:/x/y/z.xml"/>
      //     <frontier-connect>
      //       ... frontier-interpreted server/proxy xml ...
      //     </frontier-connect>
      //     <local-connect>
      //       <connectString prefix="anything1" suffix="anything2"/>
      //     </local-connect>
      //   </calib-data>
      //   <source-config>
      //     <cache-temp-dir name="/a/b/c"/>
      //     <cache-hint value="..."/>
      //     <read-hint value="..."/>
      //     <ttree-cache-size value="0"/>
      //     <native-protocols>
      //        <protocol  prefix="dcache"/>
      //        <protocol prefix="file"/>
      //     </native-protocols>
      //   </source-config>
      // </site>
      // </site-local-config>

      auto rootElement = doc.RootElement();

      for (auto site = rootElement->FirstChildElement("site"); site != nullptr;
           site = site->NextSiblingElement("site")) {
        auto subSite = site->FirstChildElement("subsite");

        // Parse the site name
        m_siteName = safe(site->Attribute("name"));
        m_subSiteName = std::string();
        if (subSite) {
          //check to make sure subSite has no children
          auto subSite_first_child = subSite->FirstChild();
          if (subSite_first_child) {
            cms::Exception ex("SiteLocalConfigService");
            ex << "Invalid site-local-config.xml. Subsite node has children!";
            ex.addContext("edm::SiteLocalConfigService::parse()");
            throw ex;
          }
          m_subSiteName = safe(subSite->Attribute("name"));
        }

        // Parsing of the event data section
        auto eventData = site->FirstChildElement("event-data");
        if (eventData) {
          auto catalog = eventData->FirstChildElement("catalog");
          if (catalog) {
            m_trivialDataCatalogs.push_back(safe(catalog->Attribute("url")));
            catalog = catalog->NextSiblingElement("catalog");
            while (catalog) {
              m_trivialDataCatalogs.push_back(safe(catalog->Attribute("url")));
              catalog = catalog->NextSiblingElement("catalog");
            }
          }
          auto rfiotype = eventData->FirstChildElement("rfiotype");
          if (rfiotype) {
            m_rfioType = safe(rfiotype->Attribute("value"));
          }
        }

        //data-access
        //let store catalog entry as: SITE,SUBSITE,STORAGE_SITE,VOLUME,PROTOCOL
        //       SITE: from <site name= /> element
        //       SUBSITE: from <subsite name= /> element. SUBSITE=SITE for site
        //       STORAGE_SITE, VOLUME and PROTOCOL: from <catalog site= volume= protocol= /> in <data-access>. If "site" attribute is not defined inside <catalog />, STORAGE_SITE is SITE
        //Therefore
        //1. if STORAGE_SITE = SITE, use local storage.json since STORAGE_SITE is not a cross site
        //2. if SUBSITE is empty, this is a site. Otherwise, this is a subsite. These are used to define the path to locate the storage.json in FileLocator. This path is provided by storageDescriptionPath() method of this class.
        //get data-access
        auto dataAccess = site->FirstChildElement("data-access");
        if (dataAccess) {
          //get catalogs
          auto catalog = dataAccess->FirstChildElement("catalog");
          if (catalog) {
            //add all info for the first catlog here
            getCatalog(*catalog, m_siteName, m_subSiteName);
            //get next catlog
            catalog = catalog->NextSiblingElement("catalog");
            while (catalog) {
              //add all info for the current catlog here
              getCatalog(*catalog, m_siteName, m_subSiteName);
              //get next catlog
              catalog = catalog->NextSiblingElement("catalog");
            }
          }
        }

        // Parsing of the calib-data section
        auto calibData = site->FirstChildElement("calib-data");

        if (calibData) {
          auto frontierConnect = calibData->FirstChildElement("frontier-connect");

          if (frontierConnect) {
            m_frontierConnect = _toParenString(*frontierConnect);
          }
          auto localConnect = calibData->FirstChildElement("local-connect");
          if (localConnect) {
            if (frontierConnect) {
              throw cms::Exception("Illegal site local configuration")
                  << "It is illegal to include both frontier-connect and local-connect in the same XML file";
            }
            m_useLocalConnectString = true;
            auto connectString = localConnect->FirstChildElement("connectString");
            if (connectString) {
              m_localConnectPrefix = safe(connectString->Attribute("prefix"));
              m_localConnectSuffix = safe(connectString->Attribute("suffix"));
            }
          }
        }

        // Parsing of the source config section
        auto sourceConfig = site->FirstChildElement("source-config");

        if (sourceConfig) {
          auto cacheTempDir = sourceConfig->FirstChildElement("cache-temp-dir");

          if (cacheTempDir) {
            m_cacheTempDir = safe(cacheTempDir->Attribute("name"));
            m_cacheTempDirPtr = &m_cacheTempDir;
          }

          auto cacheMinFree = sourceConfig->FirstChildElement("cache-min-free");

          if (cacheMinFree) {
            //TODO what did xerces do if it couldn't convert?
            m_cacheMinFree = cacheMinFree->DoubleAttribute("value");
            m_cacheMinFreePtr = &m_cacheMinFree;
          }

          auto cacheHint = sourceConfig->FirstChildElement("cache-hint");

          if (cacheHint) {
            m_cacheHint = safe(cacheHint->Attribute("value"));
            m_cacheHintPtr = &m_cacheHint;
          }

          auto cloneCacheHint = sourceConfig->FirstChildElement("clone-cache-hint");

          if (cloneCacheHint) {
            m_cloneCacheHint = safe(cloneCacheHint->Attribute("value"));
            m_cloneCacheHintPtr = &m_cloneCacheHint;
          }

          auto readHint = sourceConfig->FirstChildElement("read-hint");

          if (readHint) {
            m_readHint = safe(readHint->Attribute("value"));
            m_readHintPtr = &m_readHint;
          }

          auto ttreeCacheSize = sourceConfig->FirstChildElement("ttree-cache-size");

          if (ttreeCacheSize) {
            m_ttreeCacheSize = ttreeCacheSize->UnsignedAttribute("value");
            m_ttreeCacheSizePtr = &m_ttreeCacheSize;
          }

          auto timeout = sourceConfig->FirstChildElement("timeout-in-seconds");

          if (timeout) {
            m_timeout = timeout->UnsignedAttribute("value");
            m_timeoutPtr = &m_timeout;
          }

          auto statsDest = sourceConfig->FirstChildElement("statistics-destination");

          if (statsDest) {
            m_statisticsDestination = safe(statsDest->Attribute("endpoint"));
            if (m_statisticsDestination.empty()) {
              m_statisticsDestination = safe(statsDest->Attribute("name"));
            }
            std::string tmpStatisticsInfo = safe(statsDest->Attribute("info"));
            boost::split(m_statisticsInfo, tmpStatisticsInfo, boost::is_any_of("\t ,"));
            m_statisticsInfoAvail = !tmpStatisticsInfo.empty();
          }

          auto prefetching = sourceConfig->FirstChildElement("prefetching");

          if (prefetching) {
            m_enablePrefetching = prefetching->BoolAttribute("value");
            m_enablePrefetchingPtr = &m_enablePrefetching;
          }

          auto nativeProtocol = sourceConfig->FirstChildElement("native-protocols");

          if (nativeProtocol) {
            for (auto child = nativeProtocol->FirstChildElement(); child != nullptr;
                 child = child->NextSiblingElement()) {
              m_nativeProtocols.push_back(safe(child->Attribute("prefix")));
            }
            m_nativeProtocolsPtr = &m_nativeProtocols;
          }
        }
      }
      m_connected = true;
    }

    void SiteLocalConfigService::computeStatisticsDestination() {
      std::vector<std::string> inputStrings;
      boost::split(inputStrings, m_statisticsDestination, boost::is_any_of(":"));
      const std::string &host = inputStrings[0];
      const std::string &port = (inputStrings.size() > 1) ? inputStrings[1] : m_statisticsDefaultPort;
      struct addrinfo *res;
      struct addrinfo hints;
      memset(&hints, '\0', sizeof(hints));
      hints.ai_socktype = SOCK_DGRAM;
      hints.ai_flags = AI_ADDRCONFIG;
      hints.ai_family = AF_UNSPEC;
      int e = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
      if (e != 0) {
        // Silent failure - there's no way to report non-fatal failures from here.
        return;
      }
      m_statisticsAddrInfo = res;
    }

    void SiteLocalConfigService::fillDescriptions(ConfigurationDescriptions &descriptions) {
      ParameterSetDescription desc;
      desc.setComment("Service to translate logical file names to physical file names.");

      desc.addOptionalUntracked<std::string>("siteLocalConfigFileUrl", std::string())
          ->setComment(
              "Specify the file containing the site local config. Empty string will load from default directory.");
      desc.addOptionalUntracked<std::string>("overrideSourceCacheTempDir");
      desc.addOptionalUntracked<double>("overrideSourceCacheMinFree");
      desc.addOptionalUntracked<std::string>("overrideSourceCacheHintDir");
      desc.addOptionalUntracked<std::string>("overrideSourceCloneCacheHintDir")
          ->setComment("Provide an alternate cache hint for fast cloning.");
      desc.addOptionalUntracked<std::string>("overrideSourceReadHint");
      desc.addOptionalUntracked<std::vector<std::string> >("overrideSourceNativeProtocols");
      desc.addOptionalUntracked<unsigned int>("overrideSourceTTreeCacheSize");
      desc.addOptionalUntracked<unsigned int>("overrideSourceTimeout");
      desc.addOptionalUntracked<unsigned int>("debugLevel");
      desc.addOptionalUntracked<bool>("overridePrefetching")
          ->setComment("Request ROOT to asynchronously prefetch I/O during computation.");
      desc.addOptionalUntracked<std::string>("overrideStatisticsDestination")
          ->setComment(
              "Provide an alternate network destination for I/O statistics (must be in the form of host:port).");
      desc.addOptionalUntracked<std::vector<std::string> >("overrideStatisticsInfo")
          ->setComment(
              "Provide an alternate listing of statistics to send (comma separated list; current options are 'dn' or "
              "'nodn').  If left blank, all information is snet (including DNs).");
      desc.addOptionalUntracked<bool>("overrideUseLocalConnectString");
      desc.addOptionalUntracked<std::string>("overrideLocalConnectPrefix");
      desc.addOptionalUntracked<std::string>("overrideLocalConnectSuffix");
      descriptions.add("SiteLocalConfigService", desc);
    }
  }  // namespace service
}  // namespace edm
