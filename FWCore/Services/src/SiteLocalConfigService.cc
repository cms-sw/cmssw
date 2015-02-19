//<<<<<< INCLUDES                                                       >>>>>>

#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/util/XMLString.hpp>
#include <sstream>
#include <memory>
#include <boost/algorithm/string.hpp>

using namespace xercesc;

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
  inline std::string _toString(XMLCh const* toTranscode) {
      std::string tmp(XMLString::transcode(toTranscode));
      return tmp;
  }

  inline unsigned int _toUInt(XMLCh const* toTranscode) {
     std::istringstream iss(_toString(toTranscode));
     unsigned int returnValue;
     iss >> returnValue;
     return returnValue;
  }

  inline bool _toBool(XMLCh const* toTranscode) {
    std::string value = _toString(toTranscode);
    if ((value == "true") || (value == "1"))
      return true;
    return false;
  }

  inline double _toDouble(XMLCh const* toTranscode) {
     std::istringstream iss(_toString(toTranscode));
     double returnValue;
     iss >> returnValue;
     return returnValue;
  }

  inline XMLCh*  _toDOMS(std::string temp) {
      XMLCh* buff = XMLString::transcode(temp.c_str());
      return  buff;
  }

  // concatenate all the XML node attribute/value pairs into a
  // paren-separated string (for use by CORAL and frontier_client)
  inline std::string _toParenString(DOMNode const& nodeToConvert) {
      std::ostringstream oss;

      DOMNodeList *childList = nodeToConvert.getChildNodes();

      unsigned int numNodes = childList->getLength();
      for (unsigned int i = 0; i < numNodes; ++i) {
        DOMNode *childNode = childList->item(i);
        if (childNode->getNodeType() != DOMNode::ELEMENT_NODE) {
            continue;
        }
        DOMElement *child = static_cast<DOMElement *>(childNode);

        DOMNamedNodeMap *attributes = child->getAttributes();
        unsigned int numAttributes = attributes->getLength();
        for (unsigned int j = 0; j < numAttributes; ++j) {
            DOMNode *attributeNode = attributes->item(j);
            if (attributeNode->getNodeType() != DOMNode::ATTRIBUTE_NODE) {
              continue;
            }
            DOMAttr *attribute = static_cast<DOMAttr *>(attributeNode);

            oss << "(" << _toString(child->getTagName()) <<
                          _toString(attribute->getName()) << "=" <<
                          _toString(attribute->getValue()) << ")";
        }
      }
      return oss.str();
  }

  template<typename T>
  static
  void overrideFromPSet(char const* iName, edm::ParameterSet const& iPSet,
                        T& iHolder, T const*& iPointer) {
     if(iPSet.exists(iName)) {
        iHolder = iPSet.getUntrackedParameter<T>(iName);
        iPointer = &iHolder;
     }
  }
}

namespace edm {
  namespace service {

    const std::string SiteLocalConfigService::m_statisticsDefaultPort = "3334";

    SiteLocalConfigService::SiteLocalConfigService(ParameterSet const& pset) :
          m_url("/SITECONF/local/JobConfig/site-local-config.xml"),
          m_dataCatalog(),
          m_fallbackDataCatalog(),
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
          m_siteName() {

        char* tmp = getenv("CMS_PATH");

        if (tmp) {
          m_url = tmp + m_url;
        }

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
        const std::string * tmpStringPtr = nullptr;
        overrideFromPSet("overrideStatisticsDestination", pset, m_statisticsDestination, tmpStringPtr);
        this->computeStatisticsDestination();
        std::vector<std::string> tmpStatisticsInfo; std::vector<std::string> const *tmpStatisticsInfoPtr = nullptr;
        overrideFromPSet("overrideStatisticsInfo", pset, tmpStatisticsInfo, tmpStatisticsInfoPtr);
        if (tmpStatisticsInfoPtr) {
          m_statisticsInfoAvail = true;
          m_statisticsInfo.clear();
          for (auto &entry : tmpStatisticsInfo) {m_statisticsInfo.insert(std::move(entry));}
        }

       if(pset.exists("debugLevel")) {
            m_debugLevel = pset.getUntrackedParameter<unsigned int>("debugLevel");
       }

    }

    SiteLocalConfigService::~SiteLocalConfigService() {
      if (m_statisticsAddrInfo) {
        freeaddrinfo(m_statisticsAddrInfo);
        m_statisticsAddrInfo = nullptr;
      }
    }

    std::string const
    SiteLocalConfigService::dataCatalog(void) const {
        if (!m_connected) {
            //throw cms::Exception("Incomplete configuration")
            //    << "Valid site-local-config not found at " << m_url;
            // Return PoolFileCatalog.xml for now
            return "file:PoolFileCatalog.xml";
        }

        if (m_dataCatalog == "") {
          throw cms::Exception("Incomplete configuration")
            << "Did not find catalog in event-data section in " << m_url;
        }

        return m_dataCatalog;
    }

    std::string const
    SiteLocalConfigService::fallbackDataCatalog(void) const {
        if (!m_connected) {
            //throw cms::Exception("Incomplete configuration")
            //    << "Valid site-local-config not found at " << m_url;
            // Return PoolFileCatalog.xml for now
            return "file:PoolFileCatalog.xml";
        }

        // Note: Unlike the dataCatalog, the fallbackDataCatalog may be empty!
        return  m_fallbackDataCatalog;
    }

    std::string const
    SiteLocalConfigService::frontierConnect(std::string const& servlet) const {
        if (!m_connected) {
          throw cms::Exception("Incomplete configuration")
            << "Valid site-local-config not found at " << m_url;
        }

        if (m_frontierConnect == "") {
          throw cms::Exception("Incomplete configuration")
            << "Did not find frontier-connect in calib-data section in " << m_url;
        }

        if (servlet == "") {
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
        complexstr.append(m_frontierConnect, nextparen, m_frontierConnect.length()-nextparen);

        return complexstr;
    }

    std::string const
    SiteLocalConfigService::lookupCalibConnect(std::string const& input) const {
        static std::string const proto = "frontier://";

        if (input.substr(0,proto.length()) == proto) {
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
          std::string::size_type endservlet = input.find("(", startservlet);
          if (endservlet == std::string::npos) {
              endservlet = input.rfind('/', input.length());
          }
          std::string servlet = input.substr(startservlet, endservlet - startservlet);
          if ((servlet != "") && (servlet.find_first_of(":/)[]") == std::string::npos)) {
              if (servlet == "cms_conditions_data") {
                // use the default servlet from site-local-config.xml
                servlet = "";
              }
              return proto + frontierConnect(servlet) + input.substr(endservlet);
          }
        }
        return input;
    }

    std::string const
    SiteLocalConfigService::rfioType(void) const {
        return m_rfioType;
    }

    std::string const*
    SiteLocalConfigService::sourceCacheTempDir() const {
       return m_cacheTempDirPtr;
    }

    double const*
    SiteLocalConfigService::sourceCacheMinFree() const {
       return m_cacheMinFreePtr;
    }

    std::string const*
    SiteLocalConfigService::sourceCacheHint() const {
       return m_cacheHintPtr;
    }

    std::string const*
    SiteLocalConfigService::sourceCloneCacheHint() const {
       return m_cloneCacheHintPtr;
    }

    std::string const*
    SiteLocalConfigService::sourceReadHint() const {
       return m_readHintPtr;
    }

    unsigned int const*
    SiteLocalConfigService::sourceTTreeCacheSize() const {
       return m_ttreeCacheSizePtr;
    }

    unsigned int const*
    SiteLocalConfigService::sourceTimeout() const {
       return m_timeoutPtr;
    }

    bool
    SiteLocalConfigService::enablePrefetching() const {
       return m_enablePrefetchingPtr ? *m_enablePrefetchingPtr : false;
    }

    unsigned int
    SiteLocalConfigService::debugLevel() const {
       return m_debugLevel;
    }

    std::vector<std::string> const*
    SiteLocalConfigService::sourceNativeProtocols() const {
       return m_nativeProtocolsPtr;
    }

    struct addrinfo const*
    SiteLocalConfigService::statisticsDestination() const {
       return m_statisticsAddrInfo;
    }

    std::set<std::string> const*
    SiteLocalConfigService::statisticsInfo() const {
       return m_statisticsInfoAvail ? &m_statisticsInfo : nullptr;
    }

    std::string const&
    SiteLocalConfigService::siteName() const {
       return m_siteName;
    }

    void
    SiteLocalConfigService::parse(std::string const& url) {
      cms::concurrency::xercesInitialize();
      std::auto_ptr<XercesDOMParser> parser(new XercesDOMParser);
      try {
        parser->setValidationScheme(XercesDOMParser::Val_Auto);
        parser->setDoNamespaces(false);

        parser->parse(url.c_str());
        DOMDocument* doc = parser->getDocument();
        if (!doc) {
          return;
        }

        // The Site Config has the following format
        // <site-local-config>
        // <site name="FNAL">
        //   <event-data>
        //     <catalog url="trivialcatalog_file:/x/y/z.xml"/>
        //     <rfiotype value="castor"/>
        //   </event-data>
        //   <calib-data>
        //     <catalog url="trivialcatalog_file:/x/y/z.xml"/>
        //     <frontier-connect>
        //       ... frontier-interpreted server/proxy xml ...
            //     </frontier-connect>
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

        // FIXME: should probably use the parser for validating the XML.

        DOMNodeList *sites = doc->getElementsByTagName(_toDOMS("site"));
        unsigned int numSites = sites->getLength();
        for (unsigned int i = 0; i < numSites; ++i) {
          DOMElement *site = static_cast<DOMElement *>(sites->item(i));

          // Parse the site name
          m_siteName = _toString(site->getAttribute(_toDOMS("name")));

          // Parsing of the event data section
          {
            DOMNodeList *eventDataList = site->getElementsByTagName(_toDOMS("event-data"));
            if (eventDataList->getLength() > 0) {
              DOMElement *eventData = static_cast<DOMElement *>(eventDataList->item(0));

              DOMNodeList *catalogs = eventData->getElementsByTagName(_toDOMS("catalog"));

              if (catalogs->getLength() > 0) {
                DOMElement * catalog = static_cast<DOMElement *>(catalogs->item(0));
                m_dataCatalog = _toString(catalog->getAttribute(_toDOMS("url")));
              }

              if (catalogs->getLength() > 1) {
                DOMElement * catalog = static_cast<DOMElement *>(catalogs->item(1));
                m_fallbackDataCatalog = _toString(catalog->getAttribute(_toDOMS("url")));
              }

              DOMNodeList* rfiotypes = eventData->getElementsByTagName(_toDOMS("rfiotype"));

              if (rfiotypes->getLength() > 0) {
                DOMElement* rfiotype = static_cast<DOMElement *>(rfiotypes->item(0));
                m_rfioType = _toString(rfiotype->getAttribute(_toDOMS("value")));
              }
            }
          }

          // Parsing of the calib-data section
          {
            DOMNodeList *calibDataList = site->getElementsByTagName(_toDOMS("calib-data"));

            if (calibDataList->getLength() > 0) {
              DOMElement *calibData = static_cast<DOMElement *>(calibDataList->item(0));
              DOMNodeList *frontierConnectList = calibData->getElementsByTagName(_toDOMS("frontier-connect"));

              if (frontierConnectList->getLength() > 0) {
                DOMElement *frontierConnect = static_cast<DOMElement *>(frontierConnectList->item(0));
                m_frontierConnect = _toParenString(*frontierConnect);
              }
            }
          }
          // Parsing of the source config section
          {
            DOMNodeList *sourceConfigList = site->getElementsByTagName(_toDOMS("source-config"));

            if (sourceConfigList->getLength() > 0) {
              DOMElement *sourceConfig = static_cast<DOMElement *>(sourceConfigList->item(0));
              DOMNodeList *cacheTempDirList = sourceConfig->getElementsByTagName(_toDOMS("cache-temp-dir"));

              if (cacheTempDirList->getLength() > 0) {
                DOMElement *cacheTempDir = static_cast<DOMElement *>(cacheTempDirList->item(0));
                m_cacheTempDir = _toString(cacheTempDir->getAttribute(_toDOMS("name")));
                m_cacheTempDirPtr = &m_cacheTempDir;
              }

              DOMNodeList *cacheMinFreeList = sourceConfig->getElementsByTagName(_toDOMS("cache-min-free"));

              if (cacheMinFreeList->getLength() > 0) {
                DOMElement *cacheMinFree = static_cast<DOMElement *>(cacheMinFreeList->item(0));
                m_cacheMinFree = _toDouble(cacheMinFree->getAttribute(_toDOMS("value")));
                m_cacheMinFreePtr = &m_cacheMinFree;
              }

              DOMNodeList *cacheHintList = sourceConfig->getElementsByTagName(_toDOMS("cache-hint"));

              if (cacheHintList->getLength() > 0) {
                DOMElement *cacheHint = static_cast<DOMElement *>(cacheHintList->item(0));
                m_cacheHint = _toString(cacheHint->getAttribute(_toDOMS("value")));
                m_cacheHintPtr = &m_cacheHint;
              }

              DOMNodeList *cloneCacheHintList = sourceConfig->getElementsByTagName(_toDOMS("clone-cache-hint"));

              if (cloneCacheHintList->getLength() > 0) {
                DOMElement *cloneCacheHint = static_cast<DOMElement *>(cloneCacheHintList->item(0));
                m_cloneCacheHint = _toString(cloneCacheHint->getAttribute(_toDOMS("value")));
                m_cloneCacheHintPtr = &m_cloneCacheHint;
              }

              DOMNodeList *readHintList = sourceConfig->getElementsByTagName(_toDOMS("read-hint"));

              if (readHintList->getLength() > 0) {
                DOMElement *readHint = static_cast<DOMElement *>(readHintList->item(0));
                m_readHint = _toString(readHint->getAttribute(_toDOMS("value")));
                m_readHintPtr = &m_readHint;
              }

              DOMNodeList *ttreeCacheSizeList = sourceConfig->getElementsByTagName(_toDOMS("ttree-cache-size"));

              if (ttreeCacheSizeList->getLength() > 0) {
                DOMElement *ttreeCacheSize = static_cast<DOMElement *>(ttreeCacheSizeList->item(0));
                m_ttreeCacheSize = _toUInt(ttreeCacheSize->getAttribute(_toDOMS("value")));
                m_ttreeCacheSizePtr = &m_ttreeCacheSize;
              }

              DOMNodeList *timeoutList = sourceConfig->getElementsByTagName(_toDOMS("timeout-in-seconds"));

              if (timeoutList->getLength() > 0) {
                DOMElement *timeout = static_cast<DOMElement *>(timeoutList->item(0));
                m_timeout = _toUInt(timeout->getAttribute(_toDOMS("value")));
                m_timeoutPtr = &m_timeout;
              }

              DOMNodeList *statsDestList = sourceConfig->getElementsByTagName(_toDOMS("statistics-destination"));

              if (statsDestList->getLength() > 0) {
                DOMElement *statsDest = static_cast<DOMElement *>(statsDestList->item(0));
                m_statisticsDestination = _toString(statsDest->getAttribute(_toDOMS("endpoint")));
                if (!m_statisticsDestination.size()) {
                  m_statisticsDestination = _toString(statsDest->getAttribute(_toDOMS("name")));
                }
                std::string tmpStatisticsInfo = _toString(statsDest->getAttribute(_toDOMS("info")));
                boost::split(m_statisticsInfo, tmpStatisticsInfo, boost::is_any_of("\t ,"));
                m_statisticsInfoAvail = !tmpStatisticsInfo.empty();
              }

              DOMNodeList *prefetchingList = sourceConfig->getElementsByTagName(_toDOMS("prefetching"));

              if (prefetchingList->getLength() > 0) {
                DOMElement *prefetching = static_cast<DOMElement *>(prefetchingList->item(0));
                m_enablePrefetching = _toBool(prefetching->getAttribute(_toDOMS("value")));
                m_enablePrefetchingPtr = &m_enablePrefetching;
              }

              DOMNodeList *nativeProtocolsList = sourceConfig->getElementsByTagName(_toDOMS("native-protocols"));

              if (nativeProtocolsList->getLength() > 0) {
                DOMElement *nativeProtocol = static_cast<DOMElement *>(nativeProtocolsList->item(0));
                DOMNodeList *childList = nativeProtocol->getChildNodes();

                XMLCh* prefixXMLCh = _toDOMS("prefix");
                unsigned int numNodes = childList->getLength();
                for (unsigned int i = 0; i < numNodes; ++i) {
                  DOMNode *childNode = childList->item(i);
                  if (childNode->getNodeType() != DOMNode::ELEMENT_NODE) {
                    continue;
                  }
                  DOMElement *child = static_cast<DOMElement *>(childNode);
                  m_nativeProtocols.push_back(_toString(child->getAttribute(prefixXMLCh)));
                }
                m_nativeProtocolsPtr = &m_nativeProtocols;
              }
            }
          }
        }
        m_connected = true;
      }
      catch (xercesc::DOMException const& e) {
      }
    }

    void
    SiteLocalConfigService::computeStatisticsDestination() {
      std::vector<std::string> inputStrings;
      boost::split(inputStrings, m_statisticsDestination, boost::is_any_of(":"));
      const std::string &host=inputStrings[0];
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

    void
    SiteLocalConfigService::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setComment("Service to translate logical file names to physical file names.");

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
        ->setComment("Provide an alternate network destination for I/O statistics (must be in the form of host:port).");
      desc.addOptionalUntracked<std::vector<std::string> >("overrideStatisticsInfo")
        ->setComment("Provide an alternate listing of statistics to send (comma separated list; current options are 'dn' or 'nodn').  If left blank, all information is snet (including DNs).");

      descriptions.add("SiteLocalConfigService", desc);
    }
  }
}
