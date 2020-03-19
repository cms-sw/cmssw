#ifndef hcal_ConfigurationDatabase_hh_included
#define hcal_ConfigurationDatabase_hh_included 1

#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseException.hh"
#include "xercesc/dom/DOMDocument.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#ifdef HAVE_XDAQ
#include "log4cplus/logger.h"
#else
#include "CaloOnlineTools/HcalOnlineDb/interface/xdaq_compat.h"  // Includes typedef for log4cplus::Logger
#endif

namespace hcal {

  class ConfigurationDatabaseImpl;

  /** \brief Access to large-block configuration information such as firmwares and lookup tables

  Multithread access is not allowed.  The locking must be handled externally.
	\ingroup hcalBase
   */
  class ConfigurationDatabase {
  public:
    ConfigurationDatabase(log4cplus::Logger logger);

    /** \brief Open the database, using the given accessor */
    void open(const std::string& accessor) noexcept(false);

    typedef xercesc::DOMDocument* ApplicationConfig;

    // General configuration (applications)
    /** \brief Get the application configuration for the given item
        \note User must release all DOMDocuments
     */
    ApplicationConfig getApplicationConfig(const std::string& tag,
                                           const std::string& classname,
                                           int instance) noexcept(false);

    // Get the configuration document (whole)
    std::string getConfigurationDocument(const std::string& tag) noexcept(false);

    // Firmware-related
    /** \brief Retrieve the checksum for a given firmware version */
    unsigned int getFirmwareChecksum(const std::string& board, unsigned int version) noexcept(false);
    /** \brief Retrieve the MCS file lines for a given firmware version */
    void getFirmwareMCS(const std::string& board,
                        unsigned int version,
                        std::vector<std::string>& mcsLines) noexcept(false);

    typedef enum FPGASelectionEnum { Bottom = 0, Top = 1 } FPGASelection;

    struct FPGAId {
      FPGAId() {}
      FPGAId(int cr, int sl, FPGASelection fp) : crate(cr), slot(sl), fpga(fp) {}
      bool operator<(const FPGAId& a) const;
      int crate;
      int slot;
      FPGASelection fpga;
    };

    typedef enum LUTTypeEnum { LinearizerLUT = 1, CompressionLUT = 2 } LUTType;

    // LUT-related
    struct LUTId : public FPGAId {
      LUTId() {}
      LUTId(int crate_, int slot_, FPGASelection fpga_, int fiberorslb_, int chan_, LUTType lt)
          : FPGAId(crate_, slot_, fpga_), fiber_slb(fiberorslb_), channel(chan_), lut_type(lt) {}
      bool operator<(const LUTId& a) const;
      int fiber_slb;  // fiber for linearizing, slb for compression
      int channel;    // fiberchan or slbchan
      LUTType lut_type;
    };
    typedef std::vector<unsigned int> LUT;

    /** \brief Retrieve the data for the requested LUTs (organized this way for better database efficiency)
	\param tag Tag name for this LUT setup
	\param crate Crate number
	\param slot Slot number
	\param LUTs Return map of the LUTs
     */
    void getLUTs(const std::string& tag, int crate, int slot, std::map<LUTId, LUT>& LUTs) noexcept(false);

    typedef std::vector<unsigned char> MD5Fingerprint;

    /** \brief Retrieve the LUT checksums for all LUTs
	\param tag Tag name for this LUT setup
	\param checksums Checksums (MD5)
    */
    void getLUTChecksums(const std::string& tag, std::map<LUTId, MD5Fingerprint>& checksums) noexcept(false);

    struct PatternId : public FPGAId {
      bool operator<(const PatternId& a) const;
      PatternId(int cr, int sl, FPGASelection f, int fib) : FPGAId(cr, sl, f), fiber(fib) {}
      int fiber;
    };
    typedef std::vector<unsigned int> HTRPattern;

    /** \brief Retrieve the data for the requested pattern ram
	\brief tag Tag name for this pattern ram set
	\param crate Crate number
	\param slot Slot number
	\param patterns Return map of the patterns
    */
    void getPatterns(const std::string& tag,
                     int crate,
                     int slot,
                     std::map<PatternId, HTRPattern>& patterns) noexcept(false);

    // ZS-related
    struct ZSChannelId : public FPGAId {
      ZSChannelId() {}
      ZSChannelId(int crate_, int slot_, FPGASelection fpga_, int fiber_, int chan_)
          : FPGAId(crate_, slot_, fpga_), fiber(fiber_), channel(chan_) {}
      bool operator<(const ZSChannelId& a) const;
      int fiber;
      int channel;
    };

    /** \brief Retrieve the zs thresholds for the specified slot
	\brief tag Tag name 
	\param crate Crate number
	\param slot Slot number
	\param patterns Return map of the thresholds
    */
    void getZSThresholds(const std::string& tag,
                         int crate,
                         int slot,
                         std::map<ZSChannelId, int>& thresholds) noexcept(false);

    struct HLXMasks {
      uint32_t occMask;
      uint32_t lhcMask;
      uint32_t sumEtMask;
    };

    /** \brief Retrieve the HLX masks for the given slot
	\brief tag Tag name 
	\param crate Crate number
	\param slot Slot number
	\param patterns Return map of the masks
    */
    void getHLXMasks(const std::string& tag, int crate, int slot, std::map<FPGAId, HLXMasks>& masks) noexcept(false);

    typedef enum RBXdatumTypeEnum {
      eRBXdelay = 1,
      eRBXpedestal = 2,
      eRBXttcrxPhase = 3,
      eRBXgolCurrent = 4,
      eRBXbroadcast = 5,
      eRBXqieResetDelay = 6,
      eRBXledData = 7,
      eRBXccaPatterns = 8
    } RBXdatumType;
    typedef enum LEDdatumTypeEnum {
      eLEDnotApplicable = 0,
      eLEDenable = 1,
      eLEDamplitude = 2,
      eLEDtiming_hb = 3,
      eLEDtiming_lb = 4
    } LEDdatumType;
    struct RBXdatumId {
      bool operator<(const RBXdatumId& a) const;
      RBXdatumId(LEDdatumType lt) : rm(0), card(0), qie_or_gol(0), dtype(eRBXledData), ltype(lt) {}
      RBXdatumId(int r, int c, int qg, RBXdatumType dt)
          : rm(r), card(c), qie_or_gol(qg), dtype(dt), ltype(eLEDnotApplicable) {}
      RBXdatumId(RBXdatumType dt) : rm(0), card(0), qie_or_gol(0), dtype(dt), ltype(eLEDnotApplicable) {}
      int rm;
      int card;
      int qie_or_gol;
      RBXdatumType dtype;
      LEDdatumType ltype;
    };

    typedef unsigned char RBXdatum;

    /** \brief Retrieve the data for the requested RBX
	\param tag Tag name for this RBX config
	\param rbx RBX name
	\param dtype Datum type
	\param RBXdata Return map of the data
     */
    void getRBXdata(const std::string& tag,
                    const std::string& rbx,
                    RBXdatumType dtype,
                    std::map<RBXdatumId, RBXdatum>& RBXdata) noexcept(false);

    typedef std::vector<unsigned char> RBXpattern;

    /** \brief Retrieve CCA patterns for the requested RBX
	\param tag Tag name for this RBX config
	\param rbx RBX name
	\param patterns Return map of the patterns
     */
    void getRBXpatterns(const std::string& tag,
                        const std::string& rbx,
                        std::map<RBXdatumId, RBXpattern>& patterns) noexcept(false);

    /** \brief Close the database */
    void close();

    /** \brief Retrieve the */

    /** \brief Access the implementation directly (when open) 
	\note For expert use only
    */
    ConfigurationDatabaseImpl* getImplementation() { return m_implementation; }

  private:
    std::vector<ConfigurationDatabaseImpl*> m_implementationOptions;
    ConfigurationDatabaseImpl* m_implementation;
    log4cplus::Logger m_logger;
  };

}  // namespace hcal

#endif  // hcal_ConfigurationDatabase_hh_included
