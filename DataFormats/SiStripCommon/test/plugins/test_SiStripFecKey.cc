// system includes
#include <sstream>

// user includes
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class testSiStripFecKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripFecKey.
*/
class testSiStripFecKey : public edm::one::EDAnalyzer<> {
public:
  testSiStripFecKey(const edm::ParameterSet&);
  ~testSiStripFecKey();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);

private:
  const uint32_t crate_;
  const uint32_t slot_;
  const uint32_t ring_;
  const uint32_t ccu_;
  const uint32_t module_;
  const uint32_t lld_;
  const uint32_t i2c_;
};

using namespace sistrip;

// -----------------------------------------------------------------------------
//
testSiStripFecKey::testSiStripFecKey(const edm::ParameterSet& pset)
    : crate_(pset.getUntrackedParameter<uint32_t>("CRATE", sistrip::invalid32_)),
      slot_(pset.getUntrackedParameter<uint32_t>("SLOT", sistrip::invalid32_)),
      ring_(pset.getUntrackedParameter<uint32_t>("RING", sistrip::invalid32_)),
      ccu_(pset.getUntrackedParameter<uint32_t>("CCU", sistrip::invalid32_)),
      module_(pset.getUntrackedParameter<uint32_t>("MODULE", sistrip::invalid32_)),
      lld_(pset.getUntrackedParameter<uint32_t>("LLD", sistrip::invalid32_)),
      i2c_(pset.getUntrackedParameter<uint32_t>("I2C", sistrip::invalid32_)) {
  LogTrace(mlDqmCommon_) << "[testSiStripFecKey::" << __func__ << "]"
                         << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
testSiStripFecKey::~testSiStripFecKey() {
  LogTrace(mlDqmCommon_) << "[testSiStripFecKey::" << __func__ << "]"
                         << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void testSiStripFecKey::beginJob() {
  uint32_t cntr = 0;
  std::vector<uint32_t> keys;

  // simple loop
  for (uint16_t iloop = 0; iloop < 1; iloop++) {
    // FEC crates
    for (uint16_t icrate = 0; icrate <= sistrip::FEC_CRATE_MAX + 1; icrate++) {
      if (icrate > 1 && icrate < sistrip::FEC_CRATE_MAX) {
        continue;
      }

      // FEC slots
      for (uint16_t ifec = 0; ifec <= sistrip::SLOTS_PER_CRATE + 1; ifec++) {
        if (ifec > 1 && ifec < sistrip::SLOTS_PER_CRATE) {
          continue;
        }

        // FEC rings
        for (uint16_t iring = 0; iring <= sistrip::FEC_RING_MAX + 1; iring++) {
          if (iring > 1 && iring < sistrip::FEC_RING_MAX) {
            continue;
          }

          // CCU addr
          for (uint16_t iccu = 0; iccu <= sistrip::CCU_ADDR_MAX + 1; iccu++) {
            if (iccu > 1 && iccu < sistrip::CCU_ADDR_MAX) {
              continue;
            }

            // CCU channel
            for (uint16_t ichan = 0; ichan <= sistrip::CCU_CHAN_MAX + 1; ichan++) {
              if (ichan > 1 && ichan != sistrip::CCU_CHAN_MIN && ichan < sistrip::CCU_CHAN_MAX - 1) {
                continue;
              }

              // LLD channels
              for (uint16_t illd = 0; illd <= sistrip::LLD_CHAN_MAX + 1; illd++) {
                if (illd > 1 && illd < sistrip::LLD_CHAN_MAX) {
                  continue;
                }

                // APV
                for (uint16_t iapv = 0; iapv <= sistrip::APV_I2C_MAX + 1; iapv++) {
                  if (iapv > 1 && iapv != sistrip::APV_I2C_MIN && iapv < sistrip::APV_I2C_MAX) {
                    continue;
                  }

                  cntr++;

                  SiStripFecKey tmp1(icrate, ifec, iring, iccu, ichan, illd, iapv);
                  SiStripFecKey tmp2 = SiStripFecKey(tmp1.key());
                  SiStripFecKey tmp3 = SiStripFecKey(tmp1.path());
                  SiStripFecKey tmp4 = SiStripFecKey(tmp1);
                  SiStripFecKey tmp5;
                  tmp5 = tmp1;

                  keys.push_back(tmp1.key());

                  // Print out FEC
                  std::stringstream ss;
                  ss << "[SiStripFecKey::" << __func__ << "]" << std::endl
                     << " From loop   : "
                     << "FEC:crate/slot/ring/CCU/module/LLD/I2C= " << icrate << "/" << ifec << "/" << iring << "/"
                     << iccu << "/" << ichan << "/" << illd << "/" << iapv << std::endl
                     << " From values : ";
                  tmp1.terse(ss);
                  ss << std::endl << " From key    : ";
                  tmp1.terse(ss);
                  ss << std::endl << " From dir    : ";
                  tmp1.terse(ss);
                  ss << std::endl
                     << " Granularity : " << SiStripEnumsAndStrings::granularity(tmp1.granularity()) << std::endl
                     << " Directory   : " << tmp1.path() << std::endl
                     << std::boolalpha << " isValid     : " << tmp1.isValid() << "/" << tmp1.isValid(tmp1.granularity())
                     << "/" << tmp1.isValid(sistrip::APV) << " (general/granularity/apv)" << std::endl
                     << " isInvalid   : " << tmp1.isInvalid() << "/" << tmp1.isInvalid(tmp1.granularity()) << "/"
                     << tmp1.isInvalid(sistrip::APV) << " (general/granularity/apv)" << std::endl
                     << std::noboolalpha;

                  if (tmp1.isValid()) {
                    edm::LogVerbatim(mlDqmCommon_) << ss.str();
                  } else {
                    LogTrace(mlDqmCommon_) << ss.str();
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  edm::LogVerbatim(mlDqmCommon_) << "[SiStripFecKey::" << __func__ << "]"
                                 << " Processed " << cntr << " FED keys";

  std::sort(keys.begin(), keys.end());

  SiStripFecKey value(crate_, slot_, ring_, ccu_, module_, lld_, i2c_);

  typedef std::vector<uint32_t>::iterator iter;
  std::pair<iter, iter> temp = std::equal_range(keys.begin(), keys.end(), value.key(), ConsistentWithKey(value));

  if (temp.first != temp.second) {
    std::stringstream ss;
    ss << std::endl;
    for (iter ii = temp.first; ii != temp.second; ++ii) {
      SiStripFecKey(*ii).terse(ss);
      ss << std::endl;
    }
    LogTrace(mlDqmCommon_) << "[SiStripFecKey::" << __func__ << "]"
                           << " Beginning of list of matched keys: " << ss.str() << "[SiStripFecKey::" << __func__
                           << "]"
                           << " End of list of matched keys: ";
  }

  if (find(keys.begin(), keys.end(), value.key()) != keys.end()) {
    std::stringstream ss;
    ss << "[SiStripFecKey::" << __func__ << "]"
       << " Found key ";
    value.terse(ss);
    ss << " in list! ";
    LogTrace(mlDqmCommon_) << ss.str();
  }

  edm::LogVerbatim(mlDqmCommon_) << "[SiStripFecKey::" << __func__ << "]" << std::endl
                                 << " Total number of keys   : " << keys.size() << std::endl
                                 << " Number of matching key : " << temp.second - temp.first;
}

// -----------------------------------------------------------------------------
//
void testSiStripFecKey::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  LogTrace(mlDqmCommon_) << "[SiStripFecKey::" << __func__ << "]"
                         << " Analyzing run/event " << event.id().run() << "/" << event.id().event();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testSiStripFecKey);
