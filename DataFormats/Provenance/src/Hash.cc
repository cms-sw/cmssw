#include "DataFormats/Provenance/interface/Hash.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace detail {
    // This string is the 16-byte, non-printable version.
    std::string const& InvalidHash() {
      static const std::string invalid = cms::MD5Result().compactForm();
      return invalid;
    }
  }

  namespace hash_detail {
    value_type
    compactForm_(value_type const& hash) {
      if (isCompactForm_(hash)) {
	return hash;
      }
      value_type temp(hash);
      fixup_(temp);
      return temp;
    }

    // 'Fix' the string data member of this Hash, i.e., if it is in
    // the hexified (32 byte) representation, make it be in the
    // 16-byte (unhexified) representation.
    void
    fixup_(value_type& hash) {
      switch (hash.size()) {
        case 0: {
          hash = edm::detail::InvalidHash();
        }	
        case 16: {
          break;
        }
        case 32: {
          cms::MD5Result temp;
          temp.fromHexifiedString(hash);
          hash = temp.compactForm();
          break;
        }
        default: {
          throw edm::Exception(edm::errors::LogicError)
            << "edm::Hash<> instance with data in illegal state:\n"
            << hash
            << "\nPlease report this to the core framework developers";
        }
      }
    }

    bool
    isCompactForm_(value_type const& hash) {
      return 16 == hash.size();
    }

    bool
    isValid_(value_type const& hash) {
      return isCompactForm_(hash) ? (hash != edm::detail::InvalidHash()) : (!hash.empty());
    }

    void
    throwIfIllFormed(value_type const& hash) {
      // Fixup not needed here.
      if (hash.size() % 2 == 1) {
	throw edm::Exception(edm::errors::LogicError)
	  << "Ill-formed Hash instance. "
	  << "Please report this to the core framework developers";
      }
    }

    void
    toString_(std::string& result, value_type const& hash) {
      value_type temp1(hash);
      fixup_(temp1);
      cms::MD5Result temp;
      copy_all(temp1, temp.bytes);
      result += temp.toString();
    }

    std::ostream&
    print_(std::ostream& os, value_type const& hash) {
      value_type temp1(hash);
      fixup_(temp1);
      cms::MD5Result temp;
      copy_all(temp1, temp.bytes);
      os << temp.toString();
      return os;
    }
  }
}
