#ifndef FWCore_Catalog_InputFileCatalog_h
#define FWCore_Catalog_InputFileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// Class InputFileCatalog
//
// We use the following terminology in this class:
//
//     Configured File Name: These are passed in as an
//     argument to a constructor of this class. It is also possible to pass in
//     a std::vector<std::string> in special cases (often
//     in tests). The only modification to these names is
//     that they are trimmed of leading and trailing whitespace.
//
//     Logical File Name (LFN): An LFN is the same as a
//     "Configured File Name" if the "Configured File Name"
//     does not contain a ":" character. If you ask for
//     an LFN that corresponds to a "Configured File Name"
//     that contains a ":" character, then you will get
//     an empty string.
//
//     Physical File Name (PFN): The actual name passed to
//     the storage system to open a file. If a "Configured
//     File Name" is an "LFN", then the PFN is calculated
//     by asking the FileLocator associated with a data
//     catalog to construct a PFN from the LFN. If the
//     "Configured File Name" is not an LFN, then the PFN
//     is the same as the "Configured File Name".
//
//     Note that if an LFN does not start with "/store/",
//     then the PFN will be an empty string. It is up to the
//     client code to deal with this case, but usually this
//     is considered an error. In practice, the rules in a
//     storage.json will usually only yield an empty PFN in
//     that case, but this is not guaranteed by the code in
//     this class. The rules in storage.json might make other
//     requirements on an LFN.
//
// Each FileLocator corresponds to a data catalog
// identified in the file site-local-config.xml.
// There are rules in the storage.json associated
// with each catalog that determine how PFNs are
// constructed from LFNs. One can also pass an
// override string to the constructor to specify
// a different catalog than the catalogs specified in
// the site-local-config.xml file.
//
// Catalogs are based on Rucio.
//
// Note that support for TrivialFileCatalog was removed in 2025.
//
//////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/propagate_const.h"

namespace edm {

  class FileLocator;

  class InputFileCatalog {
  public:
    InputFileCatalog(std::vector<std::string> configuredFileNames,
                     std::string const& override,
                     bool useLFNasPFNifLFNnotFound = false);
    ~InputFileCatalog();

    // Configured File Names containing a ':' are Physical File Names.
    static bool isPhysicalFileName(std::string const& configuredFileName);

    std::vector<std::string> const& configuredFileNames() const { return configuredFileNames_; }
    bool empty() const { return configuredFileNames_.empty(); }

    std::string const& logicalFileName(unsigned int fileIndex) const;
    std::string const& logicalFileName(std::string const& configuredFileName) const;

    // Return PFNs associated with a configuredFileName. If the
    // configuredFileName is an LFN, then each PFN is associated
    // with a catalog. The PFN vector will have 1 entry if the
    // configuredFileName is not an LFN or an override catalog is used.
    std::vector<std::string> physicalFileNames(std::string const& configuredFileName) const;

    // Same as the previous function, except fileIndex is an index into configuredFileNames_
    std::vector<std::string> physicalFileNames(unsigned int fileIndex) const;

    std::string firstPFNFromFirstCatalog() const;

    // This function might use excessive memory if used in
    // cases where the number of configured file names is large
    std::vector<std::string> allPFNsFromFirstCatalog() const;

  private:
    // These functions can return an empty string.
    // They are private because they assume iCatalog is in range.
    std::string physicalFileName(unsigned int iCatalog, std::string const& configuredFileName, bool isPFN) const;
    std::string pfnFromCatalog(unsigned int iCatalog, std::string const& lfn) const;

    std::vector<std::string> configuredFileNames_;
    propagate_const<std::unique_ptr<FileLocator>> overrideFileLocator_;
    std::vector<propagate_const<std::unique_ptr<FileLocator>>> fileLocators_;
    bool useLFNasPFNifLFNnotFound_;
  };
}  // namespace edm
#endif
