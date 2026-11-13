#ifndef FWCore_Services_ProcSmaps_h
#define FWCore_Services_ProcSmaps_h

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace edm::service {
  enum class SmapsSection : unsigned int { kSharedObject = 0, kPcm, kOtherFile, kStack, kMmap, kOther, kSize };

  struct SmapsLibraryInfo {
    // The full path name
    std::string name;
    std::uint64_t vsizeKB_ = 0;
    std::uint64_t rssKB_ = 0;
    std::uint64_t pssKB_ = 0;
    std::uint64_t executableSizeKB_ = 0;
  };

  struct SmapsInfo {
    static constexpr auto sectionsSize_ = static_cast<unsigned int>(SmapsSection::kSize);

    std::uint64_t privateKB_ = 0;
    std::uint64_t pssKB_ = 0;
    std::uint64_t anonHugePagesKB_ = 0;
    std::array<std::uint64_t, sectionsSize_> sectionRss_{};
    std::array<std::uint64_t, sectionsSize_> sectionVSize_{};
    // Totals over all shared object mappings (the vsize and RSS are in sectionVSize_ and sectionRss_)
    std::array<std::uint64_t, sectionsSize_> sectionPss_{};
    std::array<std::uint64_t, sectionsSize_> sectionExecutableSize_{};
    // Filled only if the per-library information was requested
    std::vector<SmapsLibraryInfo> libraries;
  };

  // Opens and parses the smaps file.
  // Returns an empty optional if the file could not be opened or read, it is
  // up to the caller to decide whether that is an error.
  // The per-library information (SmapsInfo::libraries) is collected only if
  // collectLibraryPaths is true, because it requires memory allocations.

  std::optional<SmapsInfo> readProcSmaps(bool collectLibraryPaths = false);
}  // namespace edm::service

#endif
