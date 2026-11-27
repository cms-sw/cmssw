#ifndef FWCore_Services_ProcSmaps_h
#define FWCore_Services_ProcSmaps_h

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace edm::service {
  enum class SmapsSection : unsigned int { kSharedObject = 0, kPcm, kOtherFile, kStack, kMmap, kOther, kSize };

  struct SmapsPathInfo {
    // The full path name
    std::string path;
    std::uint64_t vsizeKB_ = 0;
    std::uint64_t rssKB_ = 0;
    std::uint64_t pssKB_ = 0;
    std::uint64_t executableSizeKB_ = 0;
  };

  struct SmapsInfo {
    static constexpr auto sectionsSize_ = static_cast<unsigned int>(SmapsSection::kSize);

    std::uint64_t privateKB_ = 0;
    std::uint64_t anonHugePagesKB_ = 0;
    std::array<std::uint64_t, sectionsSize_> sectionRss_{};
    std::array<std::uint64_t, sectionsSize_> sectionVSize_{};
    // Totals over all shared object mappings (the vsize and RSS are in sectionVSize_ and sectionRss_)
    std::array<std::uint64_t, sectionsSize_> sectionPss_{};
    std::array<std::uint64_t, sectionsSize_> sectionExecutableSize_{};
    // Filled only if the per-memory-mapped-file path information was requested
    std::vector<SmapsPathInfo> paths;
  };

  // Opens and parses the smaps file.
  // Returns an empty optional if the file could not be opened or read, it is
  // up to the caller to decide whether that is an error.
  // The per-memory-mapped-file path information (SmapsInfo::paths) is collected only if
  // collectPaths is true, because it requires memory allocations.

  std::optional<SmapsInfo> readProcSmaps(bool collectPaths = false);
}  // namespace edm::service

#endif
