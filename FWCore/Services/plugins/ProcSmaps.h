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
    // The file name of the library, or the full path if there are several libraries with the same file name
    std::string name;
    std::uint64_t sizeKB = 0;
    std::uint64_t rssKB = 0;
    std::uint64_t pssKB = 0;
    std::uint64_t executableSizeKB = 0;
  };

  struct SmapsInfo {
    static constexpr auto sectionsSize_ = static_cast<unsigned int>(SmapsSection::kSize);

    std::uint64_t private_ = 0;
    std::uint64_t pss_ = 0;
    std::uint64_t anonHugePages_ = 0;
    std::array<std::uint64_t, sectionsSize_> sectionRss_{};
    std::array<std::uint64_t, sectionsSize_> sectionVSize_{};
    // Totals over all shared object mappings (the size and RSS are in sectionVSize_ and sectionRss_)
    std::uint64_t sharedObjectPss_ = 0;
    std::uint64_t sharedObjectExecutableSize_ = 0;
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
