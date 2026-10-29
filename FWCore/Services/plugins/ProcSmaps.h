#ifndef FWCore_Services_ProcSmaps_h
#define FWCore_Services_ProcSmaps_h

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace edm::service {
  enum class SmapsSection : unsigned int { kSharedObject = 0, kPcm, kOtherFile, kStack, kMmap, kOther, kSize };

  struct SmapsLibraryInfo {
    std::string path;
    std::uint64_t sizeKB = 0;
    std::uint64_t rssKB = 0;
    std::uint64_t pssKB = 0;
    std::uint64_t executableSizeKB = 0;
  };

  struct SmapsInfo {
    static constexpr auto sectionsSize_ = static_cast<unsigned int>(SmapsSection::kSize);

    double private_ = 0;
    double pss_ = 0;
    double anonHugePages_ = 0;
    std::array<double, sectionsSize_> sectionRss_{};
    std::array<double, sectionsSize_> sectionVSize_{};
    std::vector<SmapsLibraryInfo> libraries;
  };

  SmapsInfo parseProcSmaps(std::istream& input);
  SmapsInfo readProcSmaps(std::string const& path = "/proc/self/smaps");
}  // namespace edm::service

#endif
