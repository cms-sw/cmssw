#include "ProcSmaps.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <optional>
#include <fcntl.h>
#include <map>
#include <string_view>
#include <unistd.h>

namespace {
  struct Mapping {
    edm::service::SmapsSection section = edm::service::SmapsSection::kOther;
    std::string path;
    bool executable = false;
    std::uint64_t sizeKB = 0;
    std::uint64_t rssKB = 0;
    std::uint64_t pssKB = 0;
  };

  bool isSharedLibrary(std::string_view path) {
    auto const position = path.rfind(".so");
    if (position == std::string_view::npos) {
      return false;
    }
    auto const suffix = path.substr(position + 3);
    return suffix.empty() or suffix.front() == '.';
  }

  bool isMappingHeader(std::string_view line) {
    auto const dash = line.find('-');
    auto const space = line.find(' ');
    if (dash == std::string_view::npos or space == std::string_view::npos or dash > space) {
      return false;
    }
    // matches the pattern of a mapping header line in /proc/self/smaps
    // The line should start with hexadecimal addresses, followed by a dash, more hexadecimal addresses, and then a space.
    // matches lines like these
    // 7eff0bea3000-7eff0becc000 r-xp 00002000 08:02 17615                      /usr/lib64/ld-linux-x86-64.so.2
    // 7fa111600000-7fa111c00000 rw-p 00000000 00:00 0
    return std::all_of(line.begin(), line.begin() + dash, [](unsigned char c) { return std::isxdigit(c); }) and
           std::all_of(line.begin() + dash + 1, line.begin() + space, [](unsigned char c) { return std::isxdigit(c); });
  }

  // Returns the next whitespace-delimited token of line, and removes it (and the leading whitespace) from line
  std::string_view nextToken(std::string_view& line) {
    auto const first = line.find_first_not_of(' ');
    if (first == std::string_view::npos) {
      line = {};
      return {};
    }
    line.remove_prefix(first);
    auto const last = line.find(' ');
    auto const token = line.substr(0, last);
    line.remove_prefix(last == std::string_view::npos ? line.size() : last);
    return token;
  }

  // Parses a mapping header line, reusing the storage of mapping.path
  void parseHeader(std::string_view line, Mapping& mapping, bool collectLibraryPaths) {
    mapping.section = edm::service::SmapsSection::kOther;
    mapping.executable = false;
    mapping.sizeKB = 0;
    mapping.rssKB = 0;
    mapping.pssKB = 0;
    mapping.path.clear();

    nextToken(line);  // addresses
    auto const permissions = nextToken(line);
    nextToken(line);  // offset
    nextToken(line);  // device
    nextToken(line);  // inode
    mapping.executable = permissions.size() > 2 and permissions[2] == 'x';

    // The remainder of the line is the path, which may contain spaces
    auto const first = line.find_first_not_of(' ');
    if (first == std::string_view::npos) {
      mapping.section = edm::service::SmapsSection::kMmap;
      return;
    }
    auto const path = line.substr(first);
    if (path.front() == '/') {
      if (path.ends_with(".pcm")) {
        mapping.section = edm::service::SmapsSection::kPcm;
      } else if (isSharedLibrary(path)) {
        mapping.section = edm::service::SmapsSection::kSharedObject;
      } else {
        mapping.section = edm::service::SmapsSection::kOtherFile;
      }
      // The path is needed only for the per-library information (shared objects and PCMs)
      if (collectLibraryPaths and (mapping.section == edm::service::SmapsSection::kSharedObject or
                                   mapping.section == edm::service::SmapsSection::kPcm)) {
        mapping.path.assign(path);
      }
    } else if (path == "[stack]") {
      mapping.section = edm::service::SmapsSection::kStack;
    }
  }

  std::uint64_t valueKB(std::string_view line) {
    auto const colon = line.find(':');
    if (colon == std::string_view::npos) {
      return 0;
    }
    auto const first = line.find_first_not_of(' ', colon + 1);
    if (first == std::string_view::npos) {
      return 0;
    }
    std::uint64_t value = 0;
    std::from_chars(line.data() + first, line.data() + line.size(), value);
    return value;
  }
}  // namespace

// Parses the content of an smaps file read from the open file descriptor fd.
// The per-library information (SmapsInfo::libraries) is collected only if
// collectLibraryPaths is true, because it requires memory allocations.
// Returns an empty optional if reading the file failed.

namespace edm::service {
  std::optional<SmapsInfo> parseProcSmaps(int fd, bool collectLibraryPaths) {
    SmapsInfo result;
    std::map<std::string, SmapsLibraryInfo, std::less<>> libraries;
    Mapping mapping;
    bool haveMapping = false;

    auto finishMapping = [&]() {
      if (not haveMapping) {
        return;
      }
      auto const index = static_cast<unsigned int>(mapping.section);
      result.sectionVSize_[index] += mapping.sizeKB;
      result.sectionRss_[index] += mapping.rssKB;
      if (mapping.section == SmapsSection::kSharedObject) {
        result.sharedObjectPss_ += mapping.pssKB;
        if (mapping.executable) {
          result.sharedObjectExecutableSize_ += mapping.sizeKB;
        }
      }
      if (collectLibraryPaths and mapping.section == SmapsSection::kSharedObject) {
        auto it = libraries.find(std::string_view(mapping.path));
        if (it == libraries.end()) {
          it = libraries.emplace(mapping.path, SmapsLibraryInfo{}).first;
        }
        auto& library = it->second;
        library.sizeKB += mapping.sizeKB;
        library.rssKB += mapping.rssKB;
        library.pssKB += mapping.pssKB;
        if (mapping.executable) {
          library.executableSizeKB += mapping.sizeKB;
        }
      }
    };

    auto processLine = [&](std::string_view line) {
      if (isMappingHeader(line)) {
        finishMapping();
        parseHeader(line, mapping, collectLibraryPaths);
        haveMapping = true;
      } else if (line.starts_with("Private_Clean:") or line.starts_with("Private_Dirty:")) {
        result.private_ += valueKB(line);
      } else if (line.starts_with("Pss:")) {
        auto const value = valueKB(line);
        result.pss_ += value;
        mapping.pssKB += value;
      } else if (line.starts_with("AnonHugePages:")) {
        result.anonHugePages_ += valueKB(line);
      } else if (line.starts_with("Rss:")) {
        mapping.rssKB += valueKB(line);
      } else if (line.starts_with("Size:")) {
        mapping.sizeKB += valueKB(line);
      }
    };

    // The read buffer is on the stack, in order to avoid memory allocations
    // while reading the file (that could change the content of smaps).
    std::array<char, 64 * 1024> buffer;
    std::size_t filled = 0;  // number of bytes in the buffer that are not yet processed
    bool eof = false;
    while (not eof) {
      auto const n = ::read(fd, buffer.data() + filled, buffer.size() - filled);
      if (n < 0) {
        if (errno == EINTR) {
          continue;
        }
        return std::nullopt;
      }
      eof = (n == 0);
      filled += static_cast<std::size_t>(n);

      // Process all complete lines in the buffer
      std::string_view data(buffer.data(), filled);
      std::size_t processed = 0;
      while (true) {
        auto const newline = data.find('\n', processed);
        if (newline == std::string_view::npos) {
          break;
        }
        processLine(data.substr(processed, newline - processed));
        processed = newline + 1;
      }
      if (eof) {
        // last line without a trailing newline
        if (processed < filled) {
          processLine(data.substr(processed));
        }
        break;
      }
      if (processed == 0 and filled == buffer.size()) {
        // A line longer than the buffer; no valid smaps line is, so discard the data
        filled = 0;
      } else {
        // Move the incomplete last line to the beginning of the buffer
        std::memmove(buffer.data(), buffer.data() + processed, filled - processed);
        filled -= processed;
      }
    }
    finishMapping();

    if (collectLibraryPaths) {
      // Report only the file name of a library, unless there are several
      // libraries with the same file name, in which case the full path is kept.
      auto fileName = [](std::string_view path) {
        auto const slash = path.rfind('/');
        return slash == std::string_view::npos ? path : path.substr(slash + 1);
      };
      std::map<std::string_view, unsigned int> nameCounts;
      for (auto const& entry : libraries) {
        ++nameCounts[fileName(entry.first)];
      }
      result.libraries.reserve(libraries.size());
      for (auto& [path, library] : libraries) {
        auto const name = fileName(path);
        library.name = nameCounts[name] > 1 ? path : std::string(name);
        result.libraries.emplace_back(std::move(library));
      }
    }
    return result;
  }

  std::optional<SmapsInfo> readProcSmaps(bool collectLibraryPaths) {
    std::string const& path = "/proc/self/smaps";
    int const fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      return std::nullopt;
    }
    struct FdCloser {
      int fd;
      ~FdCloser() { ::close(fd); }
    } closer{fd};
    return parseProcSmaps(fd, collectLibraryPaths);
  }
}  // namespace edm::service
