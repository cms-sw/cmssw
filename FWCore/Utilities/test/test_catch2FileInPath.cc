// Unit tests for edm::FileInPath.
//
// FileInPath caches CMSSW_SEARCH_PATH, CMSSW_RELEASE_BASE, CMSSW_BASE and
// CMSSW_DATA_PATH in function-local statics on first use, and
// disableFileLookup() sets a one-way global flag. This means a single
// process can only ever exercise one environment configuration. To work
// around this, this file builds a single executable that runs one
// "scenario" per invocation, selected by argv[1], and each scenario is run
// as a separate SCRAM test (see the "foreach" test in BuildFile.xml). The
// environment variables are set with setenv()/unsetenv() before any
// FileInPath is constructed, and are never inherited from the surrounding
// shell (see main() below).
//
// Known latent bugs in FileInPath whose tests are excluded from the
// default run (see main()) because they either hang or abort:
//
// 2. p /= relative with an absolute relative path (FileInPath.cc:85):
//    std::filesystem::operator/=  discards the prefix when the right-hand
//    side is an absolute path, so FileInPath("/etc/passwd") bypasses the
//    search path entirely, then hits bug 1 and hangs.
// 3. Empty search-path element (edm::tokenize keeps empty tokens): a
//    CMSSW_SEARCH_PATH like "a::b" yields an empty path prefix, which
//    resolves the relative path against the current working directory
//    instead of a proper prefix. Unlike bugs 1, 2 and 4, this does NOT
//    hang: an empty path's parent_path()/weakly_canonical() are also
//    empty, so the branch-path loop runs zero iterations. The file is
//    genuinely found on disk, but the loop never assigns location_, so
//    initialize_() silently discards the match and moves on (ending, in
//    this test, in the generic not-found throw). See the "emptyPathElement"
//    scenario, which is a normal (non-[hang]) test.
// 4. assert() in removeSymLinksSrc (FileInPath.cc:57): if $CMSSW_BASE/src is
//    a symlink to a directory not literally named "src", the resolved path
//    no longer ends in "/src" and the assert fires (or, with NDEBUG,
//    substr() silently truncates).

#include "catch2/catch_all.hpp"

#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {
  namespace fs = std::filesystem;

  // Absolute, symlink-resolved paths for the current scenario's tree. Filled
  // in by buildTree() and read by the TEST_CASEs below.
  struct Paths {
    fs::path root;
    fs::path local;
    fs::path release;
    fs::path data;
  };

  Paths g_paths;

  void setEnvOrUnset(char const* name, std::string const& value) {
    if (value.empty()) {
      unsetenv(name);
    } else {
      setenv(name, value.c_str(), 1);
    }
  }

  void writeFile(fs::path const& p, std::string const& content = "content\n") {
    fs::create_directories(p.parent_path());
    std::ofstream out(p);
    out << content;
  }

  // Builds the tree needed by the "local" scenario (and reused, with a
  // different environment, by "noRelease" and "sameTops" scenarios):
  //
  //   local/src/Sub/Pack/data/file.txt
  //   local/src/Sub/Pack/data/both.txt
  //   local/src/Sub/Pack/data/subdir/
  //   local/src/Sub/Pack/data/link.txt      -> file.txt
  //   local/src/Sub/Pack/data/dangling.txt  -> nonexistent
  //   release/src/Sub/Pack/data/rel.txt
  //   release/src/Sub/Pack/data/both.txt
  //   data/repo/Sub/Pack/data/dat.txt
  //
  // Note on the "data" layout: CMSSW_DATA_PATH is <root>/data, but the
  // search-path element used to locate data files is <root>/data/repo (one
  // level below), exactly mirroring how the local/release areas use "/src"
  // as the level below CMSSW_BASE/CMSSW_RELEASE_BASE.
  void buildLocalTree(fs::path const& root) {
    writeFile(root / "local/src/Sub/Pack/data/file.txt", "local file\n");
    writeFile(root / "local/src/Sub/Pack/data/both.txt", "local both\n");
    fs::create_directories(root / "local/src/Sub/Pack/data/subdir");
    fs::create_symlink("file.txt", root / "local/src/Sub/Pack/data/link.txt");
    fs::create_symlink("nonexistent", root / "local/src/Sub/Pack/data/dangling.txt");

    writeFile(root / "release/src/Sub/Pack/data/rel.txt", "release file\n");
    writeFile(root / "release/src/Sub/Pack/data/both.txt", "release both\n");

    writeFile(root / "data/repo/Sub/Pack/data/dat.txt", "data file\n");
  }

  // Sets up the environment and tree for one scenario, and returns the
  // resolved Paths for use by the test cases. Only called once, from main(),
  // before any FileInPath is constructed.
  Paths setupScenario(std::string const& scenario) {
    fs::path testRoot = fs::canonical(fs::current_path()) / ("fip_" + scenario);
    fs::remove_all(testRoot);
    fs::create_directories(testRoot);

    Paths paths;
    paths.root = testRoot;
    paths.local = testRoot / "local";
    paths.release = testRoot / "release";
    paths.data = testRoot / "data";

    if (scenario == "local") {
      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH",
                    (paths.local / "src").string() + ":" + (paths.release / "src").string() + ":" +
                        (paths.data / "repo").string());
    } else if (scenario == "noRelease") {
      // CMSSW_RELEASE_BASE unset: getEnvironment() promotes CMSSW_BASE
      // (here still named "local" in the tree, for the shared tree-builder)
      // to be the release top, and clears the local top
      // Files under the "local" tree must therefore report Release.
      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", "");
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", (paths.local / "src").string() + ":" + (paths.data / "repo").string());
    } else if (scenario == "sameTops") {
      // CMSSW_BASE == CMSSW_RELEASE_BASE: getEnvironment() takes the
      // equality branch, which also clears the
      // local top, so files report Release rather than Local.
      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", (paths.local / "src").string() + ":" + (paths.data / "repo").string());
    } else if (scenario == "noData") {
      // CMSSW_DATA_PATH unset, but the data directory is still reachable
      // via the search path. The file is found (locateFile() succeeds),
      // but since dataTop_ is empty, it won't match and will throw an exception.
      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", "");
      setEnvOrUnset("CMSSW_SEARCH_PATH", (paths.data / "repo").string());
    } else if (scenario == "symlink") {
      // real/local/src/Sub/Pack/data/file.txt, reached through an absolute
      // symlink link_local -> real/local. CMSSW_BASE, CMSSW_DATA_PATH and
      // one search-path element are all set via the symlink, to exercise
      // removeSymLinksSrc, removeSymLinks and removeSymLinksTokens
      // respectively.
      fs::path real = testRoot / "real/local";
      writeFile(real / "src/Sub/Pack/data/file.txt", "local file via symlink\n");
      fs::path realData = testRoot / "real/data/repo";
      writeFile(realData / "Sub/Pack/data/dat.txt", "data file via symlink\n");

      fs::path linkLocal = testRoot / "link_local";
      fs::create_directory_symlink(real, linkLocal);
      fs::path linkData = testRoot / "link_data";
      fs::create_directory_symlink(testRoot / "real/data", linkData);
      fs::path linkSearch = testRoot / "link_search";
      fs::create_directory_symlink(realData, linkSearch);

      paths.local = real;  // resolved (real) path, for use by the test cases
      paths.data = testRoot / "real/data";

      // CMSSW_RELEASE_BASE must be set to a *different* top than
      // CMSSW_BASE, otherwise getEnvironment() takes the "no release" or
      // "same tops" promotion branches and clears
      // the local top, which would make files under "real/local" report
      // Release instead of Local.
      setEnvOrUnset("CMSSW_BASE", linkLocal.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", linkData.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", (linkLocal / "src").string() + ":" + linkSearch.string());
    } else if (scenario == "emptySearchPath") {
      // CMSSW_SEARCH_PATH unset: getEnvironment() must throw before any
      // file lookup is attempted.
      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", "");
    } else if (scenario == "lookupDisabled") {
      // Valid environment, but FileInPath::disableFileLookup() is called
      // (in main(), before the Catch2 session runs) so that every ctor
      // becomes a no-op regardless of the environment or the filesystem.
      // The disableFileLookup() is intended to be used in other tests
      // that use FileInPath, but let's test it anyway here
      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH",
                    (paths.local / "src").string() + ":" + (paths.release / "src").string() + ":" +
                        (paths.data / "repo").string());
      edm::FileInPath::disableFileLookup();
    } else if (scenario == "absolutePath") {
      // Latent bug 2 (see file header): std::filesystem::path::operator/=
      // discards the left-hand side when the right-hand side is absolute,
      // so passing an absolute relativePath_ bypasses the search path
      // entirely: whatever the search-path element is, locateFile() ends
      // up testing the absolute path itself. To actually reach bug 1 (the
      // non-terminating branch-path loop) rather than accidentally
      // matching a top on the very first iteration, the search-path
      // element used here ("<root>/unrelated/dir") must not be, nor have
      // as an ancestor, localTop_/releaseTop_/dataTop_ -- so the tops are
      // deliberately set to paths that share no prefix with it.
      fs::create_directories(testRoot / "unrelated/dir");
      fs::create_directories(testRoot / "tops/local");
      fs::create_directories(testRoot / "tops/release");
      fs::create_directories(testRoot / "tops/data");
      writeFile(testRoot / "absolutePath.txt", "file via absolute path\n");
      setEnvOrUnset("CMSSW_BASE", (testRoot / "tops/local").string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", (testRoot / "tops/release").string());
      setEnvOrUnset("CMSSW_DATA_PATH", (testRoot / "tops/data").string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", (testRoot / "unrelated/dir").string());
    } else if (scenario == "emptyPathElement") {
      // Latent bug 3 (see file header): edm::tokenize keeps empty tokens,
      // so CMSSW_SEARCH_PATH=":" tokenizes into two empty path-prefix
      // elements (edm::tokenize("a::b", ":") == {"a", "", "b"}, so
      // tokenize(":", ":") == {"", ""}).
      //
      // This does NOT actually hang, unlike bugs 1, 2 and 4: for an empty
      // path prefix, parent_path() and weakly_canonical() are also empty,
      // so the branch-path loop's condition
      // (!weakly_canonical(br).string().empty()) is already false on the
      // very first check, and the loop runs zero iterations instead of
      // looping forever. Since it never assigns location_ or returns,
      // initialize_() simply moves on to the next path element (also
      // empty here), and once all elements are exhausted, it throws the
      // generic "file not found" exception -- even though locateFile()
      // genuinely found the file relative to the current working
      // directory just before discarding that result.
      //
      // The marker file must be written directly under the process's
      // current working directory (not under testRoot), because an empty
      // path prefix means locateFile()'s "p /= relative" leaves p equal to
      // the bare relative path, which the OS then resolves against the
      // CWD.
      fs::path marker = fs::current_path() / "fip_emptyPathElement_marker.txt";
      writeFile(marker, "found via empty search-path element\n");

      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", ":");
    } else if (scenario == "symlinkAssert") {
      // Latent bug 4 (see file header): $CMSSW_BASE/src is a symlink to a
      // directory not literally named "src" ("notsrc" here). After
      // resolveSymbolicLinks(), the resulting path no longer ends in
      // "/src", so the assert() in removeSymLinksSrc fires (or, built
      // with NDEBUG, substr() silently truncates the path instead).
      fs::path notSrc = testRoot / "local/notsrc";
      writeFile(notSrc / "Sub/Pack/data/file.txt", "local file\n");
      fs::create_directory_symlink(notSrc, testRoot / "local/src");
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", (paths.local / "src").string());
    } else {
      throw std::runtime_error("Unknown scenario: " + scenario);
    }
    return paths;
  }
}  // namespace

TEST_CASE("Default construction", "[local]") {
  edm::FileInPath fip;
  REQUIRE(fip.relativePath().empty());
  REQUIRE(fip.fullPath().empty());
  REQUIRE(fip.location() == edm::FileInPath::Unknown);
}

TEST_CASE("Locate Local file", "[local]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  REQUIRE(fip.location() == edm::FileInPath::Local);
  REQUIRE(fip.fullPath() == (g_paths.local / "src/Sub/Pack/data/file.txt").string());
  REQUIRE(fip.relativePath() == "Sub/Pack/data/file.txt");
}

TEST_CASE("Locate Release file", "[local]") {
  edm::FileInPath fip("Sub/Pack/data/rel.txt");
  REQUIRE(fip.location() == edm::FileInPath::Release);
  REQUIRE(fip.fullPath() == (g_paths.release / "src/Sub/Pack/data/rel.txt").string());
}

TEST_CASE("Locate Data file", "[local]") {
  edm::FileInPath fip("Sub/Pack/data/dat.txt");
  REQUIRE(fip.location() == edm::FileInPath::Data);
  REQUIRE(fip.fullPath() == (g_paths.data / "repo/Sub/Pack/data/dat.txt").string());
}

TEST_CASE("Local shadows Release for duplicate names", "[local]") {
  edm::FileInPath fip("Sub/Pack/data/both.txt");
  REQUIRE(fip.location() == edm::FileInPath::Local);
  REQUIRE(fip.fullPath() == (g_paths.local / "src/Sub/Pack/data/both.txt").string());
}

TEST_CASE("Relative path is normalised", "[local]") {
  edm::FileInPath fip("Sub/Pack/./data/../data/file.txt");
  REQUIRE(fip.relativePath() == "Sub/Pack/data/file.txt");
  REQUIRE(fip.location() == edm::FileInPath::Local);
}

TEST_CASE("const char* ctor matches std::string ctor", "[local]") {
  edm::FileInPath fromChar("Sub/Pack/data/file.txt");
  edm::FileInPath fromString(std::string("Sub/Pack/data/file.txt"));
  REQUIRE(fromChar.location() == fromString.location());
  REQUIRE(fromChar.relativePath() == fromString.relativePath());
  REQUIRE(fromChar.fullPath() == fromString.fullPath());
}

TEST_CASE("Null char* throws", "[local]") {
  char const* nullPath = nullptr;
  REQUIRE_THROWS_AS(edm::FileInPath{nullPath}, edm::Exception);
}

TEST_CASE("Empty relative path throws", "[local]") { REQUIRE_THROWS_AS(edm::FileInPath(""), edm::Exception); }

TEST_CASE("Missing file throws", "[local]") {
  REQUIRE_THROWS_AS(edm::FileInPath("Sub/Pack/data/nope.txt"), edm::Exception);
}

TEST_CASE("Directory as relative path throws", "[local]") {
  REQUIRE_THROWS_WITH(edm::FileInPath("Sub/Pack/data/subdir"),
                      Catch::Matchers::ContainsSubstring("is a directory, not a file"));
}

TEST_CASE("Symlink to file throws", "[local]") {
  REQUIRE_THROWS_WITH(edm::FileInPath("Sub/Pack/data/link.txt"),
                      Catch::Matchers::ContainsSubstring("is a symbolic link, not a file"));
}

TEST_CASE("Dangling symlink is treated as not found", "[local]") {
  // exists() is false for a dangling symlink, so locateFile() reports "not
  // found" and the search continues to the other path elements; since no
  // element has the file, this ends in the generic not-found throw.
  REQUIRE_THROWS_AS(edm::FileInPath("Sub/Pack/data/dangling.txt"), edm::Exception);
}

TEST_CASE("swap exchanges all fields", "[local]") {
  edm::FileInPath a("Sub/Pack/data/file.txt");
  edm::FileInPath b("Sub/Pack/data/rel.txt");

  std::string aRel = a.relativePath(), aFull = a.fullPath();
  auto aLoc = a.location();
  std::string bRel = b.relativePath(), bFull = b.fullPath();
  auto bLoc = b.location();

  a.swap(b);
  REQUIRE(a.relativePath() == bRel);
  REQUIRE(a.fullPath() == bFull);
  REQUIRE(a.location() == bLoc);
  REQUIRE(b.relativePath() == aRel);
  REQUIRE(b.fullPath() == aFull);
  REQUIRE(b.location() == aLoc);

  // Free swap function: swap back and check again.
  using edm::swap;
  swap(a, b);
  REQUIRE(a.relativePath() == aRel);
  REQUIRE(b.relativePath() == bRel);

  // write() round-trip after swap, to cover the private tops too.
  std::ostringstream os;
  a.write(os);
  REQUIRE_FALSE(os.str().empty());
}

TEST_CASE("operator== compares location and relativePath only", "[local]") {
  edm::FileInPath a("Sub/Pack/data/file.txt");
  edm::FileInPath b("Sub/Pack/data/file.txt");
  REQUIRE(a == b);
}

TEST_CASE("write/read round-trip", "[local]") {
  SECTION("Local") {
    edm::FileInPath fip("Sub/Pack/data/file.txt");
    std::ostringstream os;
    fip.write(os);
    REQUIRE(os.str().find("V001") != std::string::npos);
    REQUIRE(os.str().find(g_paths.local.string()) == std::string::npos);

    edm::FileInPath fip2;
    std::istringstream is(os.str());
    fip2.read(is);
    REQUIRE(fip2.relativePath() == fip.relativePath());
    REQUIRE(fip2.location() == fip.location());
    REQUIRE(fip2.fullPath() == fip.fullPath());
  }
  SECTION("Release") {
    edm::FileInPath fip("Sub/Pack/data/rel.txt");
    std::ostringstream os;
    fip.write(os);
    REQUIRE(os.str().find("V001") != std::string::npos);
    REQUIRE(os.str().find(g_paths.release.string()) == std::string::npos);

    edm::FileInPath fip2;
    std::istringstream is(os.str());
    fip2.read(is);
    REQUIRE(fip2.fullPath() == fip.fullPath());
  }
  SECTION("Data") {
    edm::FileInPath fip("Sub/Pack/data/dat.txt");
    std::ostringstream os;
    fip.write(os);
    REQUIRE(os.str().find("V001") != std::string::npos);
    REQUIRE(os.str().find(g_paths.data.string()) == std::string::npos);

    edm::FileInPath fip2;
    std::istringstream is(os.str());
    fip2.read(is);
    REQUIRE(fip2.fullPath() == fip.fullPath());
  }
  SECTION("Unknown (default constructed)") {
    edm::FileInPath fip;
    std::ostringstream os;
    fip.write(os);
    REQUIRE(os.str() == "V001 @ 0");

    edm::FileInPath fip2;
    std::istringstream is(os.str());
    fip2.read(is);
    REQUIRE(fip2.relativePath().empty());
    REQUIRE(fip2.location() == edm::FileInPath::Unknown);
  }
}

TEST_CASE("read() of legacy pre-CMSSW_1_5_0_pre3 format", "[local]") {
  SECTION("Release, with BASE placeholder") {
    std::istringstream is("Sub/Pack/data/rel.txt 0 BASE/src/Sub/Pack/data/rel.txt");
    edm::FileInPath fip;
    fip.read(is);
    REQUIRE(fip.location() == edm::FileInPath::Release);
    REQUIRE(fip.relativePath() == "Sub/Pack/data/rel.txt");
    REQUIRE(fip.fullPath() == g_paths.release.string() + "/src/Sub/Pack/data/rel.txt");
  }
  SECTION("Local, path used verbatim") {
    std::istringstream is("Sub/Pack/data/file.txt 1 /some/verbatim/path.txt");
    edm::FileInPath fip;
    fip.read(is);
    REQUIRE(fip.location() == edm::FileInPath::Local);
    REQUIRE(fip.relativePath() == "Sub/Pack/data/file.txt");
    REQUIRE(fip.fullPath() == "/some/verbatim/path.txt");
  }
}

TEST_CASE("read() from a failing stream leaves the object untouched", "[local]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  std::string const relBefore = fip.relativePath();
  std::string const fullBefore = fip.fullPath();
  auto const locBefore = fip.location();

  std::istringstream is;
  is.setstate(std::ios::failbit);
  fip.read(is);

  REQUIRE(fip.relativePath() == relBefore);
  REQUIRE(fip.fullPath() == fullBefore);
  REQUIRE(fip.location() == locBefore);
}

TEST_CASE("operator<< and operator>> match write/read", "[local]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  std::ostringstream os;
  os << fip;
  std::ostringstream osWrite;
  fip.write(osWrite);
  REQUIRE(os.str() == osWrite.str());

  edm::FileInPath fip2;
  std::istringstream is(os.str());
  is >> fip2;
  REQUIRE(fip2.fullPath() == fip.fullPath());
}

TEST_CASE("findFile", "[local]") {
  SECTION("existing file returns absolute path") {
    std::string found = edm::FileInPath::findFile("Sub/Pack/data/file.txt");
    REQUIRE(found == (g_paths.local / "src/Sub/Pack/data/file.txt").string());
  }
  SECTION("missing file returns empty string, does not throw") {
    std::string found = edm::FileInPath::findFile("Sub/Pack/data/nope.txt");
    REQUIRE(found.empty());
  }
  SECTION("directory still throws") {
    REQUIRE_THROWS_WITH(edm::FileInPath::findFile("Sub/Pack/data/subdir"),
                        Catch::Matchers::ContainsSubstring("is a directory, not a file"));
  }
  SECTION("symlink still throws") {
    REQUIRE_THROWS_WITH(edm::FileInPath::findFile("Sub/Pack/data/link.txt"),
                        Catch::Matchers::ContainsSubstring("is a symbolic link, not a file"));
  }
}

TEST_CASE("noRelease: local files report Release", "[noRelease]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  REQUIRE(fip.location() == edm::FileInPath::Release);
  REQUIRE(fip.fullPath() == (g_paths.local / "src/Sub/Pack/data/file.txt").string());
}

TEST_CASE("noRelease: write succeeds and strips the release top", "[noRelease]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  std::ostringstream os;
  fip.write(os);
  REQUIRE(os.str().find(g_paths.local.string()) == std::string::npos);
}

TEST_CASE("sameTops: local files report Release", "[sameTops]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  REQUIRE(fip.location() == edm::FileInPath::Release);
  REQUIRE(fip.fullPath() == (g_paths.local / "src/Sub/Pack/data/file.txt").string());
}

TEST_CASE("sameTops: write succeeds and strips the release top", "[sameTops]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  std::ostringstream os;
  fip.write(os);
  REQUIRE(os.str().find(g_paths.local.string()) == std::string::npos);
}

TEST_CASE("symlink: fullPath resolves through CMSSW_BASE symlink", "[symlink]") {
  // g_paths.local is the resolved "real/local" path (set in setupScenario);
  // fullPath() must match it, proving removeSymLinksSrc stripped the
  // "link_local" symlink from CMSSW_BASE before comparing branch paths.
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  REQUIRE(fip.location() == edm::FileInPath::Local);
  REQUIRE(fip.fullPath() == (g_paths.local / "src/Sub/Pack/data/file.txt").string());
}

TEST_CASE("symlink: write succeeds through the resolved local top", "[symlink]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  std::ostringstream os;
  fip.write(os);
  REQUIRE(os.str().find("V001") != std::string::npos);
  REQUIRE(os.str().find(g_paths.local.string()) == std::string::npos);
}

TEST_CASE("symlink: data file resolves through CMSSW_DATA_PATH and search-path symlinks", "[symlink]") {
  // Exercises both removeSymLinks (CMSSW_DATA_PATH, no "/src" suffix) and
  // removeSymLinksTokens (the "link_search" search-path element).
  edm::FileInPath fip("Sub/Pack/data/dat.txt");
  REQUIRE(fip.location() == edm::FileInPath::Data);
  REQUIRE(fip.fullPath() == (g_paths.data / "repo/Sub/Pack/data/dat.txt").string());
}

TEST_CASE("emptySearchPath: every ctor throws naming CMSSW_SEARCH_PATH", "[emptySearchPath]") {
  auto matchesSearchPath = Catch::Matchers::ContainsSubstring("CMSSW_SEARCH_PATH");
  SECTION("default ctor") { REQUIRE_THROWS_WITH(edm::FileInPath{}, matchesSearchPath); }
  SECTION("std::string ctor") {
    REQUIRE_THROWS_WITH(edm::FileInPath(std::string("Sub/Pack/data/file.txt")), matchesSearchPath);
  }
  SECTION("const char* ctor") { REQUIRE_THROWS_WITH(edm::FileInPath("Sub/Pack/data/file.txt"), matchesSearchPath); }
}

TEST_CASE("emptySearchPath: findFile returns empty", "[emptySearchPath]") {
  REQUIRE(edm::FileInPath::findFile("Sub/Pack/data/file.txt").empty());
}

TEST_CASE("lookupDisabled: ctors succeed without touching the filesystem", "[lookupDisabled]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  REQUIRE(fip.location() == edm::FileInPath::Unknown);
  REQUIRE(fip.fullPath().empty());
  REQUIRE(fip.relativePath() == "Sub/Pack/data/file.txt");
}

TEST_CASE("lookupDisabled: default ctor is also a no-op", "[lookupDisabled]") {
  edm::FileInPath fip;
  REQUIRE(fip.location() == edm::FileInPath::Unknown);
  REQUIRE(fip.fullPath().empty());
  REQUIRE(fip.relativePath().empty());
}

TEST_CASE("lookupDisabled: null char* does not throw", "[lookupDisabled]") {
  // Unlike in a normal environment, the disabled-lookup check in the
  // char const* ctor precedes the null check (FileInPath.cc:120-124), so
  // no exception is thrown; relativePath() ends up empty.
  char const* nullPath = nullptr;
  edm::FileInPath fip(nullPath);
  REQUIRE(fip.relativePath().empty());
  REQUIRE(fip.location() == edm::FileInPath::Unknown);
}

TEST_CASE("noData: file is found CMSSW_SEARCH_PATH, but element is not in any of the known search areas", "[noData]") {
  REQUIRE_THROWS_WITH(
      edm::FileInPath("Sub/Pack/data/dat.txt"),
      Catch::Matchers::ContainsSubstring("edm::FileInPath found file Sub/Pack/data/dat.txt in search path element") &&
          Catch::Matchers::ContainsSubstring("but that element is not in any of the known search"));
}

// --- latent bug 2 scenario: absolutePath ---
TEST_CASE("absolutePath: absolute relativePath_ bypasses the search path and hangs", "[absolutePath]") {
  // Any absolute path that exists is enough to reach locateFile()'s
  // exists() check; /etc/passwd is used only as an example of an absolute
  // path that is virtually always present on a POSIX system used to build
  // and run CMSSW.
  //edm::FileInPath fip("/etc/passwd");
  edm::FileInPath fip((fs::current_path() / "fip_absolutePath/absolutePath.txt").string());
  (void)fip;
}

// --- latent bug 3 scenario: emptyPathElement ---
TEST_CASE("emptyPathElement: file found via CWD is nonetheless reported as not found", "[emptyPathElement]") {
  REQUIRE_THROWS_AS(edm::FileInPath("fip_emptyPathElement_marker.txt"), edm::Exception);
}

// --- latent bug 4 scenario: symlinkAssert ---
TEST_CASE("symlinkAssert: CMSSW_BASE/src symlinked to a non-'src' directory aborts", "[symlinkAssert]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  (void)fip;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << (argc > 0 ? argv[0] : "testFWCoreUtilitiesFileInPath") << " <scenario>\n";
    return 1;
  }
  std::string const scenario = argv[1];

  g_paths = setupScenario(scenario);

  Catch::Session session;
  session.configData().testsOrTags = {"[" + scenario + "]"};
  return session.run();
}
