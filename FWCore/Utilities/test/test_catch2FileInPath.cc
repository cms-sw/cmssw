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

// Known issues in edm::FileInPath exercised (but not fixed) by these tests.
// A test that references an issue number below asserts the CURRENT behaviour;
// if the issue is fixed, update the test in the same commit.
//
//  3. Defect: locateFile() is called with the RAW relative path
//     (FileInPath.cc:406) and normalisation happens only afterwards
//     (FileInPath.cc:408), while the location test inspects the search-path
//     element rather than the resolved file (FileInPath.cc:419-431). A relative
//     path with enough ".." components therefore names a file outside every
//     top, yet is still classified and serialised.
//  4. Defect: read() assigns location_ (FileInPath.cc:220) before the stream
//     state is checked (FileInPath.cc:230), so a truncated-but-non-empty record
//     leaves the object internally inconsistent rather than untouched. Same in
//     readFromParameterSetBlob() (FileInPath.cc:291 vs :298).
//  5. Defect: the not-found message re-reads the LIVE environment with
//     std::getenv (FileInPath.cc:449) instead of the cached searchPath(). If
//     CMSSW_SEARCH_PATH was unset after the search path was cached, this is a
//     null char const* insertion, which sets badbit on the exception's stream
//     and silently truncates the rest of the message.
//  7. Inconsistency: write() uses a raw string-prefix test
//     (canonicalFilename_.find(top) != 0, FileInPath.cc:168/:179/:190) while
//     initialize_() uses the component-wise pathBeginsWith(). The two disagree
//     for sibling directories with a shared string prefix, so a record naming a
//     file under "<CMSSW_BASE>2/..." round-trips as Local.
//  8. Documented limitation (FileInPath.h:48-52): paths containing spaces are
//     not supported. write() succeeds, but read() cannot parse the result back.
//  9. Asymmetry: readFromParameterSetBlob() degrades gracefully for a missing
//     local or release top (@LOCAL / @RELEASE, FileInPath.cc:302-311) but still
//     throws for a missing data top (FileInPath.cc:325-327).
// 10. Defect (worse than Issue 4): for a truncated-but-non-empty record, the
//     failed extraction into "int loc" (FileInPath.cc:219) or "bool local"
//     (FileInPath.cc:213/:284) leaves that local variable UNINITIALISED, not
//     value-initialised to 0 as one might expect: the preceding extraction
//     already left the stream at eofbit (not failbit), so the sentry for the
//     next ">>" fails before std::num_get/do_get ever runs and the output
//     argument is left untouched. location_ is then set from this
//     indeterminate value, which is itself UB to read back via location()
//     (confirmed with UBSan: "load of value ..., which is not a valid value
//     for type 'LocationCode'"). Do not assert a specific location_/loc value
//     for these cases; only assert what is well-defined (whether
//     relativePath_ was overwritten, and that read()/readFromParameterSetBlob()
//     report failure via the stream state).

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

  // Warning emitted by the very first getEnvironment() call, captured in
  // main() before the Catch2 session runs (see the comment there).
  std::string g_firstConstructionStderr;

  // Saves and restores the value of an environment variable across the
  // lifetime of the guard, even if the variable is mutated (or unset) in
  // between. Used by tests that must leave the environment as they found
  // it, even when a REQUIRE fails partway through.
  class EnvGuard {
  public:
    explicit EnvGuard(char const* name)
        : name_(name), had_(std::getenv(name) != nullptr), old_(had_ ? std::getenv(name) : "") {}
    ~EnvGuard() {
      if (had_) {
        setenv(name_, old_.c_str(), 1);
      } else {
        unsetenv(name_);
      }
    }
    EnvGuard(EnvGuard const&) = delete;
    EnvGuard& operator=(EnvGuard const&) = delete;

  private:
    char const* name_;
    bool had_;
    std::string old_;
  };

  // Swaps std::cerr's streambuf for the lifetime of the object and exposes
  // what was written to it. Note that std::osyncstream in getEnvironment()
  // captures cerr's rdbuf at construction time, so the CerrCapture must be
  // constructed before the FileInPath that may trigger the warning.
  class CerrCapture {
  public:
    CerrCapture() : old_(std::cerr.rdbuf(buf_.rdbuf())) {}
    ~CerrCapture() { std::cerr.rdbuf(old_); }
    CerrCapture(CerrCapture const&) = delete;
    CerrCapture& operator=(CerrCapture const&) = delete;
    std::string str() const { return buf_.str(); }

  private:
    std::ostringstream buf_;
    std::streambuf* old_;
  };

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
  //   outside.txt                                (outside every area)
  //   local/src/Sub/Pack/data/with space.txt
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

    // Outside every area; used by the findFile()/".." escape tests (Issues 1, 3).
    writeFile(root / "outside.txt", "outside every top\n");

    // Used by the paths-with-spaces test (Issue 8).
    writeFile(root / "local/src/Sub/Pack/data/with space.txt", "local file with space\n");
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
    } else if (scenario == "noTops") {
      // CMSSW_BASE and CMSSW_RELEASE_BASE both unset: getEnvironment() takes
      // the "no release" promotion branch (releaseTop_ = localTop_, both
      // empty), then the "same tops" branch (localTop_ cleared again, a
      // no-op here), leaving both localTop_ and releaseTop_ empty while
      // dataTop_ is set. The data search-path element begins with dataTop_,
      // so no warning is emitted.
      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", "");
      setEnvOrUnset("CMSSW_RELEASE_BASE", "");
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", (paths.data / "repo").string());
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
      // Absolute path leads to an exception
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
      // edm::tokenize keeps empty tokens, so CMSSW_SEARCH_PATH=":" tokenizes into two empty path-prefix elements
      // (edm::tokenize("a::b", ":") == {"a", "", "b"}, so tokenize(":", ":") == {"", ""}).
      //
      // E.g. LD_LIBRARY_PATH an empty element is treated as "current working directory", and this behavior comes out
      // naturally with std::filesystem as well. The empty search element does not match to any of the
      // {local,release,data} areas, so it gets treated as "not found".
      //
      // The marker file must be written directly under the process's current working directory (not under testRoot),
      // because an empty path prefix means locateFile()'s "p /= relative" leaves p equal to the bare relative path,
      // which the OS then resolves against the CWD.
      fs::path marker = fs::current_path() / "fip_emptyPathElement_marker.txt";
      writeFile(marker, "found via empty search-path element\n");

      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", ":");
    } else if (scenario == "strayPathElement") {
      // Covers the warning added in commit 53065377 and the component-wise
      // prefix test added in 4c834e9. "local2" string-starts-with "local"
      // but is a sibling directory, not a subdirectory: pathBeginsWith()
      // must reject it even though a raw string-prefix test would accept it.
      buildLocalTree(testRoot);
      writeFile(testRoot / "local2/src/Sub/Pack/data/sib.txt", "sibling file\n");
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH",
                    (paths.local / "src").string() + ":" + (testRoot / "local2/src").string() + ":" +
                        (paths.release / "src").string() + ":" + (paths.data / "repo").string());
    } else if (scenario == "elementIsTop") {
      // The configuration the old parent_path()-based loop could never
      // match (it started at parent_path(), so a top could never equal an
      // element) - the exact hang fixed by 4c834e9. CMSSW_DATA_PATH is set
      // equal to the sole search-path element, instead of one level above it
      // as buildLocalTree()'s regular layout does.
      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string());
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", (paths.data / "repo").string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", (paths.data / "repo").string());
    } else if (scenario == "trailingSlashTop") {
      // A trailing '/' in CMSSW_BASE works
      buildLocalTree(testRoot);
      setEnvOrUnset("CMSSW_BASE", paths.local.string() + "/");
      setEnvOrUnset("CMSSW_RELEASE_BASE", paths.release.string());
      setEnvOrUnset("CMSSW_DATA_PATH", paths.data.string());
      setEnvOrUnset("CMSSW_SEARCH_PATH", (paths.local / "src").string());
    } else if (scenario == "symlinkException") {
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

TEST_CASE("local: no warning in a well-formed environment", "[local]") { REQUIRE(g_firstConstructionStderr.empty()); }

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

TEST_CASE("Relative path with .. escapes every top", "[local]") {
  // Issue 3: locateFile() is called with the RAW relative path, and the
  // location test inspects the search-path element (which is inside
  // localTop_) rather than the resolved file. A relative path with enough
  // ".." components therefore names a file outside every top, yet is still
  // classified as Local and serialised without complaint.
  edm::FileInPath fip("Sub/Pack/../../../../outside.txt");

  REQUIRE(fip.relativePath() == "../../outside.txt");  // lexically_normal
  // std::filesystem::absolute() does not normalise away the ".." components.
  REQUIRE(fip.fullPath() == (g_paths.local / "src/../../outside.txt").string());
  REQUIRE(fip.location() == edm::FileInPath::Local);

  std::ostringstream os;
  fip.write(os);
  REQUIRE(os.str() == "V001 ../../outside.txt 1 /src/../../outside.txt");

  // The clean round-trip is the point: nothing detects the escape.
  edm::FileInPath fip2;
  std::istringstream is(os.str());
  fip2.read(is);
  REQUIRE(fip2.fullPath() == fip.fullPath());
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

  // D2: a and b share identical tops (both come from the same environment),
  // so a plain field comparison after the swap would not actually exercise
  // the private tops. Capture full write() output beforehand and require it
  // survives the swap intact - this is what proves the tops moved too.
  std::ostringstream bWriteBefore;
  b.write(bWriteBefore);

  a.swap(b);
  REQUIRE(a.relativePath() == bRel);
  REQUIRE(a.fullPath() == bFull);
  REQUIRE(a.location() == bLoc);
  REQUIRE(b.relativePath() == aRel);
  REQUIRE(b.fullPath() == aFull);
  REQUIRE(b.location() == aLoc);

  std::ostringstream aWriteAfter;
  a.write(aWriteAfter);
  REQUIRE(aWriteAfter.str() == bWriteBefore.str());

  // Self-swap must leave the object unchanged.
  a.swap(a);
  REQUIRE(a.relativePath() == bRel);
  REQUIRE(a.fullPath() == bFull);
  REQUIRE(a.location() == bLoc);

  // Free swap function: swap back and check again.
  using edm::swap;
  swap(a, b);
  REQUIRE(a.relativePath() == aRel);
  REQUIRE(b.relativePath() == bRel);
}

TEST_CASE("operator== compares location and relativePath only", "[local]") {
  edm::FileInPath a("Sub/Pack/data/file.txt");

  SECTION("identical objects are equal") {
    edm::FileInPath b("Sub/Pack/data/file.txt");
    REQUIRE(a == b);
  }
  SECTION("equal across different fullPath - proves only location and relativePath are compared") {
    // Same relativePath and location (Local) as "a", but a completely
    // different fullPath (the legacy verbatim-path form).
    edm::FileInPath b;
    std::istringstream is("Sub/Pack/data/file.txt 1 /some/verbatim/path.txt");
    b.read(is);
    REQUIRE(a.fullPath() != b.fullPath());
    REQUIRE(a == b);
  }
  SECTION("not equal for the same relativePath but a different location") {
    edm::FileInPath b;
    std::istringstream is("V001 Sub/Pack/data/file.txt 2 /somewhere/else.txt");
    b.read(is);
    REQUIRE(a.relativePath() == b.relativePath());
    REQUIRE(a.location() != b.location());
    REQUIRE(a != b);
  }
  SECTION("not equal for the same location but a different relativePath") {
    edm::FileInPath b("Sub/Pack/data/both.txt");
    REQUIRE(a.location() == b.location());
    REQUIRE(a.relativePath() != b.relativePath());
    REQUIRE(a != b);
  }
}

TEST_CASE("copy construction and copy assignment", "[local]") {
  // D4: the copies are implicit (compiler-generated), but swap() is
  // hand-written, so pin copy semantics explicitly. Verify via
  // relativePath()/location()/fullPath() and a write() round-trip, so the
  // private tops are included too.
  edm::FileInPath original("Sub/Pack/data/file.txt");
  std::ostringstream originalWrite;
  original.write(originalWrite);

  SECTION("copy construction") {
    edm::FileInPath copy(original);
    REQUIRE(copy.relativePath() == original.relativePath());
    REQUIRE(copy.location() == original.location());
    REQUIRE(copy.fullPath() == original.fullPath());
    std::ostringstream os;
    copy.write(os);
    REQUIRE(os.str() == originalWrite.str());
  }
  SECTION("copy assignment") {
    edm::FileInPath assigned;
    assigned = original;
    REQUIRE(assigned.relativePath() == original.relativePath());
    REQUIRE(assigned.location() == original.location());
    REQUIRE(assigned.fullPath() == original.fullPath());
    std::ostringstream os;
    assigned.write(os);
    REQUIRE(os.str() == originalWrite.str());
  }
  SECTION("self-assignment leaves the object unchanged") {
    edm::FileInPath self("Sub/Pack/data/file.txt");
    self = self;
    REQUIRE(self.relativePath() == "Sub/Pack/data/file.txt");
    REQUIRE(self.location() == edm::FileInPath::Local);
    std::ostringstream os;
    self.write(os);
    REQUIRE(os.str() == originalWrite.str());
  }
}

TEST_CASE("environment is read only once per process", "[local]") {
  // D5: pins FileInPath.h:53-55 - all environment variables are read only
  // once, when the object is constructed (in practice, cached in
  // function-local statics on first use). The EnvGuard restore must survive
  // a failing REQUIRE, hence RAII rather than a manual setenv at the end.
  SECTION("CMSSW_SEARCH_PATH mutation after the first construction has no effect") {
    EnvGuard guard("CMSSW_SEARCH_PATH");
    edm::FileInPath first("Sub/Pack/data/file.txt");
    setenv("CMSSW_SEARCH_PATH", "/nonexistent", 1);
    edm::FileInPath second("Sub/Pack/data/file.txt");
    REQUIRE(second.fullPath() == first.fullPath());
  }
  SECTION("CMSSW_BASE mutation after the first construction has no effect") {
    EnvGuard guard("CMSSW_BASE");
    edm::FileInPath first("Sub/Pack/data/file.txt");
    setenv("CMSSW_BASE", "/nonexistent", 1);
    edm::FileInPath second("Sub/Pack/data/file.txt");
    REQUIRE(second.location() == first.location());
  }
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
  // This only covers the early return at FileInPath.cc:207-208 (failure while
  // extracting the version token, before location_ is ever touched); it does
  // not generalise to a stream that fails partway through a record - see
  // "read() of a truncated record" below.
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

TEST_CASE("read() of a truncated record leaves relativePath/fullPath unchanged but corrupts location_", "[local]") {
  // Issue 4 (read() assigns location_ before the stream state is checked) and
  // Issue 10 (that assignment reads an uninitialised local when the failure
  // happens on the extraction feeding it). We deliberately do NOT assert a
  // specific location() value in these cases: unlike the plan's original
  // assumption, the failed extraction leaves "int loc" / "bool local"
  // (FileInPath.cc:218/:212) genuinely uninitialised rather than
  // value-initialised to 0 - the preceding extraction already left the stream
  // at eofbit (not failbit), so std::num_get::do_get's sentry fails before
  // touching the output argument at all. location_ is then set from
  // indeterminate stack contents, which is UB to even read back (confirmed
  // with UBSan: "load of value ..., which is not a valid value for type
  // 'LocationCode'"), and differs between optimisation levels in practice.
  // What IS well-defined, and what these tests pin, is that relativePath_ and
  // canonicalFilename_ are left exactly as they were: both branches of read()
  // only reach "relativePath_ = relname;" (FileInPath.cc:232) after the
  // "if (!is) return;" check at FileInPath.cc:230, so a failure anywhere
  // before that point skips the assignment entirely.
  SECTION("current format, missing location field") {
    edm::FileInPath fip("Sub/Pack/data/file.txt");
    std::string const relBefore = fip.relativePath();
    std::string const fullBefore = fip.fullPath();

    std::istringstream is("V001 Sub/Pack/data/file.txt");
    fip.read(is);

    REQUIRE(is.fail());
    REQUIRE(fip.relativePath() == relBefore);
    REQUIRE(fip.fullPath() == fullBefore);
  }
  SECTION("legacy format, bare relative path only") {
    edm::FileInPath fip("Sub/Pack/data/file.txt");
    std::string const relBefore = fip.relativePath();
    std::string const fullBefore = fip.fullPath();

    std::istringstream is("someRelPath");
    fip.read(is);

    REQUIRE(is.fail());
    REQUIRE(fip.relativePath() == relBefore);
    REQUIRE(fip.fullPath() == fullBefore);
  }
  SECTION("readFromParameterSetBlob equivalent of the current-format case") {
    edm::FileInPath fip;
    std::string const relBefore = fip.relativePath();
    std::string const fullBefore = fip.fullPath();

    std::istringstream is("V001 Sub/Pack/data/file.txt");
    fip.readFromParameterSetBlob(is);

    REQUIRE(is.fail());
    REQUIRE(fip.relativePath() == relBefore);
    REQUIRE(fip.fullPath() == fullBefore);
  }
}

TEST_CASE("readFromParameterSetBlob matches read for well-formed records", "[local]") {
  // readFromParameterSetBlob() is the production deserialisation path
  // (FWCore/ParameterSet/src/types.cc, edm::decode(FileInPath&, ...)); check
  // that it agrees with read() whenever both localTop_ and releaseTop_ are
  // non-empty (the "local" scenario).
  auto compareReadAndBlob = [](std::string const& record) {
    edm::FileInPath viaRead;
    std::istringstream isRead(record);
    viaRead.read(isRead);

    edm::FileInPath viaBlob;
    std::istringstream isBlob(record);
    viaBlob.readFromParameterSetBlob(isBlob);

    REQUIRE(viaRead.location() == viaBlob.location());
    REQUIRE(viaRead.relativePath() == viaBlob.relativePath());
    REQUIRE(viaRead.fullPath() == viaBlob.fullPath());
  };

  SECTION("Local") {
    edm::FileInPath fip("Sub/Pack/data/file.txt");
    std::ostringstream os;
    fip.write(os);
    compareReadAndBlob(os.str());
  }
  SECTION("Release") {
    edm::FileInPath fip("Sub/Pack/data/rel.txt");
    std::ostringstream os;
    fip.write(os);
    compareReadAndBlob(os.str());
  }
  SECTION("Data") {
    edm::FileInPath fip("Sub/Pack/data/dat.txt");
    std::ostringstream os;
    fip.write(os);
    compareReadAndBlob(os.str());
  }
  SECTION("Unknown (default constructed)") { compareReadAndBlob("V001 @ 0"); }
  SECTION("legacy Release, BASE placeholder replaced") {
    compareReadAndBlob("Sub/Pack/data/rel.txt 0 BASE/src/Sub/Pack/data/rel.txt");
  }
  SECTION("legacy Release, pre-CMSSW_1_2_0_pre2 verbatim path") {
    compareReadAndBlob("Sub/Pack/data/rel.txt 0 /some/verbatim/rel/path.txt");
  }
  SECTION("legacy Local, path used verbatim") {
    compareReadAndBlob("Sub/Pack/data/file.txt 1 /some/verbatim/path.txt");
  }
}

TEST_CASE("write of a legacy-read object rejects a foreign path", "[local]") {
  SECTION("Local, verbatim path outside the local top") {
    edm::FileInPath fip;
    std::istringstream is("Sub/Pack/data/file.txt 1 /some/verbatim/path.txt");
    fip.read(is);
    std::ostringstream os;
    REQUIRE_THROWS_WITH(fip.write(os), Catch::Matchers::ContainsSubstring("is not in the local release area"));
  }
  SECTION("Release, verbatim path outside the release top") {
    edm::FileInPath fip;
    std::istringstream is("Sub/Pack/data/rel.txt 0 /some/verbatim/rel/path.txt");
    fip.read(is);
    std::ostringstream os;
    REQUIRE_THROWS_WITH(fip.write(os), Catch::Matchers::ContainsSubstring("is not in the base release area"));
  }
}

TEST_CASE("write to a bad stream leaves the stream bad", "[local]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  std::ostringstream os;
  os.setstate(std::ios::badbit);
  fip.write(os);
  REQUIRE(os.bad());
  REQUIRE(os.str().empty());
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
  SECTION("absolute path throws an exception") {
    std::string const outside = (g_paths.root / "outside.txt").string();
    REQUIRE_THROWS_WITH(edm::FileInPath::findFile(outside),
                        Catch::Matchers::ContainsSubstring("The path must be relative, not absolute:"));
  }
  SECTION("empty path throws an exception") {
    REQUIRE_THROWS_WITH(edm::FileInPath::findFile(""),
                        Catch::Matchers::ContainsSubstring("Relative path must not be empty"));
  }
}

TEST_CASE("not-found message re-reads the live environment", "[local]") {
  // Issue 5: the not-found message re-reads the LIVE environment with
  // std::getenv (FileInPath.cc:449) instead of the cached searchPath(). Once
  // CMSSW_SEARCH_PATH is unset, std::getenv() returns a null char const*;
  // operator<<(std::ostream&, char const*) sets badbit on that (libstdc++
  // ostream:669-678), so everything appended after the getenv() call is
  // silently dropped from the exception message. The EnvGuard restore must
  // survive a failing REQUIRE below, hence RAII rather than a manual setenv
  // at the end.
  EnvGuard guard("CMSSW_SEARCH_PATH");
  edm::FileInPath warm("Sub/Pack/data/file.txt");  // ensure searchPath() is cached
  unsetenv("CMSSW_SEARCH_PATH");

  // Lookups themselves are unaffected because the search path is cached in a
  // static; only the diagnostic message is corrupted.
  REQUIRE_THROWS_WITH(edm::FileInPath("Sub/Pack/data/nope.txt"),
                      Catch::Matchers::ContainsSubstring("unable to find file") &&
                          Catch::Matchers::ContainsSubstring("${CMSSW_SEARCH_PATH} is: ") &&
                          !Catch::Matchers::ContainsSubstring("Current directory is"));
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

TEST_CASE("noRelease: read() of a Local record throws, readFromParameterSetBlob degrades to @LOCAL", "[noRelease]") {
  // localTop_ is cleared by the "no release" promotion branch (see
  // setupScenario's comment for this scenario), so read() cannot resolve a
  // Local record...
  std::string const record = "V001 Sub/Pack/data/file.txt 1 /src/Sub/Pack/data/file.txt";

  edm::FileInPath viaRead;
  std::istringstream isRead(record);
  REQUIRE_THROWS_WITH(viaRead.read(isRead), Catch::Matchers::ContainsSubstring("CMSSW_BASE"));

  // ...while readFromParameterSetBlob() substitutes the "@LOCAL" placeholder
  // for the missing top and succeeds.
  edm::FileInPath viaBlob;
  std::istringstream isBlob(record);
  viaBlob.readFromParameterSetBlob(isBlob);
  REQUIRE(viaBlob.location() == edm::FileInPath::Local);
  REQUIRE(viaBlob.fullPath() == "@LOCAL/src/Sub/Pack/data/file.txt");
}

TEST_CASE("noTops: sanity - a Data file still resolves normally", "[noTops]") {
  edm::FileInPath fip("Sub/Pack/data/dat.txt");
  REQUIRE(fip.location() == edm::FileInPath::Data);
  REQUIRE(fip.fullPath() == (g_paths.data / "repo/Sub/Pack/data/dat.txt").string());
}

TEST_CASE("noTops: no warning is emitted", "[noTops]") {
  // The single search-path element begins with dataTop_, so
  // getEnvironment()'s "not in any of the known search areas" warning is not
  // triggered, even though localTop_ and releaseTop_ are both empty.
  REQUIRE(g_firstConstructionStderr.empty());
}

TEST_CASE("noTops: read() of a Release record throws, readFromParameterSetBlob degrades to @RELEASE", "[noTops]") {
  std::string const record = "V001 Sub/Pack/data/rel.txt 2 /src/Sub/Pack/data/rel.txt";

  edm::FileInPath viaRead;
  std::istringstream isRead(record);
  REQUIRE_THROWS_WITH(viaRead.read(isRead), Catch::Matchers::ContainsSubstring("CMSSW_RELEASE_BASE"));

  edm::FileInPath viaBlob;
  std::istringstream isBlob(record);
  viaBlob.readFromParameterSetBlob(isBlob);
  REQUIRE(viaBlob.location() == edm::FileInPath::Release);
  REQUIRE(viaBlob.fullPath() == "@RELEASE/src/Sub/Pack/data/rel.txt");
}

TEST_CASE("noTops: read() of a Local record throws, readFromParameterSetBlob degrades to @LOCAL", "[noTops]") {
  std::string const record = "V001 Sub/Pack/data/file.txt 1 /src/Sub/Pack/data/file.txt";

  edm::FileInPath viaRead;
  std::istringstream isRead(record);
  REQUIRE_THROWS_WITH(viaRead.read(isRead), Catch::Matchers::ContainsSubstring("CMSSW_BASE"));

  edm::FileInPath viaBlob;
  std::istringstream isBlob(record);
  viaBlob.readFromParameterSetBlob(isBlob);
  REQUIRE(viaBlob.location() == edm::FileInPath::Local);
  REQUIRE(viaBlob.fullPath() == "@LOCAL/src/Sub/Pack/data/file.txt");
}

TEST_CASE("noTops: swap exchanges the private tops between objects with different tops set", "[noTops]") {
  // D2: in the "local" scenario, a and b share identical tops, so swap()
  // never actually has to move different top strings around. Here x gets
  // releaseTop_ == "@RELEASE" (via readFromParameterSetBlob(), since
  // releaseTop_ is empty in this scenario) and y gets dataTop_ set (via the
  // normal ctor), with releaseTop_ empty - this is the only way to observe
  // the private tops actually moving between two distinct states.
  edm::FileInPath x;
  std::istringstream isRel("V001 Sub/Pack/data/rel.txt 2 /src/Sub/Pack/data/rel.txt");
  x.readFromParameterSetBlob(isRel);
  std::ostringstream xBefore;
  x.write(xBefore);

  edm::FileInPath y("Sub/Pack/data/dat.txt");
  std::ostringstream yBefore;
  y.write(yBefore);

  using edm::swap;
  swap(x, y);

  std::ostringstream xAfter;
  x.write(xAfter);
  std::ostringstream yAfter;
  y.write(yAfter);

  REQUIRE(yAfter.str() == xBefore.str());
  REQUIRE(xAfter.str() == yBefore.str());
}

TEST_CASE("strayPathElement: warning names all four variables and the stray element, quoted", "[strayPathElement]") {
  // Covers the warning added in commit 53065377. Match on the quoted forms:
  // the unquoted "<root>/local/src" is not a substring of
  // "<root>/local2/src", but the quoting makes the intent (and the
  // regression coverage) explicit.
  REQUIRE_THAT(g_firstConstructionStderr, Catch::Matchers::ContainsSubstring("CMSSW_SEARCH_PATH"));
  REQUIRE_THAT(g_firstConstructionStderr, Catch::Matchers::ContainsSubstring("CMSSW_BASE"));
  REQUIRE_THAT(g_firstConstructionStderr, Catch::Matchers::ContainsSubstring("CMSSW_RELEASE_BASE"));
  REQUIRE_THAT(g_firstConstructionStderr, Catch::Matchers::ContainsSubstring("CMSSW_DATA_PATH"));
  REQUIRE_THAT(g_firstConstructionStderr,
               Catch::Matchers::ContainsSubstring("'" + (g_paths.root / "local2/src").string() + "'"));
  REQUIRE_THAT(g_firstConstructionStderr,
               !Catch::Matchers::ContainsSubstring("'" + (g_paths.local / "src").string() + "'"));
}

TEST_CASE("strayPathElement: the warning is emitted only once", "[strayPathElement]") {
  // std::call_once guards the warning; a second construction must not
  // re-emit it, even though the first construction already happened (in
  // main(), before the Catch2 session ran).
  CerrCapture cap;
  edm::FileInPath dummy;
  REQUIRE(cap.str().empty());
}

TEST_CASE("strayPathElement: a file only reachable via the stray element is rejected", "[strayPathElement]") {
  // Direct regression test for pathBeginsWith(): "<root>/local2/src"
  // string-starts-with "<root>/local" but is not a path prefix of it.
  REQUIRE_THROWS_WITH(edm::FileInPath("Sub/Pack/data/sib.txt"),
                      Catch::Matchers::ContainsSubstring("not in any of the known search areas"));
}

TEST_CASE("strayPathElement: an ordinary Local file still resolves normally", "[strayPathElement]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  REQUIRE(fip.location() == edm::FileInPath::Local);
  REQUIRE(fip.fullPath() == (g_paths.local / "src/Sub/Pack/data/file.txt").string());
}

TEST_CASE("read accepts a record naming a sibling of the local top", "[local]") {
  // Issue 7: write() uses a raw string-prefix test
  // (canonicalFilename_.find(localTop_) == 0) while initialize_() uses the
  // component-wise pathBeginsWith(). A record naming a file under
  // "<CMSSW_BASE>2/..." therefore round-trips as Local via read()/write(),
  // even though the strayPathElement scenario above shows initialize_()
  // rejecting the very same directory as "not in any of the known search
  // areas" when reached through construction instead of read().
  edm::FileInPath fip;
  std::istringstream is("V001 Sub/Pack/data/sib.txt 1 2/src/Sub/Pack/data/sib.txt");
  fip.read(is);
  REQUIRE_FALSE(is.fail());
  REQUIRE(fip.location() == edm::FileInPath::Local);
  REQUIRE(fip.fullPath() == (g_paths.local.string() + "2/src/Sub/Pack/data/sib.txt"));

  std::ostringstream os;
  fip.write(os);
  REQUIRE(os.str() == "V001 Sub/Pack/data/sib.txt 1 2/src/Sub/Pack/data/sib.txt");
}

TEST_CASE("paths with spaces round-trip through write() but not read()", "[local]") {
  // Issue 8 (documented limitation, FileInPath.h:48-52): write() happily
  // serialises a path containing a space, but read() cannot parse the
  // result back, because both fields are whitespace-delimited.
  edm::FileInPath fip("Sub/Pack/data/with space.txt");
  REQUIRE(fip.relativePath() == "Sub/Pack/data/with space.txt");
  REQUIRE(fip.location() == edm::FileInPath::Local);

  std::ostringstream os;
  fip.write(os);
  REQUIRE(os.str() == "V001 Sub/Pack/data/with space.txt 1 /src/Sub/Pack/data/with space.txt");

  // >> relname stops at "Sub/Pack/data/with", then >> loc fails trying to
  // parse "space.txt" as an int, so the stream ends in fail() and the
  // object is left corrupted rather than untouched (see Issue 4/10 above -
  // this is a third, non-truncated way to reach the same inconsistency).
  edm::FileInPath fip2;
  std::istringstream is(os.str());
  fip2.read(is);
  REQUIRE(is.fail());
  REQUIRE(fip2.location() == edm::FileInPath::Unknown);
  REQUIRE(fip2.relativePath().empty());
  REQUIRE(fip2.fullPath().empty());
}

TEST_CASE("elementIsTop: a Data file resolves when CMSSW_DATA_PATH equals the search-path element", "[elementIsTop]") {
  // The configuration the old parent_path()-based loop in initialize_()
  // could never match (it started at parent_path(), so a top could never
  // equal a search-path element) - the exact hang fixed by 4c834e9.
  edm::FileInPath fip("Sub/Pack/data/dat.txt");
  REQUIRE(fip.location() == edm::FileInPath::Data);
  REQUIRE(fip.fullPath() == (g_paths.data / "repo/Sub/Pack/data/dat.txt").string());

  std::ostringstream os;
  fip.write(os);
  REQUIRE(os.str() == "V001 Sub/Pack/data/dat.txt 3 /Sub/Pack/data/dat.txt");

  edm::FileInPath fip2;
  std::istringstream is(os.str());
  fip2.read(is);
  REQUIRE_FALSE(is.fail());
  REQUIRE(fip2.fullPath() == fip.fullPath());
}

TEST_CASE("elementIsTop: no warning - lexically_relative recognises the element as the top itself", "[elementIsTop]") {
  // lexically_relative(element, dataTop) yields "." here, so pathBeginsWith()
  // correctly treats the element as being inside (in fact, equal to) the
  // data top.
  REQUIRE(g_firstConstructionStderr.empty());
}

TEST_CASE("trailingSlashTop: a trailing slash on CMSSW_BASE leads to no warnings", "[trailingSlashTop]") {
  REQUIRE(g_firstConstructionStderr.empty());
}

TEST_CASE("trailingSlashTop: an ordinary Local file resolves normally", "[trailingSlashTop]") {
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  REQUIRE(fip.location() == edm::FileInPath::Local);
  REQUIRE(fip.fullPath() == (g_paths.local / "src/Sub/Pack/data/file.txt").string());
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

TEST_CASE("symlink: no warning despite the symlinked search-path elements", "[symlink]") {
  REQUIRE(g_firstConstructionStderr.empty());
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

TEST_CASE("emptySearchPath: the throw repeats on every construction", "[emptySearchPath]") {
  // D7: an exception escaping std::call_once leaves the once_flag unset, so
  // the "CMSSW_SEARCH_PATH must be defined" check is retried on every
  // subsequent construction rather than being silently skipped. The
  // existing per-SECTION tests above do not assert this within a single
  // flow, since each SECTION reruns TEST_CASE from scratch.
  auto matchesSearchPath = Catch::Matchers::ContainsSubstring("CMSSW_SEARCH_PATH");
  REQUIRE_THROWS_WITH(edm::FileInPath{}, matchesSearchPath);
  REQUIRE_THROWS_WITH(edm::FileInPath{}, matchesSearchPath);
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

TEST_CASE("lookupDisabled: write/read of an Unknown object with a relative path", "[lookupDisabled]") {
  // D3: with lookup disabled, the ctor sets relativePath_ but leaves
  // location_ == Unknown and canonicalFilename_ empty. This is the last
  // uncovered branch of write() (FileInPath.cc:161: Unknown with a
  // non-empty relativePath_), reachable only in this scenario - everywhere
  // else, a non-empty relativePath_ implies a resolved (non-Unknown)
  // location.
  edm::FileInPath fip("Sub/Pack/data/file.txt");
  std::ostringstream os;
  fip.write(os);
  REQUIRE(os.str() == "V001 Sub/Pack/data/file.txt 0");

  edm::FileInPath fip2;
  std::istringstream is(os.str());
  fip2.read(is);
  REQUIRE_FALSE(is.fail());
  REQUIRE(fip2.relativePath() == "Sub/Pack/data/file.txt");
  REQUIRE(fip2.location() == edm::FileInPath::Unknown);
  // The canFilename extraction is skipped entirely for Unknown.
  REQUIRE(fip2.fullPath().empty());
}

TEST_CASE("noData: file is found CMSSW_SEARCH_PATH, but element is not in any of the known search areas", "[noData]") {
  REQUIRE_THROWS_WITH(
      edm::FileInPath("Sub/Pack/data/dat.txt"),
      Catch::Matchers::ContainsSubstring("edm::FileInPath found file Sub/Pack/data/dat.txt in search path element") &&
          Catch::Matchers::ContainsSubstring("but that element is not in any of the known search"));
}

TEST_CASE("noData: read() of a Data record throws naming CMSSW_DATA_PATH", "[noData]") {
  std::istringstream is("V001 Sub/Pack/data/dat.txt 3 /Sub/Pack/data/dat.txt");
  edm::FileInPath fip;
  REQUIRE_THROWS_WITH(fip.read(is), Catch::Matchers::ContainsSubstring("CMSSW_DATA_PATH"));
}

TEST_CASE("noData: readFromParameterSetBlob of a Data record also throws", "[noData]") {
  // Issue 9: unlike Local/Release, readFromParameterSetBlob() does not
  // degrade gracefully for a missing data top; it throws just like read().
  std::istringstream is("V001 Sub/Pack/data/dat.txt 3 /Sub/Pack/data/dat.txt");
  edm::FileInPath fip;
  REQUIRE_THROWS_WITH(fip.readFromParameterSetBlob(is), Catch::Matchers::ContainsSubstring("CMSSW_DATA_PATH"));
}

TEST_CASE("absolutePath: absolute relativePath_ bypasses the search path and hangs", "[absolutePath]") {
  REQUIRE_THROWS_WITH(edm::FileInPath((fs::current_path() / "fip_absolutePath/absolutePath.txt").string()),
                      Catch::Matchers::ContainsSubstring("The path must be relative, not absolute:"));
}

TEST_CASE("emptyPathElement: file that would be in CWD is nonetheless reported as not found", "[emptyPathElement]") {
  REQUIRE_THROWS_WITH(
      edm::FileInPath("fip_emptyPathElement_marker.txt"),
      Catch::Matchers::ContainsSubstring("edm::FileInPath found file fip_emptyPathElement_marker.txt in search path "
                                         "element '', but that element is not in any of the known search areas."));
}

TEST_CASE("emptyPathElement: empty search-path element is reported as ''", "[emptyPathElement]") {
  // CMSSW_SEARCH_PATH=":" tokenizes into two empty path-prefix elements
  // (edm::tokenize keeps empty tokens); confirms the behaviour the code
  // comment at FileInPath.cc:355 claims but nothing else checks.
  REQUIRE_THAT(g_firstConstructionStderr, Catch::Matchers::ContainsSubstring("''"));
}

TEST_CASE("symlinkException: CMSSW_BASE/src symlinked to a non-'src' directory throws an exception",
          "[symlinkException]") {
  REQUIRE_THROWS_WITH(
      edm::FileInPath("Sub/Pack/data/file.txt"),
      Catch::Matchers::ContainsSubstring("CMSSW_BASE/src is a symbolic link to a directory not literally "
                                         "named 'src'"));
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << (argc > 0 ? argv[0] : "testFWCoreUtilitiesFileInPath") << " <scenario>\n";
    return 1;
  }
  std::string const scenario = argv[1];

  g_paths = setupScenario(scenario);

  // Capture the warning (if any) emitted by the very first getEnvironment()
  // call. This must happen here, before any TEST_CASE runs, because the
  // warning is guarded by std::call_once (FileInPath.cc:349-375) and would
  // otherwise be observed by whichever test case happens to run first.
  // Catching is needed because the default ctor throws in the
  // "emptySearchPath" and "symlinkException" scenarios; note that a throw
  // escaping call_once leaves the flag unset, and that a throwing static
  // initializer (removeSymLinksSrc) is retried on the next call, so both
  // scenarios keep working unchanged.
  {
    CerrCapture cap;
    try {
      edm::FileInPath dummy;
    } catch (...) {
      // expected in some scenarios
    }
    g_firstConstructionStderr = cap.str();
  }

  Catch::Session session;
  session.configData().testsOrTags = {"[" + scenario + "]"};
  return session.run();
}
