#include "TFile.h"
#include "ROOT/RNTupleReader.hxx"
#include "ROOT/RNTuple.hxx"
#include <charconv>
#include <iostream>
#include <sstream>
#include <boost/program_options.hpp>

namespace {
  struct FieldInfo {
    std::string_view name;
    unsigned long long compressedSize = 0;  // in B
    double uncompressedSize = 0;            // in B
  };

  class InfoDump {
  public:
    InfoDump(std::string iOut) : dump_(std::move(iOut)) {}

    std::optional<std::string_view> nextLine();
    std::optional<FieldInfo> nextFieldInfo();
    std::optional<unsigned int> moveToNumberOfEntries();
    void moveToStartOfFields();
    void moveToLineWith(std::string_view);

  private:
    std::string dump_;
    std::string::size_type start_ = 0;
  };

  std::optional<std::string_view> InfoDump::nextLine() {
    auto lastStart = start_;
    start_ = dump_.find('\n', start_);
    if (start_ == std::string::npos) {
      return std::optional<std::string_view>();
    }
    return std::string_view(dump_.data() + lastStart, start_++ - lastStart);
  }

  std::optional<unsigned int> InfoDump::moveToNumberOfEntries() {
    while (auto line = nextLine()) {
      auto entriesPos = line->find("# Entries:");
      if (entriesPos != std::string_view::npos) {
        auto tmp = line->substr(line->find_first_not_of(" ", entriesPos + 12));
        unsigned int entries{};
        auto [ptr, ec] = std::from_chars(tmp.begin(), tmp.end(), entries);
        if (ec != std::errc{}) {
          return {};
        }
        return entries;
      }
    }
    return {};
  }

  void InfoDump::moveToStartOfFields() {
    auto line = nextLine();
    while (line) {
      if (*line == "COLUMN DETAILS") {
        nextLine();
        return;
      }
      line = nextLine();
    }
    return;
  }

  std::optional<FieldInfo> InfoDump::nextFieldInfo() {
    auto line = nextLine();
    if (not line) {
      return {};
    }
    FieldInfo info;
    info.name = line->substr(2, line->find_first_of(" .", 2) - 2);

    nextLine();         // # elements
    nextLine();         // # pages
    nextLine();         // avg elememnts / page
    nextLine();         // avg page size
    line = nextLine();  // size on storage
    //std::cout <<line->substr(line->find_first_not_of(" ",line->find_first_of(":")+1))<<std::endl;
    {
      auto tmp = line->substr(line->find_first_not_of(" ", line->find_first_of(":") + 1));
      auto [ptr, ec] = std::from_chars(tmp.begin(), tmp.end(), info.compressedSize);
      if (ec != std::errc{}) {
        return {};
      }
    }
    if (info.compressedSize > 0) {  // avoid nans
      line = nextLine();            // compression factor
      auto tmp = line->substr(line->find_first_not_of(" ", line->find_first_of(":") + 1));
      double factor{};
      auto [ptr, ec] = std::from_chars(tmp.begin(), tmp.end(), factor);
      if (ec != std::errc{}) {
        return {};
      }
      info.uncompressedSize = info.compressedSize * factor;
    }
    moveToLineWith("............................................................");
    return info;
  }

  void InfoDump::moveToLineWith(std::string_view iCheck) {
    auto line = nextLine();
    while (line) {
      if (*line == iCheck) {
        return;
      }
      line = nextLine();
    }
    return;
  }

  void printFieldSizes(InfoDump& info, std::string const& fileName, std::string const& tupleName, bool average) {
    auto entries = info.moveToNumberOfEntries();
    if (not entries) {
      std::cout << "File " << fileName << " " << tupleName << " number of entries not found" << std::endl;
      return;
    }
    std::cout << "File " << fileName << " " << tupleName << " " << *entries << std::endl;
    info.moveToStartOfFields();

    std::string presentField;
    unsigned long long compressedSize = 0;
    double uncompressedSize = 0;
    std::function<void()> printField;
    auto field = info.nextFieldInfo();
    if (average) {
      std::string unit = "Bytes/";
      if (tupleName.back() == 's') {
        unit += tupleName.substr(0, tupleName.size() - 1);
      } else {
        unit += tupleName;
      }
      std::cout << std::format("Top-level field name | Average uncompressed size ({}) | Average compressed size ({})",
                               unit,
                               unit)
                << std::endl;
      printField = [&presentField, &compressedSize, &uncompressedSize, n = *entries]() {
        double us = uncompressedSize / n;
        double cs = static_cast<double>(compressedSize) / n;
        std::cout << std::format("{} {:.2f} {:.2f}", presentField, us, cs) << std::endl;
      };
    } else {
      std::cout << "Top-level field name | Uncompressed size (Bytes) | Compressed size (Bytes)" << std::endl;
      printField = [&presentField, &compressedSize, &uncompressedSize]() {
        std::cout << presentField << " " << std::format("{:.0f}", uncompressedSize) << " " << compressedSize
                  << std::endl;
      };
    }

    while (field) {
      if (field->name == presentField) {
        compressedSize += field->compressedSize;
        uncompressedSize += field->uncompressedSize;
      } else {
        if (not presentField.empty()) {
          printField();
        }
        presentField = field->name;
        compressedSize = 0;
        uncompressedSize = 0;
      }
      field = info.nextFieldInfo();
    }
    if (not presentField.empty()) {
      printField();
    }
  }
}  // namespace

int main(int iArgc, char const* iArgv[]) {
  // Add options here

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help,h", "print help message")(
      "file,f", boost::program_options::value<std::string>(), "data file")("print,P", "Print list of data products")(
      "verbose,v", "Verbose printout")("printProductDetails,p", "Call PrintInfo() for selected rntuple")(
      "rntuple,r", boost::program_options::value<std::string>(), "Select rntuple used with -P and -p options")(
      "sizeSummary,s", "Print size on disk as well as uncompressed size for each data product")(
      "average", "With -s, print the average sizes per entry in the RNTuple")(
      "events,e",
      "Print list of all Events, Runs, and LuminosityBlocks in the file sorted by run number, luminosity block number, "
      "and event number.  Also prints the entry numbers and whether it is possible to use fast copy with the file.")(
      "eventsInLumis", "Print how many Events are in each LuminosityBlock.");

  // What rntuples do we require for this to be a valid collection?
  std::vector<std::string> expectedRNTuples;
  expectedRNTuples.push_back("MetaData");
  expectedRNTuples.push_back("Events");

  boost::program_options::positional_options_description p;
  p.add("file", -1);

  boost::program_options::variables_map vm;

  try {
    boost::program_options::store(
        boost::program_options::command_line_parser(iArgc, iArgv).options(desc).positional(p).run(), vm);
  } catch (boost::program_options::error const& x) {
    std::cerr << "Option parsing failure:\n" << x.what() << "\n\n";
    std::cerr << desc << "\n";
    return 1;
  }

  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  using namespace ROOT;

  auto fileName = vm["file"].as<std::string>();
  auto file = TFile::Open(fileName.c_str(), "r");
  if (not file) {
    std::cout << "failed to open " << vm["file"].as<std::string>() << std::endl;
    return 1;
  }
  std::string tupleToRead = "Events";
  if (vm.count("rntuple")) {
    tupleToRead = vm["rntuple"].as<std::string>();
  }

  auto ntpl = std::unique_ptr<ROOT::RNTuple>(file->Get<ROOT::RNTuple>(tupleToRead.c_str()));
  if (not ntpl) {
    std::cout << "failed to get '" << tupleToRead << "' as an RNTuple" << std::endl;
    return 1;
  }
  auto ntuple = RNTupleReader::Open(*ntpl);

  if (vm.count("printProductDetails")) {
    ntuple->PrintInfo(ENTupleInfo::kStorageDetails, std::cout);
    return 0;
  }
  std::stringstream s;
  ntuple->PrintInfo(ENTupleInfo::kStorageDetails, s);

  InfoDump info{s.str()};
  printFieldSizes(info, fileName, tupleToRead, vm.count("average"));

  return 0;
}
