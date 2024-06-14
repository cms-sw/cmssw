#include "TFile.h"
#include "ROOT/RNTupleReader.hxx"
#include "ROOT/RNTuple.hxx"
#include <iostream>
#include <sstream>
#include <boost/program_options.hpp>

namespace {

  class InfoDump {
  public:
    InfoDump(std::string iOut) : dump_(std::move(iOut)) {}

    std::optional<std::string_view> nextLine();
    std::optional<std::pair<std::string_view, unsigned long long>> nextFieldInfo();
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

  std::optional<std::pair<std::string_view, unsigned long long>> InfoDump::nextFieldInfo() {
    auto line = nextLine();
    if (not line) {
      return {};
    }
    auto name = line->substr(2, line->find_first_of(" .", 2) - 2);

    nextLine();
    nextLine();
    nextLine();
    nextLine();
    line = nextLine();
    //std::cout <<line->substr(line->find_first_not_of(" ",line->find_first_of(":")+1))<<std::endl;
    auto size = std::atoll(line->substr(line->find_first_not_of(" ", line->find_first_of(":") + 1)).data());
    moveToLineWith("............................................................");
    return std::make_pair(name, size);
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
}  // namespace

int main(int iArgc, char const* iArgv[]) {
  // Add options here

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help,h", "print help message")(
      "file,f", boost::program_options::value<std::string>(), "data file")("print,P", "Print list of data products")(
      "verbose,v", "Verbose printout")("printProductDetails,p", "Call PrintInfo() for selected rntuple")(
      "rntuple,r", boost::program_options::value<std::string>(), "Select rntuple used with -P and -p options")(
      "sizeSummary,s", "Print size on disk for each data product")(
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

  using namespace ROOT::Experimental;

  auto file = TFile::Open(vm["file"].as<std::string>().c_str(), "r");
  if (not file) {
    std::cout <<"failed to open "<<vm["file"].as<std::string>()<<std::endl;
    return 1;
  }
  
  auto events = std::unique_ptr<ROOT::RNTuple>(file->Get<ROOT::RNTuple>("Events"));
  if(not events) {
    std::cout <<"failed to get 'Events' as an RNTuple"<<std::endl;
    return 1;
  }
  auto ntuple = RNTupleReader::Open(*events);

  if (vm.count("printProductDetails")) {
    ntuple->PrintInfo(ENTupleInfo::kStorageDetails, std::cout);
    return 0;
  }
  std::stringstream s;
  ntuple->PrintInfo(ENTupleInfo::kStorageDetails, s);

  InfoDump info{s.str()};

  info.moveToStartOfFields();

  std::string presentField;
  unsigned long long size = 0;
  auto field = info.nextFieldInfo();
  while (field) {
    if (field->first == presentField) {
      size += field->second;
    } else {
      if (not presentField.empty()) {
        std::cout << presentField << " " << size << std::endl;
      }
      presentField = field->first;
      size = 0;
    }
    field = info.nextFieldInfo();
  }
  if (not presentField.empty()) {
    std::cout << presentField << " " << size << std::endl;
  }

  return 0;
}
