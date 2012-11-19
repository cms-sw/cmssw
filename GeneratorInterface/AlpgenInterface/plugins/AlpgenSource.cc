#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <memory>
#include <cmath>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/AlpgenInterface/interface/AlpgenHeader.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenEventRecordFixes.h"

class AlpgenSource : public edm::ProducerSourceFromFiles {
public:
  /// Constructor
  AlpgenSource(const edm::ParameterSet &params,
	       const edm::InputSourceDescription &desc);

  /// Destructor
  virtual ~AlpgenSource();

private:
  virtual bool setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&);
  virtual void produce(edm::Event &event);
  virtual void beginRun(edm::Run &run);

  /// Function to get parameter by name from AlpgenHeader.
  template<typename T>
  T getParameter(AlpgenHeader::Parameter index) const;
  /// Function to get parameter by name from AlpgenHeader, w/ default.
  template<typename T>
  T getParameter(AlpgenHeader::Parameter index, const T &defValue) const;

  /// Converts the AlpgenHeader::Masses to a std::string
  /// formatted as a slhaMassLine to facilitate passing them to Alpgen.
  std::string slhaMassLine(int pdgId, AlpgenHeader::Masses mass,
			   const std::string &comment) const;

  /// The Alpgen process ID. This is defined as
  /// processID() = 100*X + 10*Y + Z, 
  /// where = ihrd, Y = ihvy, Z = njets
  unsigned int processID() const;

  /// Read an event and put it into the HEPEUP. 
  bool readAlpgenEvent(lhef::HEPEUP &hepeup);

  /// Name of the input file
  std::string			fileName_;

  /// Pointer to the input file
  std::unique_ptr<std::ifstream> inputFile_;

  /// Number of events to skip
  unsigned long			skipEvents_;

  /// Number of events
  unsigned long			nEvent_;

  /// Alpgen _unw.par file as a LHE header
  LHERunInfoProduct::Header	lheAlpgenUnwParHeader;
  /// Alpgen _unw.par file as an AlpgenHeader
  AlpgenHeader			header;

  std::unique_ptr<lhef::HEPEUP> hepeup_;

  /// Name of the extra header file
  std::string			extraHeaderFileName_;

  /// Name given to the extra header
  std::string			extraHeaderName_;

  /// configuration flags
  bool				writeAlpgenWgtFile;
  bool				writeAlpgenParFile;
  bool				writeExtraHeader;
};

AlpgenSource::AlpgenSource(const edm::ParameterSet &params,
			   const edm::InputSourceDescription &desc) :
  edm::ProducerSourceFromFiles(params, desc, false), 
  skipEvents_(params.getUntrackedParameter<unsigned int>("skipEvents", 0)),
  nEvent_(0), lheAlpgenUnwParHeader("AlpgenUnwParFile"),
  extraHeaderFileName_(params.getUntrackedParameter<std::string>("extraHeaderFileName","")),
  extraHeaderName_(params.getUntrackedParameter<std::string>("extraHeaderName","")),
  writeAlpgenWgtFile(params.getUntrackedParameter<bool>("writeAlpgenWgtFile", true)),
  writeAlpgenParFile(params.getUntrackedParameter<bool>("writeAlpgenParFile", true)),
  writeExtraHeader(params.getUntrackedParameter<bool>("writeExtraHeader", false))
{
  std::vector<std::string> allFileNames = fileNames();

  // Only one filename
  if (allFileNames.size() != 1)
    throw cms::Exception("Generator|AlpgenInterface")
      << "AlpgenSource needs exactly one file specified "
      "for now." << std::endl;

  fileName_ = allFileNames[0];

  // Strip the "file:" prefix 
  if (fileName_.find("file:") != 0)
    throw cms::Exception("Generator|AlpgenInterface") << "AlpgenSource only supports the file: scheme "
      "for now." << std::endl;
  fileName_.erase(0, 5);

  // Open the _unw.par file to store additional 
  // informations in the LHERunInfoProduct
  std::ifstream reader((fileName_ + "_unw.par").c_str());
  if (!reader.good())
    throw cms::Exception("Generator|AlpgenInterface")
      << "AlpgenSource was unable to open the file \""
      << fileName_ << "_unw.par\"." << std::endl;

  // A full copy of the _unw.par file in the LHE header.
  char buffer[256];
  while(reader.getline(buffer, sizeof buffer))
    lheAlpgenUnwParHeader.addLine(std::string(buffer) + "\n");

  // Parse that header to setup an Alpgen header,
  // which will be used in the production itself.
   if (!header.parse(lheAlpgenUnwParHeader.begin(),
		    lheAlpgenUnwParHeader.end()))
    throw cms::Exception("Generator|AlpgenInterface")
      << "AlpgenSource was unable to parse the Alpgen "
      << "unweighted parameter file." << std::endl;

  // Declare the products.
  produces<LHERunInfoProduct, edm::InRun>();
  produces<LHEEventProduct>();
}

AlpgenSource::~AlpgenSource()
{
}

std::string AlpgenSource::slhaMassLine(int pdgId, AlpgenHeader::Masses mass,
                                       const std::string &comment) const
{
  std::ostringstream ss;
  ss << std::setw(9) << pdgId << "     " << std::scientific
     << std::setprecision(9) << header.masses[mass] << "   # "
     << comment << std::endl;
  return ss.str();
}

void AlpgenSource::beginRun(edm::Run &run)
{
  // At this point, the lheUnwParHeader has the full contents of the _unw.par
  // file. So we can get the HEPRUP information from the LHE header itself.
  lhef::HEPRUP heprup;

  // Get basic run information.
  // Beam identity.
  heprup.IDBMUP.first = 2212;
  switch(getParameter<int>(AlpgenHeader::ih2)) {
  case 1:
    heprup.IDBMUP.second = 2212;
    break;
  case -1:
    heprup.IDBMUP.second = -2212;
    break;
  default:
    throw cms::Exception("Generator|AlpgenInterface")
      << "AlpgenSource was unable to understand the ih2 "
      << "parameter." << std::endl;
  }

  // Beam energy.
  heprup.EBMUP.second = heprup.EBMUP.first =
    getParameter<double>(AlpgenHeader::ebeam);

  // PDF info. Initially, Alpgen doesn't fill it.
  heprup.PDFGUP.first = -1;
  heprup.PDFGUP.second = -1;
  heprup.PDFSUP.first = -1;
  heprup.PDFSUP.second = -1;

  // Unweighted events.
  heprup.IDWTUP = 3;

  // Only one process.
  heprup.resize(1);

  // Cross section and error.
  heprup.XSECUP[0] = header.xsec;
  heprup.XERRUP[0] = header.xsecErr;

  // Maximum weight.
  heprup.XMAXUP[0] = header.xsec;

  // Process code for Pythia.
  heprup.LPRUP[0] = processID();

  // Comments on top.
  LHERunInfoProduct::Header comments;
  comments.addLine("\n");
  comments.addLine("\tExtracted by AlpgenInterface\n");

  // Add SLHA header containing particle masses from Alpgen.
  // Pythia6Hadronisation will feed the masses to Pythia automatically.
  LHERunInfoProduct::Header slha("slha");
  slha.addLine("\n# SLHA header containing masses from Alpgen\n");
  slha.addLine("Block MASS      #  Mass spectrum (kinematic masses)\n");
  slha.addLine("#       PDG   Mass\n");
  slha.addLine(slhaMassLine(4, AlpgenHeader::mc, "charm    pole mass"));
  slha.addLine(slhaMassLine(5, AlpgenHeader::mb, "bottom   pole mass"));
  slha.addLine(slhaMassLine(6, AlpgenHeader::mt, "top      pole mass"));
  slha.addLine(slhaMassLine(23, AlpgenHeader::mz, "Z        mass"));
  slha.addLine(slhaMassLine(24, AlpgenHeader::mw, "W        mass"));
  slha.addLine(slhaMassLine(25, AlpgenHeader::mh, "H        mass"));

  char buffer[512];

  // We also add the information on weighted events.
  LHERunInfoProduct::Header lheAlpgenWgtHeader("AlpgenWgtFile");
  if (writeAlpgenWgtFile) {
    std::ifstream wgtascii((fileName_+".wgt").c_str());
    while(wgtascii.getline(buffer,512)) {
      lheAlpgenWgtHeader.addLine(std::string(buffer) + "\n");
    }
  }

  LHERunInfoProduct::Header lheAlpgenParHeader("AlpgenParFile");
  if (writeAlpgenParFile) {
    std::ifstream parascii((fileName_+".par").c_str());
    while(parascii.getline(buffer,512)) {
      lheAlpgenParHeader.addLine(std::string(buffer) + "\n");
    }
  }

  // If requested by the user, we also add any specific header provided.
  // Nota bene: the header is put in the LHERunInfoProduct AS IT WAS GIVEN.
  // That means NO CROSS-CHECKS WHATSOEVER. Use with care.
  LHERunInfoProduct::Header extraHeader(extraHeaderName_.c_str());
  if(writeExtraHeader) {
    std::ifstream extraascii(extraHeaderFileName_.c_str());
    while(extraascii.getline(buffer,512)) {
      extraHeader.addLine(std::string(buffer) + "\n");
    }
  }

  // Build the final Run info object. Backwards-compatible order.
  std::auto_ptr<LHERunInfoProduct> runInfo(new LHERunInfoProduct(heprup));
  runInfo->addHeader(comments);
  runInfo->addHeader(lheAlpgenUnwParHeader);
  if (writeAlpgenWgtFile)
    runInfo->addHeader(lheAlpgenWgtHeader);
  if (writeAlpgenParFile)
    runInfo->addHeader(lheAlpgenParHeader);
  runInfo->addHeader(slha);
  if(writeExtraHeader)
    runInfo->addHeader(extraHeader);
  run.put(runInfo);

  // Open the .unw file in the heap, and set the global pointer to it.
  inputFile_.reset(new std::ifstream((fileName_ + ".unw").c_str()));
  if (!inputFile_->good())
    throw cms::Exception("Generator|AlpgenInterface")
      << "AlpgenSource was unable to open the file \""
      << fileName_ << ".unw\"." << std::endl;

}

template<typename T>
T AlpgenSource::getParameter(AlpgenHeader::Parameter index) const
{
  std::map<AlpgenHeader::Parameter, double>::const_iterator pos =
    header.params.find(index);
  if (pos == header.params.end())
    throw cms::Exception("Generator|AlpgenInterface")
      << "Requested Alpgen parameter \""
      << AlpgenHeader::parameterName(index) << "\" "
      "not found in Alpgen parameter file." << std::endl;

  return T(pos->second);
}

template<typename T>
T AlpgenSource::getParameter(AlpgenHeader::Parameter index,
                             const T &defValue) const
{
  std::map<AlpgenHeader::Parameter, double>::const_iterator pos =
    header.params.find(index);
  if (pos == header.params.end())
    return defValue;
  else
    return T(pos->second);
}

unsigned int AlpgenSource::processID() const
{
  // return 661; // The original, old thing.
  // digits #XYZ: X = ihrd, Y = ihvy, Z = njets
  return header.ihrd * 100 +
    getParameter<unsigned int>(AlpgenHeader::ihvy, 0) * 10 +
    getParameter<unsigned int>(AlpgenHeader::njets, 0);
}

bool AlpgenSource::readAlpgenEvent(lhef::HEPEUP &hepeup)
{
  char buffer[512];
  double dummy;
  int nPart;
  double sWgtRes;
  double sQ;

  inputFile_->getline(buffer, sizeof buffer);
  if (!inputFile_->good())
    return false;

  std::istringstream ls(buffer);

  // Event number and process don't matter (or do they?)
  ls >> dummy >> dummy;

  // Number of particles in the record, sample's average weight and Q scale
  ls >> nPart >> sWgtRes >> sQ;

  if (ls.bad() || nPart < 1 || nPart > 1000)
    return false;

  // Make room for the particles listed in the Alpgen file
  hepeup.resize(nPart);

  // Scales, weights and process ID.
  hepeup.SCALUP = sQ;
  hepeup.XWGTUP = sWgtRes;
  hepeup.IDPRUP = processID();

  // Incoming lines
  for(int i = 0; i != 2; ++i) {
    inputFile_->getline(buffer, sizeof buffer);
    std::istringstream ls(buffer);
    int flavour; ls >> flavour;
    int colour1; ls >> colour1;
    int colour2; ls >> colour2;
    double zMomentum; ls >> zMomentum;

    if (inputFile_->bad())
      return false;

    // Setting the HEPEUP of the incoming lines.
    hepeup.IDUP[i] = flavour;
    hepeup.ISTUP[i] = -1;
    hepeup.MOTHUP[i].first = 0;
    hepeup.MOTHUP[i].second = 0;
    hepeup.PUP[i][0] = 0.;
    hepeup.PUP[i][1] = 0.;
    hepeup.PUP[i][2] = zMomentum;
    hepeup.PUP[i][3] = std::fabs(zMomentum);
    hepeup.PUP[i][4] = 0.;
    if (colour1) colour1 += 500;
    if (colour2) colour2 += 500;
    hepeup.ICOLUP[i].first = colour1;
    hepeup.ICOLUP[i].second = colour2;
  }

  // Outgoing lines
  for(int i = 2; i != nPart; ++i) {
    inputFile_->getline(buffer, sizeof buffer);
    std::istringstream ls(buffer);
    int flavour; ls >> flavour;
    int colour1; ls >> colour1;
    int colour2; ls >> colour2;
    double px, py, pz, mass; 
    ls >> px >> py >> pz >> mass;
    double energy = std::sqrt(px*px + py*py + pz*pz + mass*mass);

    if (inputFile_->bad())
      return false;

    // Setting the HEPEUP of the outgoing lines.
    hepeup.IDUP[i] = flavour;
    hepeup.ISTUP[i] = 1;
    hepeup.MOTHUP[i].first = 1;
    hepeup.MOTHUP[i].second = 2;
    hepeup.PUP[i][0] = px;
    hepeup.PUP[i][1] = py;
    hepeup.PUP[i][2] = pz;
    hepeup.PUP[i][3] = energy;
    hepeup.PUP[i][4] = mass;
    if (colour1) colour1 += 500;
    if (colour2) colour2 += 500;
    hepeup.ICOLUP[i].first = colour1;
    hepeup.ICOLUP[i].second = colour2;
  }

  return true;
}

bool AlpgenSource::setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&)
{
  // The LHE Event Record
  hepeup_.reset(new lhef::HEPEUP);

  lhef::HEPEUP& hepeup = *hepeup_;

  // Read the .unw file until it is over.
  for(;;) {
    if (!readAlpgenEvent(hepeup)) {
      if (inputFile_->eof())
	return false;

      throw cms::Exception("Generator|AlpgenInterface")
	<< "AlpgenSource is not able read event no. "
	<< nEvent_ << std::endl;
    }

    nEvent_++;
    if (skipEvents_ > 0)
      skipEvents_--;
    else
      break;
  }
  return true;
}

void AlpgenSource::produce(edm::Event &event)
{

  // Here are the Alpgen routines for filling up the rest
  // of the LHE Event Record. The .unw file has the information
  // in a compressed way, e.g. it doesn't list the W boson - 
  // one has to reconstruct it from the e nu pair.
  lhef::HEPEUP& hepeup = *hepeup_;

  switch(header.ihrd) {
  case 1:
  case 2:
  case 3:
  case 4:
  case 10:
  case 14:
  case 15:
    alpgen::fixEventWZ(hepeup);
    break;
  case 5:
    alpgen::fixEventMultiBoson(hepeup);
    break;
  case 6:
    alpgen::fixEventTTbar(hepeup);
    break;
  case 8:
    alpgen::fixEventHiggsTTbar(hepeup);
    break;
  case 13:
    alpgen::fixEventSingleTop(hepeup, header.masses[AlpgenHeader::mb],
			      int(header.params[AlpgenHeader::itopprc]));
    break;    
  case 7:
  case 9:
  case 11:
  case 12:
  case 16:
    // No fixes needed.
    break;

  default: 
    throw cms::Exception("Generator|AlpgenInterface") 
      << "Unrecognized IHRD process code" << std::endl;
  }

  // Create the LHEEventProduct and put it into the Event.
  std::auto_ptr<LHEEventProduct> lheEvent(new LHEEventProduct(hepeup));
  event.put(lheEvent);

  hepeup_.reset();
}

DEFINE_FWK_INPUT_SOURCE(AlpgenSource);
