
#include "GctDigiToPsbText.h"

#include <iomanip>
using std::setfill;
using std::setw;

GctDigiToPsbText::GctDigiToPsbText(const edm::ParameterSet &iConfig)
    : m_gctInputLabel(iConfig.getParameter<edm::InputTag>("GctInputLabel")),
      m_textFileName(iConfig.getParameter<std::string>("TextFileName")),
      m_hexUpperCase(iConfig.getUntrackedParameter<bool>("HexUpperCase", false)) {
  /// open output text files
  for (unsigned i = 0; i < 4; i++) {
    std::stringstream fileStream;
    int ii = (i < 2) ? i : i + 4;
    fileStream << m_textFileName << ii << ".txt";
    std::string fileName(fileStream.str());
    m_file[i].open(fileName.c_str(), std::ios::out);
    if (!m_file[i].good()) {
      throw cms::Exception("GctDigiToPsbTextTextFileOpenError")
          << "GctDigiToPsbText::GctDigiToPsbText : "
          << " couldn't create the file " << fileName << std::endl;
    }
  }
}

GctDigiToPsbText::~GctDigiToPsbText() {
  /// close  files
  for (unsigned i = 0; i < 4; i++)
    m_file[i].close();
}

void GctDigiToPsbText::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // static int nevt = -1; nevt++;

  // get digis
  edm::Handle<L1GctEmCandCollection> gctIsolaEm;
  edm::Handle<L1GctEmCandCollection> gctNoIsoEm;
  iEvent.getByLabel(m_gctInputLabel.label(), "isoEm", gctIsolaEm);
  iEvent.getByLabel(m_gctInputLabel.label(), "nonIsoEm", gctNoIsoEm);

  /// buffer
  uint16_t data[4][2] = {{0}};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      data[i][j] = 0;

  std::stringstream sstrm;
  if (m_hexUpperCase)
    sstrm << std::uppercase;
  else
    sstrm.unsetf(std::ios::uppercase);

  // specify cycle bit sequence 1 0 1 0 ... or 0 1 0 1 ...
  unsigned short cbs[2] = {1, 0};

  unsigned int ifile;
  unsigned int cycle;
  unsigned iIsola, iNoIso;

  for (int i = 0; i < 4; i++) {
    cycle = i / 2;
    ifile = i % 2;
    iIsola = ifile + 2;
    iNoIso = ifile;

    // get the data
    data[iIsola][cycle] = gctIsolaEm->at(i).raw();
    data[iNoIso][cycle] = gctNoIsoEm->at(i).raw();

    // print electrons
    sstrm.str("");
    sstrm << setw(4) << setfill('0') << std::hex << (data[iIsola][cycle] & 0x7fff) + ((cbs[cycle] & 0x1) << 15);
    m_file[iIsola] << sstrm.str() << std::endl;
    sstrm.str("");
    sstrm << setw(4) << setfill('0') << std::hex << (data[iNoIso][cycle] & 0x7fff) + ((cbs[cycle] & 0x1) << 15);
    m_file[iNoIso] << sstrm.str() << std::endl;
  }

  /// flush data to files
  for (unsigned i = 0; i < 4; i++)
    m_file[i] << std::flush;
}
