// L1TCaloStage1LutWriter.cc
// Author: Leonard Apanasevich
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage1TauIsolationLUT.h"

#include <iostream>
#include <fstream>
#include <sys/stat.h>

//
// class declaration
//

namespace l1t {

class L1TCaloStage1LutWriter : public edm::EDAnalyzer {
public:
  explicit L1TCaloStage1LutWriter(const edm::ParameterSet&);
  ~L1TCaloStage1LutWriter();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void writeIsoTauLut(const std::string& fileName);
  bool openOutputFile(const std::string& fileName, std::ofstream& file);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  CaloParamsHelper* m_params;
  std::string m_conditionsLabel;

  Stage1TauIsolationLUT* isoTauLut;

  bool m_writeIsoTauLut;
  // output file names
  std::string m_isoTauLutName;
  // std::string m_EGammaLutName;
};

//
// constructors and destructor
//
L1TCaloStage1LutWriter::L1TCaloStage1LutWriter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  m_writeIsoTauLut = iConfig.getUntrackedParameter<bool>("writeIsoTauLut", false );
  m_isoTauLutName  = iConfig.getUntrackedParameter<std::string>("isoTauLutName", "isoTauLut.txt" );
  m_conditionsLabel = iConfig.getParameter<std::string>("conditionsLabel");

  m_params = new CaloParamsHelper;
  isoTauLut = new Stage1TauIsolationLUT(m_params);

};

L1TCaloStage1LutWriter::~L1TCaloStage1LutWriter()
{
  delete isoTauLut;
};

// ------------ method called for each event  ------------
void
L1TCaloStage1LutWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<CaloParams> paramsHandle;
  iSetup.get<L1TCaloParamsRcd>().get(m_conditionsLabel, paramsHandle);
  m_params = new (m_params) CaloParamsHelper(*paramsHandle.product());
  if (! m_params){
    std::cout << "Could not retrieve params from Event Setup" << std::endl;
    return;
  }
  LogDebug("L1TDebug") << *m_params << std::endl;

  if (m_writeIsoTauLut) writeIsoTauLut(m_isoTauLutName);

}

bool
L1TCaloStage1LutWriter::openOutputFile(const std::string& fileName, std::ofstream& file)
{
  // Write to a new file
  struct stat buffer ;
  if ( !stat(  fileName.c_str(), &buffer ) ) {
    std::cout << "File " <<  fileName << " already exists. It will not be overwritten." << std::endl;
    return false;
  } else {
    file.open( fileName.c_str() );
    if ( ! file.good()) {    
      std::cout << "Error opening file " <<  fileName << std::endl;
      return false;
    }
  }
  return true;
}

void L1TCaloStage1LutWriter::writeIsoTauLut(const std::string& fileName)
{
  std::ofstream file;
  if (openOutputFile(fileName,file)) {
    std::cout << "Writing tau isolation LUT to: " << fileName << std::endl;
    file << "########################################\n"
	 << "# tauIsolation LUT for ISOL(A)= " << m_params->tauMaxJetIsolationA()
	 << "  ISOL(B)= " << m_params->tauMaxJetIsolationB() << "\n"
	 << "# Switch to ISOLB value at pt= " << m_params->tauMinPtJetIsolationB() << "\n"
	 << "#<header> V" <<  Stage1TauIsolationLUT::lut_version  << " "
	 << Stage1TauIsolationLUT::nbitsJet+Stage1TauIsolationLUT::nbitsTau << " "
	 << Stage1TauIsolationLUT::nbits_data << " </header>\n"
	 << "# Format:  Address  Payload  ## hwTauPt hwJetPt\n"
	 <<  "########################################\n";

    unsigned maxAdd=pow(2,Stage1TauIsolationLUT::nbitsJet+Stage1TauIsolationLUT::nbitsTau);
    for (unsigned iAdd=0; iAdd<maxAdd; iAdd++) {
      int isoFlag= isoTauLut->lutPayload(iAdd);
      file << iAdd << " " << isoFlag << "\n";
	}
  } else {
    std::cout << "%Error opening output file. Tau isolation LUT not written." << std::endl;    
  }
  file.close();

}
// ------------ method called once each job just before starting event loop  ------------
void L1TCaloStage1LutWriter::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void L1TCaloStage1LutWriter::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TCaloStage1LutWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

}
using namespace l1t;

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloStage1LutWriter);


