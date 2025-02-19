/**
 * \file EcalContainmentCorrectionAnalyzer
 * 
 * Analyzer to test Shower Containment Corrections
 *   
 * $Id: EcalContainmentCorrectionAnalyzer.cc,v 1.1 2007/07/16 17:26:29 meridian Exp $
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include "CondFormats/EcalCorrections/interface/EcalGlobalShowerContainmentCorrectionsVsEta.h"
#include "CondFormats/DataRecord/interface/EcalGlobalShowerContainmentCorrectionsVsEtaRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
 
class EcalContainmentCorrectionAnalyzer: public edm::EDAnalyzer{
  
public:
  EcalContainmentCorrectionAnalyzer(const edm::ParameterSet& ps);
  ~EcalContainmentCorrectionAnalyzer();
  
protected:
  
  void analyze( edm::Event const & iEvent, const  edm::EventSetup& iSetup);

};

DEFINE_FWK_MODULE(EcalContainmentCorrectionAnalyzer);
 
EcalContainmentCorrectionAnalyzer::EcalContainmentCorrectionAnalyzer(const edm::ParameterSet& ps){
}

EcalContainmentCorrectionAnalyzer::~EcalContainmentCorrectionAnalyzer(){
 
}

void EcalContainmentCorrectionAnalyzer::analyze( edm::Event const & iEvent, 
					     const  edm::EventSetup& iSetup){

  using namespace edm;
  using namespace std;
 
  ESHandle<EcalGlobalShowerContainmentCorrectionsVsEta> pCorr;
  iSetup.get<EcalGlobalShowerContainmentCorrectionsVsEtaRcd>().get(pCorr);

  for (int i=1;i<86;i++)
    {
      EBDetId aId(i,1,EBDetId::ETAPHIMODE);
      double e3x3 = pCorr->correction3x3(aId);
      double e5x5 = pCorr->correction5x5(aId);
      std::cout << "ieta " << aId.ieta() << " " << e3x3 << " " << e5x5 << std::endl; 
    }
}



