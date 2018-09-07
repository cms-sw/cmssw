#include "DQM/CastorMonitor/interface/CastorLEDMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//***************************************************//
//********** CastorLEDMonitor ***********************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 20.11.2008 (first version) ******//
//---------- last revision: 31.05.2011 (Panos Katsas)
//***************************************************//
//---- critical revision 26.06.2014 (Vladimir Popov)
//==================================================================//

CastorLEDMonitor::CastorLEDMonitor(const edm::ParameterSet& ps) {
  fVerbosity = ps.getUntrackedParameter<int>("debug", 0);
  if (fVerbosity > 0)
    std::cout << "CastorLEDMonitor Constructor: " << this << std::endl;
  subsystemname =
      ps.getUntrackedParameter<std::string>("subSystemFolder", "Castor");
  ievt_ = 0;
}

CastorLEDMonitor::~CastorLEDMonitor() {}

void CastorLEDMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                      const edm::Run& iRun,
                                      const edm::EventSetup& iSetup) {
  char s[60];

  ibooker.setCurrentFolder(subsystemname + "/CastorLEDMonitor");

  sprintf(s, "CastorLEDqMap(cumulative)");
  h2qMap = ibooker.book2D(s, s, 14, 0, 14, 16, 0, 16);
  h2qMap->getTH2F()->SetOption("colz");
  sprintf(s, "CastorLED_QmeanMap");
  h2meanMap = ibooker.book2D(s, s, 14, 0, 14, 16, 0, 16);
  h2meanMap->getTH2F()->GetXaxis()->SetTitle("moduleZ");
  h2meanMap->getTH2F()->GetYaxis()->SetTitle("sectorPhi");
  h2meanMap->getTH2F()->SetOption("colz");

  ievt_ = 0;
  return;
}

void CastorLEDMonitor::processEvent(const CastorDigiCollection& castorDigis,
                                    const CastorDbService& cond) {
  if (fVerbosity > 0)
    std::cout << "CastorLEDMonitor::processEvent (start)" << std::endl;

  /* be implemented
   edm::Handle<HcalTBTriggerData> trigger_data;
   iEvent.getByToken(tok_tb_, trigger_data);
   if(trigger_data.isValid())
    if(trigger_data->triggerWord()==6) LEDevent=true;
  */

  if (castorDigis.empty()) {
    if (fVerbosity > 0)
      std::cout << "CastorLEDMonitor::processEvent NO Castor Digis"
                << std::endl;
    return;
  }

  for (CastorDigiCollection::const_iterator j = castorDigis.begin();
       j != castorDigis.end(); j++) {
    const CastorDataFrame digi = (const CastorDataFrame)(*j);
    int module = digi.id().module() - 1;
    int sector = digi.id().sector() - 1;
    double qsum = 0.;
    for (int i = 0; i < digi.size(); i++) {
      int dig = digi.sample(i).adc() & 0x7f;
      float ets = LedMonAdc2fc[dig] + 0.5;
      // h2qts->Fill(i,ets);
      qsum += ets;
    }
    // int ind = sector*14 + module;
    // h2QvsPMT->Fill(ind,qsum);
    h2qMap->Fill(module, sector, qsum);
  }  // end for(CastorDigiCollection::const_iterator j=castorDigis...

  ievt_++;
  if (ievt_ % 100 == 0) {
    for (int mod = 1; mod <= 14; mod++)
      for (int sec = 1; sec <= 16; sec++) {
        double a = h2qMap->getTH2F()->GetBinContent(mod, sec);
        h2meanMap->getTH2F()->SetBinContent(mod, sec, a / double(ievt_));
      }
  }

  if (fVerbosity > 0)
    std::cout << "CastorLEDMonitor::processEvent(end)" << std::endl;
  return;
}
