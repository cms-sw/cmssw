/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCBadStrips.h"
#include "CondFormats/DataRecord/interface/CSCBadStripsRcd.h"

namespace edmtest
{
  class CSCReadBadStripsAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  CSCReadBadStripsAnalyzer(edm::ParameterSet const& p) 
    { }
    explicit  CSCReadBadStripsAnalyzer(int i) 
    { }
    virtual ~ CSCReadBadStripsAnalyzer() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
  
  void
   CSCReadBadStripsAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    using namespace edm::eventsetup;
    std::ofstream BadStripFile("dbBadStrip.dat",std::ios::app);
    int counter=0;
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<CSCBadStrips> pBadStrip;
    context.get<CSCBadStripsRcd>().get(pBadStrip);

    const CSCBadStrips* myBadStrips=pBadStrip.product();
    std::vector<CSCBadStrips::BadChamber>::const_iterator itcham;
    std::vector<CSCBadStrips::BadChannel>::const_iterator itchan;

    for( itcham=myBadStrips->chambers.begin();itcham!=myBadStrips->chambers.end(); ++itcham ){    
      counter++;
      BadStripFile<<counter<<"  "<<itcham->chamber_index<<"  "<<itcham->pointer<<"  "<<itcham->bad_channels<<std::endl;
    }

    for( itchan=myBadStrips->channels.begin();itchan!=myBadStrips->channels.end(); ++itchan ){    
      counter++;
      BadStripFile<<counter<<"  "<<itchan->layer<<"  "<<itchan->channel<<"  "<<itchan->flag1<<std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCReadBadStripsAnalyzer);
}

