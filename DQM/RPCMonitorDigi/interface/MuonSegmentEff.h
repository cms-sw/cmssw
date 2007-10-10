/** \class MuonSegmentEff
 *
 * Class for RPC Monitoring using RPCDigi and DT and CSCS egments.
 *
 *  $Date: 2007/10/08 17:02:07 $
 *  $Revision: 1.5 $
 *
 * \author Camilo Carrillo (Uniandes)
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include <DataFormats/MuonDetId/interface/RPCDetId.h>


#include<string>
#include<map>
#include<fstream>

class RPCDetId;
class TFile;
class TH1F;
class TFile;
class TCanvas;
class TH2F;


class MuonSegmentEff : public edm::EDAnalyzer {
   public:
      explicit MuonSegmentEff(const edm::ParameterSet&);
      ~MuonSegmentEff();
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::map<std::string, MonitorElement*> bookDetUnitSeg(RPCDetId & detId);

   private:

      std::vector<std::map<RPCDetId, int> > counter;
      std::vector<int> totalcounter;
      std::ofstream ofrej;
      std::ofstream ofeff;
      bool incldt;
      bool incldtMB4;
      bool inclcsc;
      int widestrip;
      int widestripRB4;
      double MinCosAng;
      double MaxD;
      std::string muonRPCDigis;
      std::string cscSegments;
      std::string dt4DSegments;
      std::string rejected;
      std::string rollseff;


      //Giuseppe
      std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
      bool EffSaveRootFile;
      int  EffSaveRootFileEventsInterval;
      std::string EffRootFileName;
      std::string nameInLog;
      DaqMonitorBEInterface * dbe;

      std::vector<uint32_t> _idList;

};
