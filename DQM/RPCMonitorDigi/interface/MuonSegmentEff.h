/** \class MuonSegmentEff
 *
 * Class for RPC Monitoring using RPCDigi and DT and CSCS egments.
 *
 *  $Date: 2008/03/01 00:40:14 $
 *  $Revision: 1.8 $
 *
 * \author Camilo Carrillo (Uniandes)
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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
      std::ofstream oftwiki;
      bool incldt;
      bool incldtMB4;
      bool inclcsc;
      double widestrip;
      double widestripRB4;
      double MinCosAng;
      double MaxD;
      std::string muonRPCDigis;
      std::string cscSegments;
      std::string dt4DSegments;
      std::string rejected;
      std::string rollseff;


      //Giuseppe
      std::map<std::string, std::map<std::string, MonitorElement*> >  meCollection;
      bool EffSaveRootFile;
      int  EffSaveRootFileEventsInterval;
      std::string EffRootFileName;
      std::string nameInLog;
      DQMStore * dbe;

      std::vector<std::string> _idList;

};
