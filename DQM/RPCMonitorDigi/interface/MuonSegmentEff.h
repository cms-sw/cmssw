/** \class MuonSegmentEff
 *
 * Class for RPC Monitoring using RPCDigi and DT and CSCS egments.
 *
 *  $Date: 2006/10/24 05:50:06 $
 *  $Revision: 1.6 $
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



class MuonSegmentEff : public edm::EDAnalyzer {
   public:
      explicit MuonSegmentEff(const edm::ParameterSet&);
      ~MuonSegmentEff();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      std::vector<std::map<RPCDetId, int> > counter;
      std::vector<int> totalcounter;
      std::ofstream ofrej;

};

