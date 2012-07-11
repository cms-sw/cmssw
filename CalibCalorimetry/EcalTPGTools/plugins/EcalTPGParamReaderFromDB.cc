// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class EcalTPGParamReaderFromDB : public edm::EDAnalyzer {
        public:
                explicit EcalTPGParamReaderFromDB(const edm::ParameterSet&);
                ~EcalTPGParamReaderFromDB();

        private:
                virtual void beginJob() ;
                virtual void analyze(const edm::Event&, const edm::EventSetup&);
                virtual void endJob() ;

                std::string host;
                std::string sid;
                std::string user;
                std::string pass;
                int port;
                int min_run;
                int n_run;
};


#include "CalibCalorimetry/EcalTPGTools/plugins/EcalTPGDBApp.h"


EcalTPGParamReaderFromDB::EcalTPGParamReaderFromDB(const edm::ParameterSet & ps)
{
        host    = ps.getParameter<std::string>("host");
        sid     = ps.getParameter<std::string>("sid");
        user    = ps.getParameter<std::string>("user");
        pass    = ps.getParameter<std::string>("pass");
        port    = ps.getParameter<int>("port");
        min_run = ps.getParameter<int>("min_run");
        n_run   = ps.getParameter<int>("n_run");
}



EcalTPGParamReaderFromDB::~EcalTPGParamReaderFromDB()
{
}



void EcalTPGParamReaderFromDB::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
        try {
                EcalTPGDBApp app( sid, user, pass);

                //int i ; 
                //app.readTPGPedestals(i);
                //app.writeTPGLUT();
                //app.writeTPGWeights();

        } catch (std::exception &e) {
                std::cout << "ERROR:  " << e.what() << std::endl;
        } catch (...) {
                std::cout << "Unknown error caught" << std::endl;
        }

        std::cout << "All Done." << std::endl;
}



void EcalTPGParamReaderFromDB::beginJob()
{
}



void EcalTPGParamReaderFromDB::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalTPGParamReaderFromDB);
