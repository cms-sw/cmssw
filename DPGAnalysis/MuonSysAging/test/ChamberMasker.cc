// -*- C++ -*-
//
// Class:      ChamberMasker
// 
//
// Original Author:  Sunil Bansal
//         Created:  Wed, 29 Jun 2016 16:27:31 GMT
//
//


// system include files
#include <memory>
#include <regex>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include "CondFormats/MuonSystemAging/interface/MuonSystemAging.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class ChamberMasker : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit ChamberMasker(const edm::ParameterSet&);
      ~ChamberMasker();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      std::vector<int> m_maskedRPCIDs;
      std::vector<std::string> m_maskedDTIDs;
      std::vector<int> m_maskedGE11PlusIDs;
      std::vector<int> m_maskedGE11MinusIDs;
      std::vector<int> m_maskedGE21PlusIDs;
      std::vector<int> m_maskedGE21MinusIDs;
      std::vector<int> m_maskedME0PlusIDs;
      std::vector<int> m_maskedME0MinusIDs;

      void createCSCAgingMap(edm::ESHandle<CSCGeometry> & cscGeom);
      std::vector<std::string> m_ChamberRegEx;
      std::map<uint32_t, std::pair<unsigned int, float>> m_CSCChambEffs;


      double m_ineffCSC;      

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ChamberMasker::ChamberMasker(const edm::ParameterSet& iConfig)

{  
   m_ineffCSC = iConfig.getParameter<double>("CSCineff"); 
   m_ChamberRegEx = iConfig.getParameter<std::vector<std::string>>("maskedChRegEx"); 
   for ( auto rpc_ids : iConfig.getParameter<std::vector<int>>("maskedRPCIDs"))
    {
      m_maskedRPCIDs.push_back(rpc_ids);
    }

    for ( auto ge11plus_ids : iConfig.getParameter<std::vector<int>>("maskedGE11PlusIDs"))
    {
      m_maskedGE11PlusIDs.push_back(ge11plus_ids);
    }

    for ( auto ge11minus_ids : iConfig.getParameter<std::vector<int>>("maskedGE11MinusIDs"))
    {
      m_maskedGE11MinusIDs.push_back(ge11minus_ids);
    }


   for ( auto ge21plus_ids : iConfig.getParameter<std::vector<int>>("maskedGE21PlusIDs"))
    {
      m_maskedGE21PlusIDs.push_back(ge21plus_ids);
    }

    for ( auto ge21minus_ids : iConfig.getParameter<std::vector<int>>("maskedGE21MinusIDs"))
    {
      m_maskedGE21MinusIDs.push_back(ge21minus_ids);
    }


    for ( auto me0plus_ids : iConfig.getParameter<std::vector<int>>("maskedME0PlusIDs"))
    {
      m_maskedME0PlusIDs.push_back(me0plus_ids);
    }

    for ( auto me0minus_ids : iConfig.getParameter<std::vector<int>>("maskedME0MinusIDs"))
    {
      m_maskedME0MinusIDs.push_back(me0minus_ids);
    }


    for ( auto regStr : iConfig.getParameter<std::vector<std::string>>("maskedChRegEx") )
    {
    m_maskedDTIDs.push_back(regStr);

    }


   //now do what ever initialization is needed
   usesResource("TFileService");

}


ChamberMasker::~ChamberMasker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
ChamberMasker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  edm::ESHandle<CSCGeometry> cscGeom;
  iSetup.get<MuonGeometryRecord>().get(cscGeom);

  createCSCAgingMap(cscGeom);

 MuonSystemAging* pList = new MuonSystemAging();
 for(unsigned int i = 0; i < m_maskedRPCIDs.size();++i){
 pList->m_RPCchambers.push_back(m_maskedRPCIDs.at(i));
 }
 for(unsigned int i = 0; i < m_maskedDTIDs.size();++i){
 pList->m_DTchambers.push_back(std::string(m_maskedDTIDs.at(i)));
 }

for(unsigned int i = 0; i < m_maskedGE11PlusIDs.size();++i){
 pList->m_GE11Pluschambers.push_back(m_maskedGE11PlusIDs.at(i));
 }

for(unsigned int i = 0; i < m_maskedGE11MinusIDs.size();++i){
 pList->m_GE11Minuschambers.push_back(m_maskedGE11MinusIDs.at(i));
 }

for(unsigned int i = 0; i < m_maskedGE21PlusIDs.size();++i){
 pList->m_GE21Pluschambers.push_back(m_maskedGE21PlusIDs.at(i));
 }

for(unsigned int i = 0; i < m_maskedGE21MinusIDs.size();++i){
 pList->m_GE21Minuschambers.push_back(m_maskedGE21MinusIDs.at(i));
 }

for(unsigned int i = 0; i < m_maskedME0PlusIDs.size();++i){
 pList->m_ME0Pluschambers.push_back(m_maskedME0PlusIDs.at(i));
 }

for(unsigned int i = 0; i < m_maskedME0MinusIDs.size();++i){
 pList->m_ME0Minuschambers.push_back(m_maskedME0MinusIDs.at(i));
 }



 pList->m_CSCChambEffs = m_CSCChambEffs;
 pList->m_CSCineff = m_ineffCSC; 
 edm::Service<cond::service::PoolDBOutputService> poolDbService;
 if( poolDbService.isAvailable() ) poolDbService->writeOne( pList, poolDbService->currentTime(),"MuonSystemAgingRcd" );

}


// ------------ method called once each job just before starting event loop  ------------
void 
ChamberMasker::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ChamberMasker::endJob() 
{
}

void
ChamberMasker::createCSCAgingMap(edm::ESHandle<CSCGeometry> & cscGeom)
{

    const auto chambers = cscGeom->chambers();

    for ( const auto *ch : chambers) {

        CSCDetId chId = ch->id();


        std::string chTag = (chId.zendcap() == 1 ? "ME+" : "ME-")
            + std::to_string(chId.station())
            + "/" + std::to_string(chId.ring())
            + "/" + std::to_string(chId.chamber());

        int type = 0;
        float eff = 1.;

        for (auto & chRegExStr : m_ChamberRegEx) {

            int loc = chRegExStr.find(":");
            // if there's no :, then we don't have to correct format
            if (loc < 0) continue;

            std::string effTag(chRegExStr.substr(loc));

            const std::regex chRegEx(chRegExStr.substr(0,chRegExStr.find(":")));
            const std::regex predicateRegEx("(\\d*,\\d*\\.\\d*)");

            std::smatch predicate;

            if ( std::regex_search(chTag, chRegEx) && std::regex_search(effTag, predicate, predicateRegEx)) {
                std::string predicateStr = predicate.str();
                std::string typeStr = predicateStr.substr(0,predicateStr.find(","));
                std::string effStr = predicateStr.substr(predicateStr.find(",")+1);
                type = std::atoi(typeStr.c_str());
                eff = std::atof(effStr.c_str());

                std::cout << "Setting chamber " << chTag << " to have inefficiency of " << eff << ", type " << type << std::endl;
            }

        } 

        // Note, layer 0 for chamber specification
        int rawId = chId.rawIdMaker(chId.endcap(), chId.station(), chId.ring(), chId.chamber(), 0);
        m_CSCChambEffs[rawId] = std::make_pair(type, eff);

    }

}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ChamberMasker);
