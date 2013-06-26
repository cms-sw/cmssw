// -*- C++ -*-
//
// Package:    TestBtagPayloads
// Class:      TestBtagPayloads
// 
/**\class TestBtagPayloads.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michael Segala
//         Created:  Wed Jun 20 02:47:47 CDT 2012
// $Id: TestBtagPayloads.cc,v 1.2 2013/01/31 17:54:44 msegala Exp $
//
//

// system include files
#include <memory>
#include <stdio.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "RecoBTag/Records/interface/BTagPerformanceRecord.h"
#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"
#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"


class TestBtagPayloads : public edm::EDAnalyzer {
   public:
      explicit TestBtagPayloads(const edm::ParameterSet&);
      ~TestBtagPayloads();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

};

TestBtagPayloads::TestBtagPayloads(const edm::ParameterSet& iConfig){}
TestBtagPayloads::~TestBtagPayloads(){}

//
// member functions
//

// ------------ method called for each event  ------------
void
TestBtagPayloads::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   edm::ESHandle<BtagPerformance> perfH;
   std::string name = "";
   BinningPointByMap p;

   
   //
   //++++++++++++------  TESTING FOR TTBAR SF's and efficiencies using CONTINIOUS DISCRIMINATORS      --------+++++++++++++
   //
   
   printf("\033[22;31m \n TESTING FOR TTBAR SF's and efficiencies using CONTINIOUS DISCRIMINATORS \n\033[0m");

   //Possible algorithms: TTBARDISCRIMBTAGCSV, TTBARDISCRIMBTAGJP, TTBARDISCRIMBTAGTCHP
   name = "TTBARDISCRIMBTAGCSV";

   std::cout <<" Studying performance with label "<<name <<std::endl;
   iSetup.get<BTagPerformanceRecord>().get(name,perfH);   
   const BtagPerformance & perf = *(perfH.product());
      
   std::cout <<" My Performance Object is indeed a "<<typeid(perfH.product()).name()<<std::endl;
   
   std::cout <<" The WP is defined by a cut at "<<perf.workingPoint().cut()<<std::endl;
   std::cout <<" Discriminant is "<<perf.workingPoint().discriminantName()<<std::endl;
   std::cout <<" Is cut based WP "<<perf.workingPoint().cutBased()<<std::endl;

   p.insert(BinningVariables::JetEta,0.6);
   p.insert(BinningVariables::Discriminator,0.23);

   std::cout <<" test eta=0.6, discrim = 0.23"<<std::endl;
   std::cout <<" bSF/bFSerr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERRCORR,p)<<std::endl;
   std::cout <<" bSF/bSFerr ="<<perf.getResult(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERRCORR,p)<<std::endl;

   std::cout << std::endl;
   std::cout << std::endl;
   
   //
   //++++++++++++------  TESTING FOR TTBAR WP's   --------+++++++++++++
   //
   
   printf("\033[22;31m TESTING FOR TTBAR WP's \n\033[0m");
   
   //Possible algorithms: TTBARWPBTAGCSVL,  TTBARWPBTAGCSVM,   TTBARWPBTAGCSVT
   //                     TTBARWPBTAGJPL,   TTBARWPBTAGJPM,    TTBARWPBTAGJPT
   //                                                          TTBARWPBTAGTCHPT


   std::map<std::string,PerformanceResult::ResultType> measureMap;
   measureMap["BTAGBEFFCORR"]=PerformanceResult::BTAGBEFFCORR;
   measureMap["BTAGBERRCORR"]=PerformanceResult::BTAGBERRCORR;
   measureMap["BTAGCEFFCORR"]=PerformanceResult::BTAGCEFFCORR;
   measureMap["BTAGCERRCORR"]=PerformanceResult::BTAGCERRCORR;

   std::vector< std::string > measureName;
   std::vector< std::string > measureType;

   measureName.push_back("TTBARWPBTAGCSVL");measureName.push_back("TTBARWPBTAGCSVL");measureName.push_back("TTBARWPBTAGCSVL");measureName.push_back("TTBARWPBTAGCSVL");
   measureName.push_back("TTBARWPBTAGJPT");measureName.push_back("TTBARWPBTAGJPT");measureName.push_back("TTBARWPBTAGJPT");measureName.push_back("TTBARWPBTAGJPT");

   measureType.push_back("BTAGBEFFCORR");measureType.push_back("BTAGBERRCORR");measureType.push_back("BTAGCEFFCORR");measureType.push_back("BTAGCERRCORR");
   measureType.push_back("BTAGBEFFCORR");measureType.push_back("BTAGBERRCORR");measureType.push_back("BTAGCEFFCORR");measureType.push_back("BTAGCERRCORR");
   

   if( measureName.size() != measureType.size() )
     {
       std::cout << "measureName, measureType size mismatch!" << std::endl;
       exit(-1);
     }


   for( size_t iMeasure = 0; iMeasure < measureName.size(); iMeasure++ )
     {
       std::cout << "Testing: " << measureName[ iMeasure ] << " of type " << measureType[ iMeasure ] << std::endl;

       //Setup our measurement
       iSetup.get<BTagPerformanceRecord>().get( measureName[ iMeasure ],perfH);
       const BtagPerformance & perf2 = *(perfH.product());

       //Working point
       std::cout << "Working point: " << perf2.workingPoint().cut() << std::endl;
       //Setup the point we wish to test!
       BinningPointByMap measurePoint;
       measurePoint.insert(BinningVariables::JetEt,50);
       measurePoint.insert(BinningVariables::JetEta,0.6);

       std::cout << "Is it OK? " << perf2.isResultOk( measureMap[ measureType[ iMeasure] ], measurePoint)
		 << " result at 50 GeV, 0,6 |eta| = " << perf2.getResult( measureMap[ measureType[ iMeasure] ], measurePoint)
		 << std::endl;

     }

   std::cout << std::endl;
   std::cout << std::endl;
   
   
   //
   //++++++++++++------  TESTING FOR Mu+Jets WP's   --------+++++++++++++
   //

   printf("\033[22;31m TESTING FOR Mu+Jets WP's \n\033[0m");
      

   //Possible algorithms: MUJETSWPBTAGCSVL,  MUJETSWPBTAGCSVM,   MUJETSWPBTAGCSVT
   //                     MUJETSWPBTAGJPL,   MUJETSWPBTAGJPM,    MUJETSWPBTAGJPT
   //                                                            MUJETSWPBTAGTCHPT


   name = "MUJETSWPBTAGCSVL";

   std::cout <<" Studying performance with label "<<name <<std::endl;
   iSetup.get<BTagPerformanceRecord>().get(name,perfH);   
   const BtagPerformance & perf3 = *(perfH.product());
   
   std::cout <<" My Performance Object is indeed a "<<typeid(perfH.product()).name()<<std::endl;
   
   std::cout <<" The WP is defined by a cut at "<<perf3.workingPoint().cut()<<std::endl;
   std::cout <<" Discriminant is "<<perf3.workingPoint().discriminantName()<<std::endl;
   std::cout <<" Is cut based WP "<<perf3.workingPoint().cutBased()<<std::endl;

   p.insert(BinningVariables::JetEta,0.6);
   p.insert(BinningVariables::JetEt,50);

   std::cout <<" test eta=0.6, et = 50"<<std::endl;
   std::cout <<" bSF/bFSerr ?"<<perf3.isResultOk(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf3.isResultOk(PerformanceResult::BTAGBERRCORR,p)<<std::endl;
   std::cout <<" bSF/bSFerr ="<<perf3.getResult(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf3.getResult(PerformanceResult::BTAGBERRCORR,p)<<std::endl;

   std::cout << std::endl;
   std::cout << std::endl;
   
   
   //
   //++++++++++++------  TESTING FOR Mu+Jets Mistags   --------+++++++++++++
   //
   
   printf("\033[22;31m TESTING FOR Mu+Jets Mistags \n\033[0m");      

   //Possible algorithms: MISTAGCSVLAB,  MISTAGCSVMAB,   MISTAGCSVTAB
   //                     MISTAGJPLAB,   MISTAGJPMAB,    MISTAGJPTAB                // Data period 2012 AB
   //                                                    MISTAGTCHPTAB
   //
   //                     MISTAGCSVLABCD,  MISTAGCSVMABCD,   MISTAGCSVTABCD
   //                     MISTAGJPLABCD,   MISTAGJPMABCD,    MISTAGJPTABCD          // Data period 2012 ABCD
   //                                                        MISTAGTCHPTABCD
   //
   //                     MISTAGCSVLC,  MISTAGCSVMC,   MISTAGCSVTC
   //                     MISTAGJPLC,   MISTAGJPMC,    MISTAGJPTC                   // Data period 2012 C
   //                                                  MISTAGTCHPTC
   //
   //                     MISTAGCSVLD,  MISTAGCSVMD,   MISTAGCSVTD
   //                     MISTAGJPLD,   MISTAGJPMD,    MISTAGJPTD                   // Data period 2012 D
   //                                                  MISTAGTCHPTD


   name = "MISTAGTCHPTD";

   std::cout <<" Studying performance with label "<<name <<std::endl;
   iSetup.get<BTagPerformanceRecord>().get(name,perfH);   
   const BtagPerformance & perf4 = *(perfH.product());
   
   std::cout <<" My Performance Object is indeed a "<<typeid(perfH.product()).name()<<std::endl;
   
   std::cout <<" The WP is defined by a cut at "<<perf4.workingPoint().cut()<<std::endl;
   std::cout <<" Discriminant is "<<perf4.workingPoint().discriminantName()<<std::endl;
   std::cout <<" Is cut based WP "<<perf4.workingPoint().cutBased()<<std::endl;

   p.insert(BinningVariables::JetEta,0.6);
   p.insert(BinningVariables::JetEt,50);

   std::cout <<" test eta=0.6, et = 50"<<std::endl;
   std::cout <<" leff ?"<<perf4.isResultOk(PerformanceResult::BTAGLEFF,p)<<std::endl;
   std::cout <<" leff ="<<perf4.getResult(PerformanceResult::BTAGLEFF,p)<<std::endl;
   std::cout <<" bSF/bFSerr ?"<<perf4.isResultOk(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf4.isResultOk(PerformanceResult::BTAGLERRCORR,p)<<std::endl;
   std::cout <<" bSF/bSFerr ="<<perf4.getResult(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf4.getResult(PerformanceResult::BTAGLERRCORR,p)<<std::endl;
   
   

}


// ------------ method called once each job just before starting event loop  ------------
void 
TestBtagPayloads::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestBtagPayloads::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestBtagPayloads);
