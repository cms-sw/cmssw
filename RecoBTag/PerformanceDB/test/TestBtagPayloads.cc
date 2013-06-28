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
   std::string ttbar_disc_algorithms[3] = {"TTBARDISCRIMBTAGCSV", "TTBARDISCRIMBTAGJP", "TTBARDISCRIMBTAGTCHP"};

   for (int alg = 0; alg<3; alg++) {

     name = ttbar_disc_algorithms[alg];

     std::cout <<"\033[22;32m  Studying performance with label "<<name <<"\033[0m"<<std::endl;
     iSetup.get<BTagPerformanceRecord>().get(name,perfH);   
     const BtagPerformance & perf = *(perfH.product());
     
     std::cout <<"    My Performance Object is indeed a "<<typeid(perfH.product()).name()<<std::endl;
     
     std::cout <<"    The WP is defined by a cut at "<<perf.workingPoint().cut()<<std::endl;
     std::cout <<"    Discriminant is "<<perf.workingPoint().discriminantName()<<std::endl;
     std::cout <<"    Is cut based WP "<<perf.workingPoint().cutBased()<<std::endl;
    
     float DiscValue[5] = {0.2, 0.4, 0.6, 0.8, 0.9};

     for (int ds = 0; ds<5; ds++) {

       float DiscCut = DiscValue[ds];
       if (alg==2) DiscCut *= 4.;
 
       p.insert(BinningVariables::JetEta,0.6);
       p.insert(BinningVariables::Discriminator,DiscCut);
       
       std::cout <<"      test eta=0.6, discrim = " << DiscCut<<std::endl;
       std::cout <<"      bSF/bFSerr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERRCORR,p)<<std::endl;
       std::cout <<"    bSF/bSFerr ="<<perf.getResult(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERRCORR,p)<<std::endl;
   
     }
       
     std::cout << std::endl;

   }

   std::cout << std::endl;
   std::cout << std::endl;

   //
   //++++++++++++------  TESTING FOR TTBAR WP's   --------+++++++++++++
   //
   
   printf("\033[22;31m TESTING FOR TTBAR WP's \n\033[0m");
   
   //Possible algorithms: TTBARWPBTAGCSVL,  TTBARWPBTAGCSVM,   TTBARWPBTAGCSVT
   //                     TTBARWPBTAGJPL,   TTBARWPBTAGJPM,    TTBARWPBTAGJPT
   //                                                          TTBARWPBTAGTCHPT
   std::string ttbar_wp_algorithms[7] = {"TTBARWPBTAGCSVL", "TTBARWPBTAGCSVM",  "TTBARWPBTAGCSVT",
					 "TTBARWPBTAGJPL",  "TTBARWPBTAGJPM",   "TTBARWPBTAGJPT",
					 "TTBARWPBTAGTCHPT"};

   for (int alg = 0; alg<7; alg++) {

     name = ttbar_wp_algorithms[alg];


     std::map<std::string,PerformanceResult::ResultType> measureMap;
     measureMap["BTAGBEFFCORR"]=PerformanceResult::BTAGBEFFCORR;
     measureMap["BTAGBERRCORR"]=PerformanceResult::BTAGBERRCORR;
     measureMap["BTAGCEFFCORR"]=PerformanceResult::BTAGCEFFCORR;
     measureMap["BTAGCERRCORR"]=PerformanceResult::BTAGCERRCORR;
     
     std::vector< std::string > measureName;
     std::vector< std::string > measureType;

     measureName.push_back(name);measureName.push_back(name);measureName.push_back(name);measureName.push_back(name);
     
     measureType.push_back("BTAGBEFFCORR");measureType.push_back("BTAGBERRCORR");measureType.push_back("BTAGCEFFCORR");measureType.push_back("BTAGCERRCORR");
     
     if( measureName.size() != measureType.size() )
       {
	 std::cout << "measureName, measureType size mismatch!" << std::endl;
	 exit(-1);
       }
     

     for( size_t iMeasure = 0; iMeasure < measureName.size(); iMeasure++ )
       {

	 std::cout << "\033[22;32m  Testing: " << measureName[ iMeasure ] << " of type " << measureType[ iMeasure ] << "\033[0m"<<std::endl;
	 //Setup our measurement
	 iSetup.get<BTagPerformanceRecord>().get( measureName[ iMeasure ],perfH);
	 const BtagPerformance & perf2 = *(perfH.product());
	 
	 //Working point
	 std::cout << "    Working point: " << perf2.workingPoint().cut() << std::endl;
	 //Setup the point we wish to test!
	 BinningPointByMap measurePoint;
	 measurePoint.insert(BinningVariables::JetEt,50);
	 measurePoint.insert(BinningVariables::JetEta,0.6);
	 
	 std::cout << "    Is it OK? " << perf2.isResultOk( measureMap[ measureType[ iMeasure] ], measurePoint)
		   << "    result at 50 GeV, 0,6 |eta| = " << perf2.getResult( measureMap[ measureType[ iMeasure] ], measurePoint)
		   << std::endl;
	 
       }
     
     std::cout << std::endl;

   }

   std::cout << std::endl;
   std::cout << std::endl;

   //
   //++++++++++++------  TESTING FOR Mu+Jets NoTTbar WP's   --------+++++++++++++
   //

   printf("\033[22;31m TESTING FOR Mu+Jets with No TTbar WP's \n\033[0m");
      

   //Possible algorithms: MUJETSWPBTAGNOTTBARCSVL,  MUJETSWPBTAGNOTTBARCSVM,   MUJETSWPBTAGNOTTBARCSVT
   //                     MUJETSWPBTAGNOTTBARJPL,   MUJETSWPBTAGNOTTBARJPM,    MUJETSWPBTAGNOTTBARJPT
   //                                                                          MUJETSWPBTAGNOTTBARTCHPT
   //                     MUJETSWPBTAGNOTTBARCSVV1L,  MUJETSWPBTAGNOTTBARCSVV1M,   MUJETSWPBTAGNOTTBARCSVV1T
   //                     MUJETSWPBTAGNOTTBARCSVSLV1L,  MUJETSWPBTAGNOTTBARCSVSLV1M,   MUJETSWPBTAGNOTTBARCSVSLV1T
   std::string mujetsnottbar_wp_algorithms[13] = {"MUJETSWPBTAGNOTTBARCSVL",  "MUJETSWPBTAGNOTTBARCSVM",   "MUJETSWPBTAGNOTTBARCSVT",
						  "MUJETSWPBTAGNOTTBARJPL",   "MUJETSWPBTAGNOTTBARJPM",    "MUJETSWPBTAGNOTTBARJPT",
						  "MUJETSWPBTAGNOTTBARTCHPT",
						  "MUJETSWPBTAGNOTTBARCSVV1L","MUJETSWPBTAGNOTTBARCSVV1M", "MUJETSWPBTAGNOTTBARCSVV1T",
						  "MUJETSWPBTAGNOTTBARCSVSLV1L",  "MUJETSWPBTAGNOTTBARCSVSLV1M",   "MUJETSWPBTAGNOTTBARCSVSLV1T"};

   for (int alg = 0; alg<13; alg++) {

     name = mujetsnottbar_wp_algorithms[alg];

     std::cout <<"\033[22;32m   Studying performance with label "<<name <<" \033[0m"<<std::endl;
     iSetup.get<BTagPerformanceRecord>().get(name,perfH);   
     const BtagPerformance & perf3 = *(perfH.product());
   
     std::cout <<"    My Performance Object is indeed a "<<typeid(perfH.product()).name()<<std::endl;
   
     std::cout <<"    The WP is defined by a cut at "<<perf3.workingPoint().cut()<<std::endl;
     std::cout <<"    Discriminant is "<<perf3.workingPoint().discriminantName()<<std::endl;
     std::cout <<"    Is cut based WP "<<perf3.workingPoint().cutBased()<<std::endl;


     float PtJet[17] = {25., 35., 45., 55., 65., 75., 90., 110., 140., 185., 235., 290., 360., 450., 550., 700., 1000.};
     
     for (int ptb = 0; ptb<17; ptb++) {

       if (alg>=10 && PtJet[ptb]>400.) continue;

       p.insert(BinningVariables::JetEta,0.6);
       p.insert(BinningVariables::JetEt,PtJet[ptb]);

       std::cout <<"      test eta=0.6, et = " << PtJet[ptb]<<std::endl;
       std::cout <<"      bSF/bFSerr ?"<<perf3.isResultOk(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf3.isResultOk(PerformanceResult::BTAGBERRCORR,p)<<std::endl;
       std::cout <<"      bSF/bSFerr ="<<perf3.getResult(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf3.getResult(PerformanceResult::BTAGBERRCORR,p)<<std::endl;
       
     }
     
     std::cout << std::endl;
     
   }

   std::cout << std::endl;
   std::cout << std::endl;
   

   //
   //++++++++++++------  TESTING FOR Mu+Jets with TTbar WP's   --------+++++++++++++
   //

   printf("\033[22;31m TESTING FOR Mu+Jets with TTbar WP's \n\033[0m");
      

   //Possible algorithms: MUJETSWPBTAGTTBARCSVL,  MUJETSWPBTAGTTBARCSVM,   MUJETSWPBTAGTTBARCSVT
   //                                                                          MUJETSWPBTAGTTBARTCHPT
   //                     MUJETSWPBTAGTTBARCSVV1L,  MUJETSWPBTAGTTBARCSVV1M,   MUJETSWPBTAGTTBARCSVV1T
   //                     MUJETSWPBTAGTTBARCSVSLV1L,  MUJETSWPBTAGTTBARCSVSLV1M,   MUJETSWPBTAGTTBARCSVSLV1T
   std::string mujetsttbar_wp_algorithms[10] = {"MUJETSWPBTAGTTBARCSVL",  "MUJETSWPBTAGTTBARCSVM",   "MUJETSWPBTAGTTBARCSVT",
						  "MUJETSWPBTAGTTBARTCHPT",
						  "MUJETSWPBTAGTTBARCSVV1L","MUJETSWPBTAGTTBARCSVV1M", "MUJETSWPBTAGTTBARCSVV1T",
						  "MUJETSWPBTAGTTBARCSVSLV1L",  "MUJETSWPBTAGTTBARCSVSLV1M",   "MUJETSWPBTAGTTBARCSVSLV1T"};

   for (int alg = 0; alg<10; alg++) {

     name = mujetsttbar_wp_algorithms[alg];

     std::cout <<"\033[22;32m   Studying performance with label "<<name <<" \033[0m"<<std::endl;
     iSetup.get<BTagPerformanceRecord>().get(name,perfH);   
     const BtagPerformance & perf3bis = *(perfH.product());
   
     std::cout <<"    My Performance Object is indeed a "<<typeid(perfH.product()).name()<<std::endl;
   
     std::cout <<"    The WP is defined by a cut at "<<perf3bis.workingPoint().cut()<<std::endl;
     std::cout <<"    Discriminant is "<<perf3bis.workingPoint().discriminantName()<<std::endl;
     std::cout <<"    Is cut based WP "<<perf3bis.workingPoint().cutBased()<<std::endl;


     float PtJet[17] = {25., 35., 45., 55., 65., 75., 90., 110., 140., 185., 235., 290., 360., 450., 550., 700., 1000.};
     
     for (int ptb = 0; ptb<17; ptb++) {

       if (alg>=7 && PtJet[ptb]>400.) continue;

       p.insert(BinningVariables::JetEta,0.6);
       p.insert(BinningVariables::JetEt,PtJet[ptb]);

       std::cout <<"      test eta=0.6, et = " << PtJet[ptb]<<std::endl;
       std::cout <<"      bSF/bFSerr ?"<<perf3bis.isResultOk(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf3bis.isResultOk(PerformanceResult::BTAGBERRCORR,p)<<std::endl;
       std::cout <<"      bSF/bSFerr ="<<perf3bis.getResult(PerformanceResult::BTAGBEFFCORR,p)<<"/"<<perf3bis.getResult(PerformanceResult::BTAGBERRCORR,p)<<std::endl;
       
     }
     
     std::cout << std::endl;
     
   }

   std::cout << std::endl;
   std::cout << std::endl;
   
   //
   //++++++++++++------  TESTING FOR Mu+Jets Mistags   --------+++++++++++++
   //
   
   printf("\033[22;31m TESTING FOR Mu+Jets Mistags \n\033[0m");      

   //Possible algorithms: MISTAGCSVL,     MISTAGCSVM,     MISTAGCSVT
   //                     MISTAGJPL,      MISTAGJPM,      MISTAGJPT                // Data period 2012 ABCD
   //                                                     MISTAGTCHPT
   //                     MISTAGCSVV1L,   MISTAGCSVV1M,   MISTAGCSVV1T
   //                     MISTAGCSVSLV1L, MISTAGCSVSLV1M, MISTAGCSVSLV1T
   std::string mistag_wp_algorithms[13] = {"MISTAGCSVL",     "MISTAGCSVM",     "MISTAGCSVT",
					   "MISTAGJPL",      "MISTAGJPM",      "MISTAGJPT",                
					   "MISTAGTCHPT",
					   "MISTAGCSVV1L",   "MISTAGCSVV1M",   "MISTAGCSVV1T",
					   "MISTAGCSVSLV1L", "MISTAGCSVSLV1M", "MISTAGCSVSLV1T"};

   for (int alg = 0; alg<13; alg++) {

     name = mistag_wp_algorithms[alg];
     
     std::cout <<"\033[22;32m    Studying performance with label "<<name <<"\033[0m"<<std::endl;
     iSetup.get<BTagPerformanceRecord>().get(name,perfH);   
     const BtagPerformance & perf4 = *(perfH.product());
   
     std::cout <<"    My Performance Object is indeed a "<<typeid(perfH.product()).name()<<std::endl;
     
     std::cout <<"    The WP is defined by a cut at "<<perf4.workingPoint().cut()<<std::endl;
     std::cout <<"    Discriminant is "<<perf4.workingPoint().discriminantName()<<std::endl;
     std::cout <<"    Is cut based WP "<<perf4.workingPoint().cutBased()<<std::endl;
     
     

     float PtJet[5] = {50., 100., 200., 400., 800.};
     float EtaJet[4] = {0.4, 0.9, 1.2, 2.0};

     for (int ptb = 0; ptb<5; ptb++) {
       for (int etab = 0; etab<4; etab++) {

	 if (etab>0 && (alg==2 || alg==5 || alg==6 || alg==9 || alg==12)) continue;

	 p.insert(BinningVariables::JetEta,EtaJet[etab]);
	 p.insert(BinningVariables::JetEt,PtJet[ptb]);
     
	 std::cout <<"      test eta=" << EtaJet[etab]<<", et = " << PtJet[ptb]<<std::endl;
	 //std::cout <<"      leff ?"<<perf4.isResultOk(PerformanceResult::BTAGLEFF,p)<<std::endl;
	 //std::cout <<"      leff ="<<perf4.getResult(PerformanceResult::BTAGLEFF,p)<<std::endl;
	 std::cout <<"      bSF/bFSerr ?"<<perf4.isResultOk(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf4.isResultOk(PerformanceResult::BTAGLERRCORR,p)<<std::endl;
	 std::cout <<"      bSF/bSFerr ="<<perf4.getResult(PerformanceResult::BTAGLEFFCORR,p)<<"/"<<perf4.getResult(PerformanceResult::BTAGLERRCORR,p)<<std::endl;
	 
       }
     }
     
     std::cout << std::endl;       
     
   }

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
