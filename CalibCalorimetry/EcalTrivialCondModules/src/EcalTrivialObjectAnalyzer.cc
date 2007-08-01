//
// $Id: EcalTrivialObjectAnalyzer.cc,v 1.10 2007/07/31 19:05:13 torimoto Exp $
// Created: 2 Mar 2006
//          Shahram Rahatlou, University of Rome & INFN
//
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialObjectAnalyzer.h"

#include <stdexcept>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
 
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
 
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"

//#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
//#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"

#include "CLHEP/Matrix/Matrix.h"

using namespace std;

  void
   EcalTrivialObjectAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<">>> EcalTrivialObjectAnalyzer: processing run "<<e.id().run() << " event: " << e.id().event() << std::endl;

    edm::ESHandle<EcalPedestals> pPeds;
    context.get<EcalPedestalsRcd>().get(pPeds);

    // ADC -> GeV Scale
    edm::ESHandle<EcalADCToGeVConstant> pAgc;
    context.get<EcalADCToGeVConstantRcd>().get(pAgc);
    const EcalADCToGeVConstant* agc = pAgc.product();
    std::cout << "Global ADC->GeV scale: EB " << std::setprecision(6) << agc->getEBValue() << " GeV/ADC count" 
	      " EE " << std::setprecision(6) << agc->getEEValue() << " GeV/ADC count" << std::endl;

    // use a channel to fetch values from DB
    double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
    int ieta =  int( 1 + r1*85 );
    r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
    int iphi =  int( 1 + r1*20 );

    EBDetId ebid(ieta,iphi); //eta,phi
    std::cout << "EcalTrivialObjectAnalyzer: using EBDetId: " << ebid << std::endl;

    const EcalPedestals* myped=pPeds.product();
    std::map<const unsigned int,EcalPedestals::Item>::const_iterator it=myped->m_pedestals.find(ebid.rawId());
    if( it!=myped->m_pedestals.end() ){
      std::cout << "EcalPedestal: "
                << "  mean_x1:  " << std::setprecision(8) << it->second.mean_x1 << " rms_x1: " << it->second.rms_x1
                << "  mean_x6:  " <<it->second.mean_x6 << " rms_x6: " << it->second.rms_x6
                << "  mean_x12: " <<it->second.mean_x12 << " rms_x12: " << it->second.rms_x12
                << std::endl;
    } else {
     std::cout << "No pedestal found for this xtal! something wrong with EcalPedestals in your DB? "
               << std::endl;
    }

    // fetch map of groups of xtals
    edm::ESHandle<EcalWeightXtalGroups> pGrp;
    context.get<EcalWeightXtalGroupsRcd>().get(pGrp);
    const EcalWeightXtalGroups* grp = pGrp.product();

    EcalWeightXtalGroups::EcalXtalGroupsMap::const_iterator git = grp->getMap().find( ebid.rawId() );
    EcalXtalGroupId gid;
    if( git != grp->getMap().end() ) {
      std::cout << "XtalGroupId.id() = " << std::setprecision(3) << git->second.id() << std:: endl;
      gid = git->second;
    } else {
      std::cout << "No group id found for this crystal. something wrong with EcalWeightXtalGroups in your DB?"
                << std::endl;
   }

    // Gain Ratios
    edm::ESHandle<EcalGainRatios> pRatio;
    context.get<EcalGainRatiosRcd>().get(pRatio);
    const EcalGainRatios* gr = pRatio.product();

    EcalGainRatios::EcalGainRatioMap::const_iterator grit=gr->getMap().find(ebid.rawId());
    EcalMGPAGainRatio mgpa;
    if( grit!=gr->getMap().end() ){
      mgpa = grit->second;

      std::cout << "EcalMGPAGainRatio: "
                << "gain 12/6 :  " << std::setprecision(4) << mgpa.gain12Over6() << " gain 6/1: " << mgpa.gain6Over1()
                << std::endl;
    } else {
     std::cout << "No MGPA Gain Ratio found for this xtal! something wrong with EcalGainRatios in your DB? "
               << std::endl;
    }

    // Intercalib constants
    edm::ESHandle<EcalIntercalibConstants> pIcal;
    context.get<EcalIntercalibConstantsRcd>().get(pIcal);
    const EcalIntercalibConstants* ical = pIcal.product();

    EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(ebid.rawId());
    EcalIntercalibConstants::EcalIntercalibConstant icalconst;
    if( icalit!=ical->getMap().end() ){
      icalconst = icalit->second;

      std::cout << "EcalIntercalibConstant: "
                <<std::setprecision(6)
                << icalconst
                << std::endl;
    } else {
     std::cout << "No intercalib const found for this xtal! something wrong with EcalIntercalibConstants in your DB? "
               << std::endl;
    }

   // fetch TB weights
   std::cout <<"Fetching EcalTBWeights from DB " << std::endl;
   edm::ESHandle<EcalTBWeights> pWgts;
   context.get<EcalTBWeightsRcd>().get(pWgts);
   const EcalTBWeights* wgts = pWgts.product();
   std::cout << "EcalTBWeightMap.size(): " << std::setprecision(3) << wgts->getMap().size() << std::endl;

   // look up the correct weights for this  xtal
   //EcalXtalGroupId gid( git->second );
   EcalTBWeights::EcalTDCId tdcid(1);

   std::cout << "Lookup EcalWeightSet for groupid: " << std::setprecision(3) 
             << gid.id() << " and TDC id " << tdcid << std::endl;
   EcalTBWeights::EcalTBWeightMap::const_iterator wit = wgts->getMap().find( std::make_pair(gid,tdcid) );
   EcalWeightSet  wset;
   if( wit != wgts->getMap().end() ) {
      wset = wit->second;
      std::cout << "check size of data members in EcalWeightSet" << std::endl;
      //wit->second.print(std::cout);


      //typedef std::vector< std::vector<EcalWeight> > EcalWeightMatrix;
      const EcalWeightSet::EcalWeightMatrix& mat1 = wit->second.getWeightsBeforeGainSwitch();
      const EcalWeightSet::EcalWeightMatrix& mat2 = wit->second.getWeightsAfterGainSwitch();

//       std::cout << "WeightsBeforeGainSwitch.size: " << mat1.size()
//                 << ", WeightsAfterGainSwitch.size: " << mat2.size() << std::endl;


      HepMatrix clmat1(3,10,0);
      HepMatrix clmat2(3,10,0);
      for(int irow=0; irow<3; irow++) {
       for(int icol=0; icol<10; icol++) {
         clmat1[irow][icol] = mat1(irow,icol);
         clmat2[irow][icol] = mat2(irow,icol);
       }
     }
     std::cout << "weight matrix before gain switch:" << std::endl;
     std::cout << clmat1 << std::endl;
     std::cout << "weight matrix after gain switch:" << std::endl;
     std::cout << clmat2 << std::endl;

   } else {
     std::cout << "No weights found for EcalGroupId: " << gid.id() << " and  EcalTDCId: " << tdcid << std::endl;
  }

   // laser correction

   // laser alphas
   edm::ESHandle<EcalLaserAlphas> pAlpha; 
   context.get<EcalLaserAlphasRcd>().get(pAlpha);
   const EcalLaserAlphas* lalpha = pAlpha.product();
   
   EcalLaserAlphas::EcalLaserAlphaMapIterator lalphait;
   lalphait = lalpha->getMap().find(ebid.rawId());
   EcalLaserAlphas::EcalLaserAlpha lalphaconst;

   if( lalphait!=lalpha->getMap().end() ){
     lalphaconst = lalphait->second;     
     std::cout << "EcalLaserAlpha: "
       //	       <<e.id().run() << " " << e.id().event() << " " 
	       <<std::setprecision(6)
	       << lalphaconst
	       << std::endl;
   } else {
     std::cout << "No laser alpha found for this xtal! something wrong with EcalLaserAlphas in your DB? "
               << std::endl;
   }

   // laser apdpnref
   edm::ESHandle<EcalLaserAPDPNRatiosRef> pAPDPNRatiosRef;          
   context.get<EcalLaserAPDPNRatiosRefRcd>().get(pAPDPNRatiosRef);  
   const EcalLaserAPDPNRatiosRef* lref = pAPDPNRatiosRef.product(); 
   
   EcalLaserAPDPNRatiosRef::EcalLaserAPDPNRatiosRefMapIterator lrefit;
   lrefit = lref->getMap().find(ebid.rawId());
   EcalLaserAPDPNRatiosRef::EcalLaserAPDPNref lrefconst;      

   if( lrefit!=lref->getMap().end() ){
     lrefconst = lrefit->second;     
     std::cout << "EcalLaserAPDPNRatiosRef: "
       //	       <<e.id().run() << " " << e.id().event() << " " 
  	       <<std::setprecision(6)
  	       << lrefconst
  	       << std::endl;
   } else {
     std::cout << "No laser apd/pn ref found for this xtal! something wrong with EcalLaserAPDPNRatiosRef in your DB? "
	       << std::endl;
   }


   // laser apdpn ratios 
   edm::ESHandle<EcalLaserAPDPNRatios> pAPDPNRatios;          
   context.get<EcalLaserAPDPNRatiosRcd>().get(pAPDPNRatios);  
   const EcalLaserAPDPNRatios* lratio = pAPDPNRatios.product();

   EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMapIterator lratioit; 
   lratioit = lratio->getLaserMap().find(ebid.rawId());
   EcalLaserAPDPNRatios::EcalLaserAPDPNpair lratioconst;
   
   if( lratioit!=lratio->getLaserMap().end() ){
     lratioconst = lratioit->second;
     
     std::cout << "EcalLaserAPDPNRatios: "
       //	       <<e.id().run() << " " << e.id().event() << " " 
    	       << std::setprecision(6)
    	       << lratioconst.p1 << " " << lratioconst.p2
    	       << std::endl;
   } else {
     std::cout << "No laser apd/pn ratio found for this xtal! something wrong with EcalLaserAPDPNRatios in your DB? "
  	       << std::endl;
   }

   // laser timestamps
   EcalLaserAPDPNRatios::EcalLaserTimeStampMapIterator ltimeit;
   EcalLaserAPDPNRatios::EcalLaserTimeStamp ltimestamp;
   
   std::cout << "EcalLaserTimeStamp: ";
   for (int i=1; i<=92; i++) {     
     ltimeit = lratio->getTimeMap().find(i);
     if( ltimeit != lratio->getTimeMap().end() ) {
       ltimestamp = ltimeit->second;
       
	 //		 <<e.id().run() << " " << e.id().event() << " " 
       std::cout << std::setprecision(6)
		   << ltimestamp.t1.value() << " " << ltimestamp.t2.value() << " : " ;
     } else {
       std::cout << "No laser timestamp found for this xtal! something wrong with EcalLaserAPDPNRatios in your DB? "
		 << std::endl;
     }
   }
   std::cout << std::endl;


   // laser transparency correction
   


/*
    std::cout << "make CLHEP matrices from vector<vector<Ecalweight>>" << std::endl;
    HepMatrix clmat1(3,8,0);
    HepMatrix clmat2(3,8,0);
    for(int irow=0; irow<3; irow++) {
     for(int icol=0; icol<8; icol++) {
       clmat1[irow][icol] = (mat1[irow])[icol]();
       clmat2[irow][icol] = (mat2[irow])[icol]();
     }
   }
   std::cout << clmat1 << std::endl;
   std::cout << clmat2 << std::endl;
*/

  } //end of ::Analyze()
  //DEFINE_FWK_MODULE(EcalTrivialObjectAnalyzer);
