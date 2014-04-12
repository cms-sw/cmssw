// -*- C++ -*-
//
// Package:    ZeePlots
// Class:      ZeePlots
//
/*

 Description: <one line class summary>
    this is an analyzer that reads pat::CompositeCandidate ZeeCandidates and creates some plots
    For more details see also WenuPlots class description
 Implementation:
  09Dec09: option to have a different selection for the 2nd leg of the Z added
  24Feb10: more variables added E/P and TIP
           option to choose CMSSW defined electron ID, the same or different
           for each leg

 Contact:
 Stilianos Kesisoglou - Institute of Nuclear Physics
 NCSR Demokritos

*/
//
// Original Author:  Nikolaos Rompotis


#include "ElectroWeakAnalysis/ZEE/interface/ZeePlots.h"

#include "DataFormats/Math/interface/deltaR.h"

ZeePlots::ZeePlots(const edm::ParameterSet& iConfig)
{

//                   I N P U T      P A R A M E T E R S
///////
//  ZEE COLLECTION   //////////////////////////////////////////////////////
//

    zeeCollectionToken_ = consumes<pat::CompositeCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("zeeCollectionTag"));

    // code parameters
    //
    std::string outputFile_D = "histos.root";
    outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", outputFile_D);

    ZEE_VBTFselectionFileName_ = iConfig.getUntrackedParameter<std::string>("ZEE_VBTFselectionFileName");
    ZEE_VBTFpreseleFileName_   = iConfig.getUntrackedParameter<std::string>("ZEE_VBTFpreseleFileName");

    DatasetTag_ = iConfig.getUntrackedParameter<Int_t>("DatasetTag");

    useSameSelectionOnBothElectrons_ = iConfig.getUntrackedParameter<Bool_t>("useSameSelectionOnBothElectrons",false);

    //  Here choose if the two legs will be treated individually or not.
    //
    if ( useSameSelectionOnBothElectrons_ ) {

        // use of precalculatedID. if you use it, then no other cuts are applied

        /*  Electron 1  */
        usePrecalcID1_ = iConfig.getUntrackedParameter<Bool_t>("usePrecalcID0",false);

        if ( usePrecalcID1_ ) {

            usePrecalcIDType1_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDType0");
            usePrecalcIDSign1_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDSign0","=");
            usePrecalcIDValue1_= iConfig.getUntrackedParameter<Double_t>("usePrecalcIDValue0");

            std::cout << "ZeePlots: WARNING: you have chosen to use CMSSW precalculated ID for electron #1 with name: >>> " << usePrecalcIDType1_<< " <<< such that the value map " << usePrecalcIDSign1_ << " "<< usePrecalcIDValue1_ << std::endl;
        }

        /*  Electron 2  */
        usePrecalcID2_ = iConfig.getUntrackedParameter<Bool_t>("usePrecalcID0",false);

        if ( usePrecalcID2_ ) {

            usePrecalcIDType2_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDType0");
            usePrecalcIDSign2_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDSign0","=");
            usePrecalcIDValue2_= iConfig.getUntrackedParameter<Double_t>("usePrecalcIDValue0");

            std::cout << "ZeePlots: WARNING: you have chosen to use CMSSW precalculated ID for electron #2 with name: >>> " << usePrecalcIDType2_<< " <<< such that the value map " << usePrecalcIDSign2_ << " "<< usePrecalcIDValue2_ << std::endl;
        }

        // use of preselection
        //
        useValidFirstPXBHit1_            = iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit0",false);
        useValidFirstPXBHit2_            = iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit0",false);

        useConversionRejection1_         = iConfig.getUntrackedParameter<Bool_t>("useConversionRejection0",false);
        useConversionRejection2_         = iConfig.getUntrackedParameter<Bool_t>("useConversionRejection0",false);

        useExpectedMissingHits1_         = iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits0",false);
        useExpectedMissingHits2_         = iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits0",false);

        maxNumberOfExpectedMissingHits1_  = iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits0",1);
        maxNumberOfExpectedMissingHits2_ = iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits0",1);


        // Selection Cuts:
        //

	     /*  Electron 1  */
	 	trackIso1_EB_         =  iConfig.getUntrackedParameter<Double_t>("trackIso0_EB",1000.0);       trackIso1_EE_         =  iConfig.getUntrackedParameter<Double_t>("trackIso0_EE",1000.0);
	 	ecalIso1_EB_          =  iConfig.getUntrackedParameter<Double_t>("ecalIso0_EB",1000.0);        ecalIso1_EE_          =  iConfig.getUntrackedParameter<Double_t>("ecalIso0_EE",1000.0);
	 	hcalIso1_EB_          =  iConfig.getUntrackedParameter<Double_t>("hcalIso0_EB",1000.0);        hcalIso1_EE_          =  iConfig.getUntrackedParameter<Double_t>("hcalIso0_EE",1000.0);

	 	sihih1_EB_            =  iConfig.getUntrackedParameter<Double_t>("sihih0_EB");                 sihih1_EE_            =  iConfig.getUntrackedParameter<Double_t>("sihih0_EE");
	 	dphi1_EB_             =  iConfig.getUntrackedParameter<Double_t>("dphi0_EB");                  dphi1_EE_             =  iConfig.getUntrackedParameter<Double_t>("dphi0_EE");
	 	deta1_EB_             =  iConfig.getUntrackedParameter<Double_t>("deta0_EB");                  deta1_EE_             =  iConfig.getUntrackedParameter<Double_t>("deta0_EE");
	 	hoe1_EB_              =  iConfig.getUntrackedParameter<Double_t>("hoe0_EB");                   hoe1_EE_              =  iConfig.getUntrackedParameter<Double_t>("hoe0_EE");
	 	cIso1_EB_             =  iConfig.getUntrackedParameter<Double_t>("cIso0_EB",1000.0);           cIso1_EE_             =  iConfig.getUntrackedParameter<Double_t>("cIso0_EE",1000.0);
	 	tip_bspot1_EB_        =  iConfig.getUntrackedParameter<Double_t>("tip_bspot0_EB",1000.0);      tip_bspot1_EE_        =  iConfig.getUntrackedParameter<Double_t>("tip_bspot0_EE",1000.0);
	 	eop1_EB_              =  iConfig.getUntrackedParameter<Double_t>("eop0_EB",1000.0);            eop1_EE_              =  iConfig.getUntrackedParameter<Double_t>("eop0_EE",1000.0);

	 	trackIsoUser1_EB_     =  iConfig.getUntrackedParameter<Double_t>("trackIsoUser0_EB",1000.0);   trackIsoUser1_EE_     =  iConfig.getUntrackedParameter<Double_t>("trackIsoUser0_EE",1000.0);
	 	ecalIsoUser1_EB_      =  iConfig.getUntrackedParameter<Double_t>("ecalIsoUser0_EB",1000.0);    ecalIsoUser1_EE_      =  iConfig.getUntrackedParameter<Double_t>("ecalIsoUser0_EE",1000.0);
	 	hcalIsoUser1_EB_      =  iConfig.getUntrackedParameter<Double_t>("hcalIsoUser0_EB",1000.0);    hcalIsoUser1_EE_      =  iConfig.getUntrackedParameter<Double_t>("hcalIsoUser0_EE",1000.0);

	     //  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	 	trackIso1_EB_inv      =  iConfig.getUntrackedParameter<Bool_t>("trackIso0_EB_inv",false);      trackIso1_EE_inv      =  iConfig.getUntrackedParameter<Bool_t>("trackIso0_EE_inv",false);
	 	ecalIso1_EB_inv       =  iConfig.getUntrackedParameter<Bool_t>("ecalIso0_EB_inv",false);       ecalIso1_EE_inv       =  iConfig.getUntrackedParameter<Bool_t>("ecalIso0_EE_inv",false);
	 	hcalIso1_EB_inv       =  iConfig.getUntrackedParameter<Bool_t>("hcalIso0_EB_inv",false);       hcalIso1_EE_inv       =  iConfig.getUntrackedParameter<Bool_t>("hcalIso0_EE_inv",false);

	 	sihih1_EB_inv         =  iConfig.getUntrackedParameter<Bool_t>("sihih0_EB_inv",false);         sihih1_EE_inv         =  iConfig.getUntrackedParameter<Bool_t>("sihih0_EE_inv",false);
	 	dphi1_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("dphi0_EB_inv",false);          dphi1_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("dphi0_EE_inv",false);
	 	deta1_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("deta0_EB_inv",false);          deta1_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("deta0_EE_inv",false);
	 	hoe1_EB_inv           =  iConfig.getUntrackedParameter<Bool_t>("hoe0_EB_inv",false);           hoe1_EE_inv           =  iConfig.getUntrackedParameter<Bool_t>("hoe0_EE_inv",false);
	 	cIso1_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("cIso0_EB_inv",false);          cIso1_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("cIso0_EE_inv",false);
	 	tip_bspot1_EB_inv     =  iConfig.getUntrackedParameter<Bool_t>("tip_bspot0_EB_inv",false);     tip_bspot1_EE_inv     =  iConfig.getUntrackedParameter<Bool_t>("tip_bspot0_EE_inv",false);
	 	eop1_EB_inv           =  iConfig.getUntrackedParameter<Bool_t>("eop0_EB_inv",false);           eop1_EE_inv           =  iConfig.getUntrackedParameter<Bool_t>("eop0_EE_inv",false);

	 	trackIsoUser1_EB_inv  =  iConfig.getUntrackedParameter<Bool_t>("trackIsoUser0_EB_inv",false);  trackIsoUser1_EE_inv  =  iConfig.getUntrackedParameter<Bool_t>("trackIsoUser0_EE_inv",false);
	 	ecalIsoUser1_EB_inv   =  iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser0_EB_inv",false);   ecalIsoUser1_EE_inv   =  iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser0_EE_inv",false);
	 	hcalIsoUser1_EB_inv   =  iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser0_EB_inv",false);   hcalIsoUser1_EE_inv   =  iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser0_EE_inv",false);

	     /*  Electron 2  */
	 	trackIso2_EB_         =  iConfig.getUntrackedParameter<Double_t>("trackIso0_EB",1000.0);       trackIso2_EE_         =  iConfig.getUntrackedParameter<Double_t>("trackIso0_EE",1000.0);
	 	ecalIso2_EB_          =  iConfig.getUntrackedParameter<Double_t>("ecalIso0_EB",1000.0);        ecalIso2_EE_          =  iConfig.getUntrackedParameter<Double_t>("ecalIso0_EE",1000.0);
	 	hcalIso2_EB_          =  iConfig.getUntrackedParameter<Double_t>("hcalIso0_EB",1000.0);        hcalIso2_EE_          =  iConfig.getUntrackedParameter<Double_t>("hcalIso0_EE",1000.0);

	 	sihih2_EB_            =  iConfig.getUntrackedParameter<Double_t>("sihih0_EB");                 sihih2_EE_            =  iConfig.getUntrackedParameter<Double_t>("sihih0_EE");
	 	dphi2_EB_             =  iConfig.getUntrackedParameter<Double_t>("dphi0_EB");                  dphi2_EE_             =  iConfig.getUntrackedParameter<Double_t>("dphi0_EE");
	 	deta2_EB_             =  iConfig.getUntrackedParameter<Double_t>("deta0_EB");                  deta2_EE_             =  iConfig.getUntrackedParameter<Double_t>("deta0_EE");
	 	hoe2_EB_              =  iConfig.getUntrackedParameter<Double_t>("hoe0_EB");                   hoe2_EE_              =  iConfig.getUntrackedParameter<Double_t>("hoe0_EE");
	 	cIso2_EB_             =  iConfig.getUntrackedParameter<Double_t>("cIso0_EB",1000.0);           cIso2_EE_             =  iConfig.getUntrackedParameter<Double_t>("cIso0_EE",1000.0);
	 	tip_bspot2_EB_        =  iConfig.getUntrackedParameter<Double_t>("tip_bspot0_EB",1000.0);      tip_bspot2_EE_        =  iConfig.getUntrackedParameter<Double_t>("tip_bspot0_EE",1000.0);
	 	eop2_EB_              =  iConfig.getUntrackedParameter<Double_t>("eop0_EB",1000.0);            eop2_EE_              =  iConfig.getUntrackedParameter<Double_t>("eop0_EE",1000.0);

	 	trackIsoUser2_EB_     =  iConfig.getUntrackedParameter<Double_t>("trackIsoUser0_EB",1000.0);   trackIsoUser2_EE_     =  iConfig.getUntrackedParameter<Double_t>("trackIsoUser0_EE",1000.0);
	 	ecalIsoUser2_EB_      =  iConfig.getUntrackedParameter<Double_t>("ecalIsoUser0_EB",1000.0);    ecalIsoUser2_EE_      =  iConfig.getUntrackedParameter<Double_t>("ecalIsoUser0_EE",1000.0);
	 	hcalIsoUser2_EB_      =  iConfig.getUntrackedParameter<Double_t>("hcalIsoUser0_EB",1000.0);    hcalIsoUser2_EE_      =  iConfig.getUntrackedParameter<Double_t>("hcalIsoUser0_EE",1000.0);

	     //  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	 	trackIso2_EB_inv      =  iConfig.getUntrackedParameter<Bool_t>("trackIso0_EB_inv",false);      trackIso2_EE_inv      =  iConfig.getUntrackedParameter<Bool_t>("trackIso0_EE_inv",false);
	 	ecalIso2_EB_inv       =  iConfig.getUntrackedParameter<Bool_t>("ecalIso0_EB_inv",false);       ecalIso2_EE_inv       =  iConfig.getUntrackedParameter<Bool_t>("ecalIso0_EE_inv",false);
	 	hcalIso2_EB_inv       =  iConfig.getUntrackedParameter<Bool_t>("hcalIso0_EB_inv",false);       hcalIso2_EE_inv       =  iConfig.getUntrackedParameter<Bool_t>("hcalIso0_EE_inv",false);

	 	sihih2_EB_inv         =  iConfig.getUntrackedParameter<Bool_t>("sihih0_EB_inv",false);         sihih2_EE_inv         =  iConfig.getUntrackedParameter<Bool_t>("sihih0_EE_inv",false);
	 	dphi2_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("dphi0_EB_inv",false);          dphi2_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("dphi0_EE_inv",false);
	 	deta2_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("deta0_EB_inv",false);          deta2_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("deta0_EE_inv",false);
	 	hoe2_EB_inv           =  iConfig.getUntrackedParameter<Bool_t>("hoe0_EB_inv",false);           hoe2_EE_inv           =  iConfig.getUntrackedParameter<Bool_t>("hoe0_EE_inv",false);
	 	cIso2_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("cIso0_EB_inv",false);          cIso2_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("cIso0_EE_inv",false);
	 	tip_bspot2_EB_inv     =  iConfig.getUntrackedParameter<Bool_t>("tip_bspot0_EB_inv",false);     tip_bspot2_EE_inv     =  iConfig.getUntrackedParameter<Bool_t>("tip_bspot0_EE_inv",false);
	 	eop2_EB_inv           =  iConfig.getUntrackedParameter<Bool_t>("eop0_EB_inv",false);           eop2_EE_inv           =  iConfig.getUntrackedParameter<Bool_t>("eop0_EE_inv",false);

	 	trackIsoUser2_EB_inv  =  iConfig.getUntrackedParameter<Bool_t>("trackIsoUser0_EB_inv",false);  trackIsoUser2_EE_inv  =  iConfig.getUntrackedParameter<Bool_t>("trackIsoUser0_EE_inv",false);
	 	ecalIsoUser2_EB_inv   =  iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser0_EB_inv",false);   ecalIsoUser2_EE_inv   =  iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser0_EE_inv",false);
	 	hcalIsoUser2_EB_inv   =  iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser0_EB_inv",false);   hcalIsoUser2_EE_inv   =  iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser0_EE_inv",false);

    }
    else {

        // use of precalculatedID. if you use it, then no other cuts are applied

        /*  Electron 1  */
        usePrecalcID1_ = iConfig.getUntrackedParameter<Bool_t>("usePrecalcID1",false);

        if ( usePrecalcID1_ ) {

            usePrecalcIDType1_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDType1");
            usePrecalcIDSign1_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDSign1","=");
            usePrecalcIDValue1_= iConfig.getUntrackedParameter<Double_t>("usePrecalcIDValue1");

            std::cout << "ZeePlots: WARNING: you have chosen to use CMSSW precalculated ID for electron #1 with name: >>> " << usePrecalcIDType1_<< " <<< such that the value map " << usePrecalcIDSign1_ << " "<< usePrecalcIDValue1_ << std::endl;
        }

        /*  Electron 2  */
        usePrecalcID2_ = iConfig.getUntrackedParameter<Bool_t>("usePrecalcID2",false);

        if ( usePrecalcID2_ ) {

            usePrecalcIDType2_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDType2");
            usePrecalcIDSign2_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDSign2","=");
            usePrecalcIDValue2_= iConfig.getUntrackedParameter<Double_t>("usePrecalcIDValue2");

            std::cout << "ZeePlots: WARNING: you have chosen to use CMSSW precalculated ID for electron #2 with name: >>> " << usePrecalcIDType2_<< " <<< such that the value map " << usePrecalcIDSign2_ << " "<< usePrecalcIDValue2_ << std::endl;
        }

        // use of preselection
        //
        useValidFirstPXBHit1_            = iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit1",false);
        useValidFirstPXBHit2_            = iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit2",false);

        useConversionRejection1_         = iConfig.getUntrackedParameter<Bool_t>("useConversionRejection1",false);
        useConversionRejection2_         = iConfig.getUntrackedParameter<Bool_t>("useConversionRejection2",false);

        useExpectedMissingHits1_         = iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits1",false);
        useExpectedMissingHits2_         = iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits2",false);

        maxNumberOfExpectedMissingHits1_  = iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits1",1);
        maxNumberOfExpectedMissingHits2_ = iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits2",1);


        // Selection Cuts:
        //

	     /*  Electron 1  */
	 	trackIso1_EB_         =  iConfig.getUntrackedParameter<Double_t>("trackIso1_EB",1000.0);       trackIso1_EE_         =  iConfig.getUntrackedParameter<Double_t>("trackIso1_EE",1000.0);
	 	ecalIso1_EB_          =  iConfig.getUntrackedParameter<Double_t>("ecalIso1_EB",1000.0);        ecalIso1_EE_          =  iConfig.getUntrackedParameter<Double_t>("ecalIso1_EE",1000.0);
	 	hcalIso1_EB_          =  iConfig.getUntrackedParameter<Double_t>("hcalIso1_EB",1000.0);        hcalIso1_EE_          =  iConfig.getUntrackedParameter<Double_t>("hcalIso1_EE",1000.0);

	 	sihih1_EB_            =  iConfig.getUntrackedParameter<Double_t>("sihih1_EB");                 sihih1_EE_            =  iConfig.getUntrackedParameter<Double_t>("sihih1_EE");
	 	dphi1_EB_             =  iConfig.getUntrackedParameter<Double_t>("dphi1_EB");                  dphi1_EE_             =  iConfig.getUntrackedParameter<Double_t>("dphi1_EE");
	 	deta1_EB_             =  iConfig.getUntrackedParameter<Double_t>("deta1_EB");                  deta1_EE_             =  iConfig.getUntrackedParameter<Double_t>("deta1_EE");
	 	hoe1_EB_              =  iConfig.getUntrackedParameter<Double_t>("hoe1_EB");                   hoe1_EE_              =  iConfig.getUntrackedParameter<Double_t>("hoe1_EE");
	 	cIso1_EB_             =  iConfig.getUntrackedParameter<Double_t>("cIso1_EB",1000.0);           cIso1_EE_             =  iConfig.getUntrackedParameter<Double_t>("cIso1_EE",1000.0);
	 	tip_bspot1_EB_        =  iConfig.getUntrackedParameter<Double_t>("tip_bspot1_EB",1000.0);      tip_bspot1_EE_        =  iConfig.getUntrackedParameter<Double_t>("tip_bspot1_EE",1000.0);
	 	eop1_EB_              =  iConfig.getUntrackedParameter<Double_t>("eop1_EB",1000.0);            eop1_EE_              =  iConfig.getUntrackedParameter<Double_t>("eop1_EE",1000.0);

	 	trackIsoUser1_EB_     =  iConfig.getUntrackedParameter<Double_t>("trackIsoUser1_EB",1000.0);   trackIsoUser1_EE_     =  iConfig.getUntrackedParameter<Double_t>("trackIsoUser1_EE",1000.0);
	 	ecalIsoUser1_EB_      =  iConfig.getUntrackedParameter<Double_t>("ecalIsoUser1_EB",1000.0);    ecalIsoUser1_EE_      =  iConfig.getUntrackedParameter<Double_t>("ecalIsoUser1_EE",1000.0);
	 	hcalIsoUser1_EB_      =  iConfig.getUntrackedParameter<Double_t>("hcalIsoUser1_EB",1000.0);    hcalIsoUser1_EE_      =  iConfig.getUntrackedParameter<Double_t>("hcalIsoUser1_EE",1000.0);
        //  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	 	trackIso1_EB_inv      =  iConfig.getUntrackedParameter<Bool_t>("trackIso1_EB_inv",false);      trackIso1_EE_inv      =  iConfig.getUntrackedParameter<Bool_t>("trackIso1_EE_inv",false);
	 	ecalIso1_EB_inv       =  iConfig.getUntrackedParameter<Bool_t>("ecalIso1_EB_inv",false);       ecalIso1_EE_inv       =  iConfig.getUntrackedParameter<Bool_t>("ecalIso1_EE_inv",false);
	 	hcalIso1_EB_inv       =  iConfig.getUntrackedParameter<Bool_t>("hcalIso1_EB_inv",false);       hcalIso1_EE_inv       =  iConfig.getUntrackedParameter<Bool_t>("hcalIso1_EE_inv",false);

	 	sihih1_EB_inv         =  iConfig.getUntrackedParameter<Bool_t>("sihih1_EB_inv",false);         sihih1_EE_inv         =  iConfig.getUntrackedParameter<Bool_t>("sihih1_EE_inv",false);
	 	dphi1_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("dphi1_EB_inv",false);          dphi1_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("dphi1_EE_inv",false);
	 	deta1_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("deta1_EB_inv",false);          deta1_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("deta1_EE_inv",false);
	 	hoe1_EB_inv           =  iConfig.getUntrackedParameter<Bool_t>("hoe1_EB_inv",false);           hoe1_EE_inv           =  iConfig.getUntrackedParameter<Bool_t>("hoe1_EE_inv",false);
	 	cIso1_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("cIso1_EB_inv",false);          cIso1_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("cIso1_EE_inv",false);
	 	tip_bspot1_EB_inv     =  iConfig.getUntrackedParameter<Bool_t>("tip_bspot1_EB_inv",false);     tip_bspot1_EE_inv     =  iConfig.getUntrackedParameter<Bool_t>("tip_bspot1_EE_inv",false);
	 	eop1_EB_inv           =  iConfig.getUntrackedParameter<Bool_t>("eop1_EB_inv",false);           eop1_EE_inv           =  iConfig.getUntrackedParameter<Bool_t>("eop1_EE_inv",false);

	 	trackIsoUser1_EB_inv  =  iConfig.getUntrackedParameter<Bool_t>("trackIsoUser1_EB_inv",false);  trackIsoUser1_EE_inv  =  iConfig.getUntrackedParameter<Bool_t>("trackIsoUser1_EE_inv",false);
	 	ecalIsoUser1_EB_inv   =  iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser1_EB_inv",false);   ecalIsoUser1_EE_inv   =  iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser1_EE_inv",false);
	 	hcalIsoUser1_EB_inv   =  iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser1_EB_inv",false);   hcalIsoUser1_EE_inv   =  iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser1_EE_inv",false);

	     /*  Electron 2  */
	 	trackIso2_EB_         =  iConfig.getUntrackedParameter<Double_t>("trackIso2_EB",1000.0);       trackIso2_EE_         =  iConfig.getUntrackedParameter<Double_t>("trackIso2_EE",1000.0);
	 	ecalIso2_EB_          =  iConfig.getUntrackedParameter<Double_t>("ecalIso2_EB",1000.0);        ecalIso2_EE_          =  iConfig.getUntrackedParameter<Double_t>("ecalIso2_EE",1000.0);
	 	hcalIso2_EB_          =  iConfig.getUntrackedParameter<Double_t>("hcalIso2_EB",1000.0);        hcalIso2_EE_          =  iConfig.getUntrackedParameter<Double_t>("hcalIso2_EE",1000.0);

	 	sihih2_EB_            =  iConfig.getUntrackedParameter<Double_t>("sihih2_EB");                 sihih2_EE_            =  iConfig.getUntrackedParameter<Double_t>("sihih2_EE");
	 	dphi2_EB_             =  iConfig.getUntrackedParameter<Double_t>("dphi2_EB");                  dphi2_EE_             =  iConfig.getUntrackedParameter<Double_t>("dphi2_EE");
	 	deta2_EB_             =  iConfig.getUntrackedParameter<Double_t>("deta2_EB");                  deta2_EE_             =  iConfig.getUntrackedParameter<Double_t>("deta2_EE");
	 	hoe2_EB_              =  iConfig.getUntrackedParameter<Double_t>("hoe2_EB");                   hoe2_EE_              =  iConfig.getUntrackedParameter<Double_t>("hoe2_EE");
	 	cIso2_EB_             =  iConfig.getUntrackedParameter<Double_t>("cIso2_EB",1000.0);           cIso2_EE_             =  iConfig.getUntrackedParameter<Double_t>("cIso2_EE",1000.0);
	 	tip_bspot2_EB_        =  iConfig.getUntrackedParameter<Double_t>("tip_bspot2_EB",1000.0);      tip_bspot2_EE_        =  iConfig.getUntrackedParameter<Double_t>("tip_bspot2_EE",1000.0);
	 	eop2_EB_              =  iConfig.getUntrackedParameter<Double_t>("eop2_EB",1000.0);            eop2_EE_              =  iConfig.getUntrackedParameter<Double_t>("eop2_EE",1000.0);

	 	trackIsoUser2_EB_     =  iConfig.getUntrackedParameter<Double_t>("trackIsoUser2_EB",1000.0);   trackIsoUser2_EE_     =  iConfig.getUntrackedParameter<Double_t>("trackIsoUser2_EE",1000.0);
	 	ecalIsoUser2_EB_      =  iConfig.getUntrackedParameter<Double_t>("ecalIsoUser2_EB",1000.0);    ecalIsoUser2_EE_      =  iConfig.getUntrackedParameter<Double_t>("ecalIsoUser2_EE",1000.0);
	 	hcalIsoUser2_EB_      =  iConfig.getUntrackedParameter<Double_t>("hcalIsoUser2_EB",1000.0);    hcalIsoUser2_EE_      =  iConfig.getUntrackedParameter<Double_t>("hcalIsoUser2_EE",1000.0);
        //  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	 	trackIso2_EB_inv      =  iConfig.getUntrackedParameter<Bool_t>("trackIso2_EB_inv",false);      trackIso2_EE_inv      =  iConfig.getUntrackedParameter<Bool_t>("trackIso2_EE_inv",false);
	 	ecalIso2_EB_inv       =  iConfig.getUntrackedParameter<Bool_t>("ecalIso2_EB_inv",false);       ecalIso2_EE_inv       =  iConfig.getUntrackedParameter<Bool_t>("ecalIso2_EE_inv",false);
	 	hcalIso2_EB_inv       =  iConfig.getUntrackedParameter<Bool_t>("hcalIso2_EB_inv",false);       hcalIso2_EE_inv       =  iConfig.getUntrackedParameter<Bool_t>("hcalIso2_EE_inv",false);

	 	sihih2_EB_inv         =  iConfig.getUntrackedParameter<Bool_t>("sihih2_EB_inv",false);         sihih2_EE_inv         =  iConfig.getUntrackedParameter<Bool_t>("sihih2_EE_inv",false);
	 	dphi2_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("dphi2_EB_inv",false);          dphi2_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("dphi2_EE_inv",false);
	 	deta2_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("deta2_EB_inv",false);          deta2_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("deta2_EE_inv",false);
	 	hoe2_EB_inv           =  iConfig.getUntrackedParameter<Bool_t>("hoe2_EB_inv",false);           hoe2_EE_inv           =  iConfig.getUntrackedParameter<Bool_t>("hoe2_EE_inv",false);
	 	cIso2_EB_inv          =  iConfig.getUntrackedParameter<Bool_t>("cIso2_EB_inv",false);          cIso2_EE_inv          =  iConfig.getUntrackedParameter<Bool_t>("cIso2_EE_inv",false);
	 	tip_bspot2_EB_inv     =  iConfig.getUntrackedParameter<Bool_t>("tip_bspot2_EB_inv",false);     tip_bspot2_EE_inv     =  iConfig.getUntrackedParameter<Bool_t>("tip_bspot2_EE_inv",false);
	 	eop2_EB_inv           =  iConfig.getUntrackedParameter<Bool_t>("eop2_EB_inv",false);           eop2_EE_inv           =  iConfig.getUntrackedParameter<Bool_t>("eop2_EE_inv",false);

	 	trackIsoUser2_EB_inv  =  iConfig.getUntrackedParameter<Bool_t>("trackIsoUser2_EB_inv",false);  trackIsoUser2_EE_inv  =  iConfig.getUntrackedParameter<Bool_t>("trackIsoUser2_EE_inv",false);
	 	ecalIsoUser2_EB_inv   =  iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser2_EB_inv",false);   ecalIsoUser2_EE_inv   =  iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser2_EE_inv",false);
	 	hcalIsoUser2_EB_inv   =  iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser2_EB_inv",false);   hcalIsoUser2_EE_inv   =  iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser2_EE_inv",false);

    }

    usePreselection1_ = ( useValidFirstPXBHit1_ || useConversionRejection1_ || useExpectedMissingHits1_ ) ? true : false ;
    usePreselection2_ = ( useValidFirstPXBHit2_ || useConversionRejection2_ || useExpectedMissingHits2_ ) ? true : false ;

    //  Display Massages
    //
    if ( useValidFirstPXBHit1_ )       {
        std::cout << "ZeePlots: Warning: you have demanded ValidFirstPXBHit on 1st electron" << std::endl;
    }
    if ( useValidFirstPXBHit2_ )      {
        std::cout << "ZeePlots: Warning: you have demanded ValidFirstPXBHit on 2nd electron" << std::endl;
    }
    if ( useConversionRejection1_ )    {
        std::cout << "ZeePlots: Warning: you have demanded Conversion Rejection on 1st electron" << std::endl;
    }
    if ( useConversionRejection2_ )   {
        std::cout << "ZeePlots: Warning: you have demanded Conversion Rejection on 2nd electron" << std::endl;
    }
    if ( useExpectedMissingHits1_ )    {
        std::cout << "ZeePlots: Warning: you have demanded Expected Missing Hits on 1st electron no more than " << maxNumberOfExpectedMissingHits1_    << std::endl;
    }
    if ( useExpectedMissingHits2_ )   {
        std::cout << "ZeePlots: Warning: you have demanded Expected Missing Hits on 2nd electron no more than " << maxNumberOfExpectedMissingHits2_    << std::endl;
    }


    //  JETS
    //
    includeJetInformationInNtuples_ = iConfig.getUntrackedParameter<Bool_t>("includeJetInformationInNtuples", false);

    if ( includeJetInformationInNtuples_ ) {

        caloJetCollectionToken_ = mayConsume< reco::CaloJetCollection >(iConfig.getUntrackedParameter<edm::InputTag>("caloJetCollectionTag"));
        pfJetCollectionToken_   = mayConsume< reco::PFJetCollection >(iConfig.getUntrackedParameter<edm::InputTag>("pfJetCollectionTag"));
        DRJetFromElectron_    = iConfig.getUntrackedParameter<Double_t>("DRJetFromElectron");

    }

}


ZeePlots::~ZeePlots()
{

    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void ZeePlots::analyze(const edm::Event& iEvent, const edm::EventSetup& es)
{
    using namespace std;
    //
    //  Get the collections here
    //
    edm::Handle<pat::CompositeCandidateCollection> ZeeCands;
    iEvent.getByToken(zeeCollectionToken_, ZeeCands);

    if ( ! ZeeCands.isValid() ) {
        std::cout << "Warning: No valid Zee candidates in this event..." << std::endl;
        return;
    }

    const pat::CompositeCandidateCollection *zcands = ZeeCands.product();
    const pat::CompositeCandidateCollection::const_iterator zeeIter = zcands->begin();
    const pat::CompositeCandidate zee = *zeeIter;

    // get the parts of the composite candidate:
    const pat::Electron * myElec1 = dynamic_cast<const pat::Electron*>( zee.daughter("electron1") );
    const pat::Electron * myElec2 = dynamic_cast<const pat::Electron*>( zee.daughter("electron2") );

    const pat::MET * myMet   = dynamic_cast<const pat::MET*>( zee.daughter("met") );
    const pat::MET * myPfMet = dynamic_cast<const pat::MET*>( zee.daughter("pfmet") );
    const pat::MET * myTcMet = dynamic_cast<const pat::MET*>( zee.daughter("tcmet") );

    // _______________________________________________________________________
    //
    // VBTF Root tuple production --------------------------------------------
    // _______________________________________________________________________
    //
    // .......................................................................
    // vbtf  produces 2 root tuples: one that contains the highest pT electrons
    //  that  passes a  user  defined selection  and one  other  with only the
    //  preselection criteria applied
    // .......................................................................
    //

    // fill the tree variables
    runNumber   = iEvent.run();
    eventNumber = (Long64_t)( iEvent.eventAuxiliary().event() );
    lumiSection = (Int_t)iEvent.luminosityBlock();

    ele1_sc_eta         = (Float_t)( myElec1->superCluster()->eta() );
    ele1_sc_phi         = (Float_t)( myElec1->superCluster()->phi() );
    ele1_sc_energy      = (Float_t)( myElec1->superCluster()->energy() );
    ele1_sc_gsf_et      = (Float_t)( myElec1->superCluster()->energy() / TMath::CosH(myElec1->gsfTrack()->eta()) );
    ele1_cand_eta       = (Float_t)( myElec1->eta() );
    ele1_cand_phi       = (Float_t)( myElec1->phi() );
    ele1_cand_et        = (Float_t)( myElec1->et() );

    ele1_iso_track      = (Float_t)( myElec1->dr03IsolationVariables().tkSumPt / ele1_cand_et );
    ele1_iso_ecal       = (Float_t)( myElec1->dr03IsolationVariables().ecalRecHitSumEt / ele1_cand_et );
    ele1_iso_hcal       = (Float_t)( ( myElec1->dr03IsolationVariables().hcalDepth1TowerSumEt + myElec1->dr03IsolationVariables().hcalDepth2TowerSumEt ) / ele1_cand_et );

    ele1_id_sihih       = (Float_t)( myElec1->sigmaIetaIeta() );
    ele1_id_deta        = (Float_t)( myElec1->deltaEtaSuperClusterTrackAtVtx() );
    ele1_id_dphi        = (Float_t)( myElec1->deltaPhiSuperClusterTrackAtVtx() );
    ele1_id_hoe         = (Float_t)( myElec1->hadronicOverEm() );

    ele1_cr_mhitsinner  = (Float_t)( myElec1->gsfTrack()->trackerExpectedHitsInner().numberOfHits() );
    ele1_cr_dcot        = (Float_t)( myElec1->userFloat("Dcot") );
    ele1_cr_dist        = (Float_t)( myElec1->userFloat("Dist") );

    ele1_vx             = (Float_t)( myElec1->vx() );
    ele1_vy             = (Float_t)( myElec1->vy() );
    ele1_vz             = (Float_t)( myElec1->vz() );

    pv_x1               = (Float_t)( myElec1->userFloat("pv_x") );
    pv_y1               = (Float_t)( myElec1->userFloat("pv_y") );
    pv_z1               = (Float_t)( myElec1->userFloat("pv_z") );

    ele1_gsfCharge      = (Int_t)  ( myElec1->gsfTrack()->charge() );
    ele1_ctfCharge      = (Int_t)  ( myElec1->closestCtfTrackRef().isNonnull() ? ( myElec1->closestCtfTrackRef()->charge() ) : -9999 ) ;
    ele1_scPixCharge    = (Int_t)  ( myElec1->chargeInfo().scPixCharge );
    ele1_eop            = (Float_t)( myElec1->eSuperClusterOverP() );
    ele1_tip_bs         = (Float_t)( (-1.0) * myElec1->dB() );
    ele1_tip_pv         = (Float_t)( myElec1->userFloat("ele_tip_pv") );


    ele2_sc_eta         = (Float_t)( myElec2->superCluster()->eta() );
    ele2_sc_phi         = (Float_t)( myElec2->superCluster()->phi() );
    ele2_sc_energy      = (Float_t)( myElec2->superCluster()->energy() );
    ele2_sc_gsf_et      = (Float_t)( myElec2->superCluster()->energy() / TMath::CosH(myElec2->gsfTrack()->eta()) );
    ele2_cand_eta       = (Float_t)( myElec2->eta() );
    ele2_cand_phi       = (Float_t)( myElec2->phi() );
    ele2_cand_et        = (Float_t)( myElec2->et() );

    ele2_iso_track      = (Float_t)( myElec2->dr03IsolationVariables().tkSumPt / ele2_cand_et );
    ele2_iso_ecal       = (Float_t)( myElec2->dr03IsolationVariables().ecalRecHitSumEt/ele2_cand_et );
    ele2_iso_hcal       = (Float_t)( ( myElec2->dr03IsolationVariables().hcalDepth1TowerSumEt + myElec2->dr03IsolationVariables().hcalDepth2TowerSumEt ) / ele2_cand_et );

    ele2_id_sihih       = (Float_t)( myElec2->sigmaIetaIeta() );
    ele2_id_deta        = (Float_t)( myElec2->deltaEtaSuperClusterTrackAtVtx() );
    ele2_id_dphi        = (Float_t)( myElec2->deltaPhiSuperClusterTrackAtVtx() );
    ele2_id_hoe         = (Float_t)( myElec2->hadronicOverEm() );

    ele2_cr_mhitsinner  = (Float_t)( myElec2->gsfTrack()->trackerExpectedHitsInner().numberOfHits() );
    ele2_cr_dcot        = (Float_t)( myElec2->userFloat("Dcot") );
    ele2_cr_dist        = (Float_t)( myElec2->userFloat("Dist") );

    ele2_vx             = (Float_t)( myElec2->vx() );
    ele2_vy             = (Float_t)( myElec2->vy() );
    ele2_vz             = (Float_t)( myElec2->vz() );

    pv_x2               = (Float_t)( myElec2->userFloat("pv_x") );
    pv_y2               = (Float_t)( myElec2->userFloat("pv_y") );
    pv_z2               = (Float_t)( myElec2->userFloat("pv_z") );

    ele2_gsfCharge      = (Int_t)  ( myElec2->gsfTrack()->charge() );
    ele2_ctfCharge      = (Int_t)  ( myElec2->closestCtfTrackRef().isNonnull() ? ( myElec2->closestCtfTrackRef()->charge() ) : -9999 );
    ele2_scPixCharge    = (Int_t)  ( myElec2->chargeInfo().scPixCharge );
    ele2_eop            = (Float_t)( myElec2->eSuperClusterOverP() );
    ele2_tip_bs         = (Float_t)( (-1.0) * myElec2->dB() );
    ele2_tip_pv         = (Float_t)( myElec2->userFloat("ele_tip_pv") );

    event_caloMET       = (Float_t)( myMet->et() );
    event_pfMET         = (Float_t)( myPfMet->et() );
    event_tcMET         = (Float_t)( myTcMet->et() );

    event_caloMET_phi   = (Float_t)( myMet->phi() );
    event_pfMET_phi     = (Float_t)( myPfMet->phi() );
    event_tcMET_phi     = (Float_t)( myTcMet->phi() );


    TLorentzVector p4e1;
    TLorentzVector p4e2;

    p4e1.SetPtEtaPhiM(ele1_sc_gsf_et, ele1_cand_eta, ele1_cand_phi, 0.000511);
    p4e2.SetPtEtaPhiM(ele2_sc_gsf_et, ele2_cand_eta, ele2_cand_phi, 0.000511);

    TLorentzVector Zp4 = p4e1 + p4e2 ;

    event_Mee = (Float_t)( Zp4.M() );

    event_datasetTag = DatasetTag_ ;

    // jet information - only if the user asks for it
    // keep the 5 highest et jets of the event that are further than DR > DRJetFromElectron_

    if ( includeJetInformationInNtuples_ ) {

        // initialize the array of the jet information

        for ( Int_t i=0; i < 5; ++i ) {

            calojet_et[i]  = -999999;
            calojet_eta[i] = -999999;
            calojet_phi[i] = -999999;

            pfjet_et[i]  = -999999;
            pfjet_eta[i] = -999999;
            pfjet_phi[i] = -999999;

        }

        // get hold of the jet collections
        edm::Handle< reco::CaloJetCollection > pCaloJets;
        iEvent.getByToken(caloJetCollectionToken_, pCaloJets);

        edm::Handle< reco::PFJetCollection > pPfJets;
        iEvent.getByToken(pfJetCollectionToken_, pPfJets);

        // calo jets now:
        if ( pCaloJets.isValid() ) {

            const reco::CaloJetCollection  *caloJets = pCaloJets.product();
            Int_t nCaloJets = (Int_t)( caloJets->size() );

            if ( nCaloJets > 0 ) {

                Float_t *nCaloET  = new Float_t[nCaloJets];
                Float_t *nCaloEta = new Float_t[nCaloJets];
                Float_t *nCaloPhi = new Float_t[nCaloJets];

//                 reco::CaloJetCollection::const_iterator cjet  =  caloJets->begin();

                Int_t counter = 0;

                for (reco::CaloJetCollection::const_iterator cjet = caloJets->begin() ; cjet != caloJets->end(); ++cjet) {

                    // store them only if they are far enough from the electron
                    Double_t DR1 = reco::deltaR(cjet->eta(), cjet->phi(), myElec1->gsfTrack()->eta(), ele1_sc_phi);
                    Double_t DR2 = reco::deltaR(cjet->eta(), cjet->phi(), myElec2->gsfTrack()->eta(), ele2_sc_phi);

                    if ( ( DR1 > DRJetFromElectron_ ) && ( DR2 > DRJetFromElectron_ ) ) {

                        nCaloET[counter]  = cjet->et();
                        nCaloEta[counter] = cjet->eta();
                        nCaloPhi[counter] = cjet->phi();
                        ++counter;
                    }
                }

                Int_t *caloJetSorted = new Int_t[nCaloJets];

                TMath::Sort(nCaloJets, nCaloET, caloJetSorted, true);

                for ( Int_t i = 0; i < nCaloJets; ++i ) {

                    if ( i >= 5 ) {
                        break;
                    }

                    calojet_et[i]  = nCaloET[ caloJetSorted[i] ];
                    calojet_eta[i] = nCaloEta[ caloJetSorted[i] ];
                    calojet_phi[i] = nCaloPhi[ caloJetSorted[i] ];
                }

                delete [] caloJetSorted;
                delete [] nCaloET;
                delete [] nCaloEta;
                delete [] nCaloPhi;
            }
        }

        // pf jets now:
        if ( pPfJets.isValid()) {

            const  reco::PFJetCollection  *pfJets = pPfJets.product();
            Int_t nPfJets = (Int_t) pfJets->size();

            if ( nPfJets > 0 ) {

                Float_t *nPfET  = new Float_t[nPfJets];
                Float_t *nPfEta = new Float_t[nPfJets];
                Float_t *nPfPhi = new Float_t[nPfJets];

//                 reco::PFJetCollection::const_iterator pjet  =  pfJets->begin();

                Int_t counter = 0;

                for (reco::PFJetCollection::const_iterator pjet = pfJets->begin(); pjet !=  pfJets->end(); ++pjet) {

                    // store them only if they are far enough from the electron

                    Double_t DR1 = reco::deltaR(pjet->eta(), pjet->phi(), myElec1->gsfTrack()->eta(), ele1_sc_phi);
                    Double_t DR2 = reco::deltaR(pjet->eta(), pjet->phi(), myElec2->gsfTrack()->eta(), ele2_sc_phi);

                    if ( ( DR1 > DRJetFromElectron_ ) && ( DR2 > DRJetFromElectron_ ) ) {

                        nPfET[counter]  = pjet->et();
                        nPfEta[counter] = pjet->eta();
                        nPfPhi[counter] = pjet->phi();
                        ++counter;
                    }
                }

                Int_t *pfJetSorted = new Int_t[nPfJets];

                TMath::Sort(nPfJets, nPfET, pfJetSorted, true);

                for ( Int_t i = 0; i < nPfJets; ++i ) {

                    if ( i >= 5 ) {
                        break;
                    }

                    pfjet_et[i]  = nPfET[ pfJetSorted[i] ];
                    pfjet_eta[i] = nPfEta[ pfJetSorted[i] ];
                    pfjet_phi[i] = nPfPhi[ pfJetSorted[i] ];

                }

                delete [] pfJetSorted;
                delete [] nPfET;
                delete [] nPfEta;
                delete [] nPfPhi;

            }
        }

    }

    // if the electrons pass the selection
    // it is meant to be a precalculated selection here, in order to include
    // conversion rejection too
    if ( CheckCuts1(myElec1) && CheckCuts2(myElec2) ) {
        vbtfSele_tree->Fill();
    }

    vbtfPresele_tree->Fill();



    //
    // _______________________________________________________________________
    //
    // histogram production --------------------------------------------------
    // _______________________________________________________________________
    //
    // if you want some preselection: Conv rejection, hit pattern

//     if ( usePreselection_ ) {
//
//         Bool_t a1 = PassPreselectionCriteria1(myElec1);
//         Bool_t a2 = PassPreselectionCriteria2(myElec2);
//
//         if ( ! (a1 && a2) ) {
//             return ;
//         }
//     }


    Bool_t passPre1 = ( usePreselection1_ ) ? PassPreselectionCriteria1(myElec1) : true ;
    Bool_t passPre2 = ( usePreselection2_ ) ? PassPreselectionCriteria2(myElec2) : true ;

    if ( ! ( passPre1 && passPre2 ) ) {

        std::cout << "At least one electron fails preselection: Electron #1 = " << passPre1 << " - Electron #2 = " << passPre2 << std::endl;

        return ;
    }


    TLorentzVector e1;
    TLorentzVector e2;

    //  math::XYZVector p1  =    myElec1->trackMomentumAtVtx();
    //  math::XYZVector p2  =    myElec2->trackMomentumAtVtx();
    //  e1.SetPxPyPzE(p1.X(), p1.Y(), p1.Z(), myElec1->caloEnergy());
    //  e2.SetPxPyPzE(p2.X(), p2.Y(), p2.Z(), myElec2->caloEnergy());

    // Use directly the et,eta,phi from pat::Electron; assume e mass  =  0.000511 GeV
    e1.SetPtEtaPhiM(myElec1->et(),myElec1->eta(),myElec1->phi(),0.000511);
    e2.SetPtEtaPhiM(myElec2->et(),myElec2->eta(),myElec2->phi(),0.000511);


    TLorentzVector Z = e1 + e2;

    Double_t mee = Z.M();

    // the selection plots:
    Bool_t pass = ( CheckCuts1(myElec1) && CheckCuts2(myElec2) ) ;

    //cout << "This event passes? " << pass << ", mee is: " << mee
    //   << " and the histo is filled." << endl;

    if ( ! pass ) {
        return ;
    }

    h_mee->Fill(mee);

    if ( TMath::Abs(e1.Eta())<1.479 && TMath::Abs(e2.Eta())<1.479 ) {
        h_mee_EBEB->Fill(mee) ;
    }
    if ( TMath::Abs(e1.Eta())<1.479 && TMath::Abs(e2.Eta())>1.479 ) {
        h_mee_EBEE->Fill(mee) ;
    }
    if ( TMath::Abs(e1.Eta())>1.479 && TMath::Abs(e2.Eta())<1.479 ) {
        h_mee_EBEE->Fill(mee) ;
    }
    if ( TMath::Abs(e1.Eta())>1.479 && TMath::Abs(e2.Eta())>1.479 ) {
        h_mee_EEEE->Fill(mee) ;
    }

    h_Zcand_PT->Fill(Z.Pt());
    h_Zcand_Y->Fill(Z.Rapidity());

    h_e_PT->Fill(e1.Pt());
    h_e_PT->Fill(e2.Pt());
    h_e_ETA->Fill(e1.Eta());
    h_e_ETA->Fill(e2.Eta());
    h_e_PHI->Fill(e1.Phi());
    h_e_PHI->Fill(e2.Phi());

    if ( TMath::Abs(myElec1->eta()) < 1.479 ) {
        h_EB_trkiso->Fill(ReturnCandVar(myElec1,0));
        h_EB_ecaliso->Fill(ReturnCandVar(myElec1,1));
        h_EB_hcaliso->Fill(ReturnCandVar(myElec1,2));
        h_EB_sIetaIeta->Fill(myElec1->scSigmaIEtaIEta());
        h_EB_dphi->Fill(myElec1->deltaPhiSuperClusterTrackAtVtx());
        h_EB_deta->Fill(myElec1->deltaEtaSuperClusterTrackAtVtx());
        h_EB_HoE->Fill(myElec1->hadronicOverEm());
    }
    else {
        h_EE_trkiso->Fill(ReturnCandVar(myElec1,0));
        h_EE_ecaliso->Fill(ReturnCandVar(myElec1,1));
        h_EE_hcaliso->Fill(ReturnCandVar(myElec1,2));
        h_EE_sIetaIeta->Fill(myElec1->scSigmaIEtaIEta());
        h_EE_dphi->Fill(myElec1->deltaPhiSuperClusterTrackAtVtx());
        h_EE_deta->Fill(myElec1->deltaEtaSuperClusterTrackAtVtx());
        h_EE_HoE->Fill(myElec1->hadronicOverEm());
    }

    if ( TMath::Abs(myElec2->eta()) < 1.479 ) {
        h_EB_trkiso->Fill(ReturnCandVar(myElec2,0));
        h_EB_ecaliso->Fill(ReturnCandVar(myElec2,1));
        h_EB_hcaliso->Fill(ReturnCandVar(myElec2,2));
        h_EB_sIetaIeta->Fill(myElec2->scSigmaIEtaIEta());
        h_EB_dphi->Fill(myElec2->deltaPhiSuperClusterTrackAtVtx());
        h_EB_deta->Fill(myElec2->deltaEtaSuperClusterTrackAtVtx());
        h_EB_HoE->Fill(myElec2->hadronicOverEm());
    }
    else {
        h_EE_trkiso->Fill(ReturnCandVar(myElec2,0));
        h_EE_ecaliso->Fill(ReturnCandVar(myElec2,1));
        h_EE_hcaliso->Fill(ReturnCandVar(myElec2,2));
        h_EE_sIetaIeta->Fill(myElec2->scSigmaIEtaIEta());
        h_EE_dphi->Fill(myElec2->deltaPhiSuperClusterTrackAtVtx());
        h_EE_deta->Fill(myElec2->deltaEtaSuperClusterTrackAtVtx());
        h_EE_HoE->Fill(myElec2->hadronicOverEm());
    }

    //Double_tscEta=myElec->superCluster()->eta();
    //Double_tscPhi=myElec->superCluster()->phi();
    //Double_tscEt=myElec->superCluster()->energy()/cosh(scEta);

}


/***********************************************************************
 *
 *  Checking Cuts and making selections:
 *  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *  all the available methods take input a pointer to a  pat::Electron
 *
 *  Bool_t  CheckCuts(const pat::Electron *):
 *                               true if the input selection is satisfied
 *  Bool_t  CheckCutsInverse(const pat::Electron *ele):
 *               true if the cuts with inverted the ones specified in the
 *               cfg are satisfied
 *  Bool_t  CheckCutsNminusOne(const pat::Electron *ele, Int_t jj):
 *               true if all the cuts with cut #jj ignored are satisfied
 *
 ***********************************************************************/
Bool_t ZeePlots::CheckCuts1( const pat::Electron *ele )
{
    if ( usePrecalcID1_ ) {

        if ( ! ele-> isElectronIDAvailable(usePrecalcIDType1_) ) {
            std::cout << "Error! not existing ID with name: " << usePrecalcIDType1_ << " function will return true!" << std::endl;
            return true;
        }

        Double_t val = ele->electronID( usePrecalcIDType1_ );

        if ( usePrecalcIDSign1_ == "<" ) {
            return ( val < usePrecalcIDValue1_ ) ;
        }
        else if ( usePrecalcIDSign1_ == ">" ) {
            return ( val > usePrecalcIDValue1_ ) ;
        }
        else { // equality: it returns 0,1,2,3 but as Float_t
            return ( TMath::Abs(val-usePrecalcIDValue1_) < 0.1 );
        }
    }
    else {

        for ( Int_t i = 0; i < nBarrelVars_; ++i ) {
            if ( ! CheckCut1(ele,i) ) {
                return false;
            }
        }

        return true;
    }
}


Bool_t ZeePlots::CheckCuts2( const pat::Electron *ele )
{
    if ( usePrecalcID2_ ) {
        if ( ! ele-> isElectronIDAvailable(usePrecalcIDType2_) ) {
            std::cout << "Error! not existing ID with name: " << usePrecalcIDType2_ << " function will return true!" << std::endl;
            return true;
        }

        Double_t val = ele->electronID( usePrecalcIDType2_ );

        if ( usePrecalcIDSign2_ == "<" ) {
            return ( val < usePrecalcIDValue2_ ) ;
        }
        else if ( usePrecalcIDSign2_  ==  ">" ) {
            return ( val > usePrecalcIDValue2_ ) ;
        }
        else { // equality: it returns 0,1,2,3 but as Float_t
            return ( TMath::Abs(val-usePrecalcIDValue2_) < 0.1 ) ;
        }
    }
    else {

        for ( Int_t i = 0; i < nBarrelVars_; ++i ) {
            if ( ! CheckCut2(ele, i) ) {
                return false;
            }
        }

        return true;
    }
}



Bool_t ZeePlots::CheckCuts1Inverse(const pat::Electron *ele)
{
    for ( Int_t i = 0; i < nBarrelVars_; ++i ) {

        if ( CheckCut1Inv(ele,i) == false ) {
            return false;
        }

    }

    return true;
}


Bool_t ZeePlots::CheckCuts2Inverse(const pat::Electron *ele)
{
    for (Int_t i = 0; i < nBarrelVars_; ++i ) {

        if ( CheckCut2Inv(ele,i) == false ) {
            return false;
        }
    }

    return true;
}



Bool_t ZeePlots::CheckCuts1NminusOne(const pat::Electron *ele, Int_t jj)
{
    for ( Int_t i = 0; i < nBarrelVars_ ; ++i ) {

        if ( i == jj ) {
            continue;
        }

        if ( CheckCut1(ele, i) == false ) {
            return false;
        }

    }

    return true;
}

Bool_t ZeePlots::CheckCuts2NminusOne(const pat::Electron *ele, Int_t jj)
{
    for ( Int_t i = 0; i < nBarrelVars_; ++i ) {

        if ( i == jj ) {
            continue;
        }

        if ( CheckCut2(ele, i) == false ) {
            return false;
        }
    }

    return true;
}




Bool_t ZeePlots::CheckCut1(const pat::Electron *ele, Int_t i) {

    Double_t fabseta = TMath::Abs(ele->superCluster()->eta());

    if ( fabseta < 1.479 ) {
        return ( TMath::Abs(ReturnCandVar(ele, i)) < CutVars1_[i] ) ;
    }

    return ( TMath::Abs(ReturnCandVar(ele, i)) < CutVars1_[i+nBarrelVars_] ) ;
}


Bool_t ZeePlots::CheckCut2(const pat::Electron *ele, Int_t i) {

    Double_t fabseta = TMath::Abs(ele->superCluster()->eta());

    if ( fabseta < 1.479 ) {
        return ( TMath::Abs(ReturnCandVar(ele, i)) < CutVars2_[i] ) ;
    }

    return ( TMath::Abs(ReturnCandVar(ele, i)) < CutVars2_[i+nBarrelVars_] ) ;
}



Bool_t ZeePlots::CheckCut1Inv(const pat::Electron *ele, Int_t i) {

    Double_t fabseta = TMath::Abs(ele->superCluster()->eta());

    if ( fabseta < 1.479 ) {

        if ( InvVars1_[i] ) {
            return ( TMath::Abs(ReturnCandVar(ele, i)) > CutVars1_[i] ) ;
        }

        return ( TMath::Abs(ReturnCandVar(ele, i)) < CutVars1_[i] ) ;

    }

    if ( InvVars1_[i+nBarrelVars_] ) {
        if ( InvVars1_[i] ) {
            return ( TMath::Abs(ReturnCandVar(ele, i)) > CutVars1_[i+nBarrelVars_] ) ;
        }
    }

    return ( TMath::Abs(ReturnCandVar(ele, i)) < CutVars1_[i+nBarrelVars_] ) ;
}


Bool_t ZeePlots::CheckCut2Inv(const pat::Electron *ele, Int_t i) {

    Double_t fabseta = TMath::Abs(ele->superCluster()->eta());

    if ( fabseta < 1.479 ) {

        if ( InvVars2_[i] ) {
            return ( TMath::Abs(ReturnCandVar(ele, i)) > CutVars2_[i] ) ;
        }

        return ( TMath::Abs(ReturnCandVar(ele, i)) < CutVars2_[i] ) ;
    }

    if ( InvVars2_[i+nBarrelVars_] ) {
        if ( InvVars2_[i] ) {
            return ( TMath::Abs(ReturnCandVar(ele, i)) > CutVars2_[i+nBarrelVars_] ) ;
        }
    }

    return ( TMath::Abs(ReturnCandVar(ele, i)) < CutVars2_[i+nBarrelVars_] ) ;
}



Double_t ZeePlots::ReturnCandVar(const pat::Electron *ele, Int_t i) {

    if      ( i ==  0 ) {
        return ( ele->dr03TkSumPt() / ele->p4().Pt() ) ;
    }
    else if ( i ==  1 ) {
        return ( ele->dr03EcalRecHitSumEt() / ele->p4().Pt() ) ;
    }
    else if ( i ==  2 ) {
        return ( ele->dr03HcalTowerSumEt() / ele->p4().Pt() ) ;
    }
    else if ( i ==  3 ) {
        return ( ele->scSigmaIEtaIEta() ) ;
    }
    else if ( i ==  4 ) {
        return ( ele->deltaPhiSuperClusterTrackAtVtx() ) ;
    }
    else if ( i ==  5 ) {
        return ( ele->deltaEtaSuperClusterTrackAtVtx() ) ;
    }
    else if ( i ==  6 ) {
        return ( ele->hadronicOverEm() ) ;
    }
    else if ( i ==  7 ) {
        // pedestal subtraction is only in barrel
        if ( ele->isEB() ) {
            return ( ( ele->dr03TkSumPt() + TMath::Max( 0.0 , ele->dr03EcalRecHitSumEt()-1.0 ) + ele->dr03HcalTowerSumEt() ) / ele->p4().Pt() );
        }
        else {
            return ( ( ele->dr03TkSumPt() + ele->dr03EcalRecHitSumEt() + ele->dr03HcalTowerSumEt() ) / ele->p4().Pt() ) ;
        }
    }
//     else if ( i ==  8 ) { return ele->gsfTrack()->dxy(bspotPosition_); }
    else if ( i ==  8 ) {
        return ( ele->dB() ) ;
    }
    else if ( i ==  9 ) {
        return ( ele->eSuperClusterOverP() ) ;
    }
    else if ( i == 10 ) {
        return ( ele->userIsolation(pat::TrackIso) ) ;
    }
    else if ( i == 11 ) {
        return ( ele->userIsolation(pat::EcalIso) ) ;
    }
    else if ( i == 12 ) {
        return ( ele->userIsolation(pat::HcalIso) ) ;
    }

    std::cout << "Error in ZeePlots::ReturnCandVar" << std::endl;

    return (-1.0) ;

}


//Bool_t ZeePlots::CheckCuts2( const pat::Electron *ele)
//{
//  for (Int_t i = 0; i<nBarrelVars_; ++i) {
//    if ( ! CheckCut2(ele, i)) return false;
//  }
//  return true;
//}



//
// special preselection criteria
Bool_t ZeePlots::PassPreselectionCriteria1(const pat::Electron *ele) {

    Bool_t passConvRej = true;
    Bool_t passPXB     = true;
    Bool_t passEMH     = true;

    if ( useConversionRejection1_ ) {

        if ( ele->hasUserInt("PassConversionRejection") ) {     //std::cout << "con rej: " << ele->userInt("PassConversionRejection") << std::endl;

            if ( ! (ele->userInt("PassConversionRejection") == 1) ) {
                passConvRej = false;
            }
        }
        else {
            std::cout << "ZeePlots: WARNING: Conversion Rejection Request for electron #1 Disregarded: " << "you must calculate it before " << std::endl;
        }
    }

    if ( useValidFirstPXBHit1_ ) {

        if ( ele->hasUserInt("PassValidFirstPXBHit") ) {    //std::cout << "valid1stPXB: " << ele->userInt("PassValidFirstPXBHit") << std::endl;

            if ( ! (ele->userInt("PassValidFirstPXBHit") == 1) ) {
                passPXB = false;
            }
        }
        else {
            std::cout << "ZeePlots: WARNING: Valid First PXB Hit Request for electron #1 Disregarded: " << "you must calculate it before " << std::endl;
        }
    }

    if ( useExpectedMissingHits1_ ) {

        if ( ele->hasUserInt("NumberOfExpectedMissingHits") ) {     //std::cout << "missing hits: " << ele->userInt("NumberOfExpectedMissingHits") << std::endl;

            if ( ! (ele->userInt("NumberOfExpectedMissingHits") <= maxNumberOfExpectedMissingHits1_) ) {
                passEMH = false;
            }
        }
        else {
            std::cout << "ZeePlots: WARNING: Number of Expected Missing Hits Request for electron #1 Disregarded: " << "you must calculate it before " << std::endl;
        }
    }

    return ( passConvRej && passPXB && passEMH ) ;
}

Bool_t ZeePlots::PassPreselectionCriteria2(const pat::Electron *ele) {

    Bool_t passConvRej = true;
    Bool_t passPXB     = true;
    Bool_t passEMH     = true;

    if ( useConversionRejection2_ ) {

        if ( ele->hasUserInt("PassConversionRejection") ) {     //std::cout << "con rej: " << ele->userInt("PassConversionRejection") << std::endl;

            if ( ! (ele->userInt("PassConversionRejection") == 1) ) {
                passConvRej = false;
            }
        }
        else {
            std::cout << "ZeePlots: WARNING: Conversion Rejection Request for electron #2 Disregarded: " << "you must calculate it before " << std::endl;
        }
    }

    if ( useValidFirstPXBHit2_ ) {

        if ( ele->hasUserInt("PassValidFirstPXBHit") ) {    //std::cout << "valid1stPXB: " << ele->userInt("PassValidFirstPXBHit") << std::endl;

            if ( ! (ele->userInt("PassValidFirstPXBHit") == 1) ) {
                passPXB = false;
            }
        }
        else {
            std::cout << "ZeePlots: WARNING: Valid First PXB Hit Request for electron #2 Disregarded: " << "you must calculate it before " << std::endl;
        }
    }

    if ( useExpectedMissingHits2_ ) {

        if ( ele->hasUserInt("NumberOfExpectedMissingHits") ) {     //std::cout << "missing hits: " << ele->userInt("NumberOfExpectedMissingHits") << std::endl;

            if ( ! (ele->userInt("NumberOfExpectedMissingHits") <= maxNumberOfExpectedMissingHits2_) ) {
                passEMH = false;
            }
        }
        else {
            std::cout << "ZeePlots: WARNING: Number of Expected Missing Hits Request for electron #2 Disregarded: " << "you must calculate it before " << std::endl;
        }
    }

    return ( passConvRej && passPXB && passEMH ) ;
}





//
// ------------ method called once each job just before starting event loop  --
void
ZeePlots::beginJob()
{
    //std::cout << "In beginJob()" << std::endl;

    h_mee      = new TH1F("h_mee"      , "h_mee"      , 200 ,  0.0 , 200.0) ;
    h_mee_EBEB = new TH1F("h_mee_EBEB" , "h_mee_EBEB" , 200 ,  0.0 , 200.0) ;
    h_mee_EBEE = new TH1F("h_mee_EBEE" , "h_mee_EBEE" , 200 ,  0.0 , 200.0) ;
    h_mee_EEEE = new TH1F("h_mee_EEEE" , "h_mee_EEEE" , 200 ,  0.0 , 200.0) ;
    h_Zcand_PT = new TH1F("h_Zcand_PT" , "h_Zcand_PT" , 200 ,  0.0 , 100.0) ;
    h_Zcand_Y  = new TH1F("h_Zcand_Y"  , "h_Zcand_Y"  , 200 , -5.0 ,   5.0) ;
    h_e_PT     = new TH1F("h_e_PT"     , "h_e_PT"     , 200 ,  0.0 , 100.0) ;
    h_e_ETA    = new TH1F("h_e_ETA"    , "h_e_ETA"    , 200 , -3.0 ,   3.0) ;
    h_e_PHI    = new TH1F("h_e_PHI"    , "h_e_PHI"    , 200 , -4.0 ,   4.0) ;


    //VALIDATION PLOTS
    //  EB
    h_EB_trkiso    = new TH1F("h_EB_trkiso"    , "h_EB_trkiso"    , 200 ,  0.00 , 9.00) ;
    h_EB_ecaliso   = new TH1F("h_EB_ecaliso"   , "h_EB_ecaliso"   , 200 ,  0.00 , 9.00) ;
    h_EB_hcaliso   = new TH1F("h_EB_hcaliso"   , "h_EB_hcaliso"   , 200 ,  0.00 , 9.00) ;
    h_EB_sIetaIeta = new TH1F("h_EB_sIetaIeta" , "h_EB_sIetaIeta" , 200 ,  0.00 , 0.02) ;
    h_EB_dphi      = new TH1F("h_EB_dphi"      , "h_EB_dphi"      , 200 , -0.03 , 0.03) ;
    h_EB_deta      = new TH1F("h_EB_deta"      , "h_EB_deta"      , 200 , -0.01 , 0.01) ;
    h_EB_HoE       = new TH1F("h_EB_HoE"       , "h_EB_HoE"       , 200 ,  0.00 , 0.20) ;
    //  EE
    h_EE_trkiso    = new TH1F("h_EE_trkiso"    , "h_EE_trkiso"    , 200 ,  0.00 , 9.00) ;
    h_EE_ecaliso   = new TH1F("h_EE_ecaliso"   , "h_EE_ecaliso"   , 200 ,  0.00 , 9.00) ;
    h_EE_hcaliso   = new TH1F("h_EE_hcaliso"   , "h_EE_hcaliso"   , 200 ,  0.00 , 9.00) ;
    h_EE_sIetaIeta = new TH1F("h_EE_sIetaIeta" , "h_EE_sIetaIeta" , 200 ,  0.00 , 0.10) ;
    h_EE_dphi      = new TH1F("h_EE_dphi"      , "h_EE_dphi"      , 200 , -0.03 , 0.03) ;
    h_EE_deta      = new TH1F("h_EE_deta"      , "h_EE_deta"      , 200 , -0.01 , 0.01) ;
    h_EE_HoE       = new TH1F("h_EE_HoE"       , "h_EE_HoE"       , 200 ,  0.00 , 0.20) ;


    // if you add some new variable change the nBarrelVars_ accordingly
    // reminder: in the current implementation you must have the same number
    //  of vars in both barrel and endcaps

    nBarrelVars_ = 13;

    //
    // Put EB variables together and EE variables together
    // number of barrel variables  =  number of endcap variable
    // if you don't want to use some variable put a very high cut

    //  1st Leg variables
    CutVars1_.push_back( trackIso1_EB_ )       ;  // 0
    CutVars1_.push_back( ecalIso1_EB_ )        ;  // 1
    CutVars1_.push_back( hcalIso1_EB_ )        ;  // 2
    CutVars1_.push_back( sihih1_EB_ )          ;  // 3
    CutVars1_.push_back( dphi1_EB_ )           ;  // 4
    CutVars1_.push_back( deta1_EB_ )           ;  // 5
    CutVars1_.push_back( hoe1_EB_ )            ;  // 6
    CutVars1_.push_back( cIso1_EB_ )           ;  // 7
    CutVars1_.push_back( tip_bspot1_EB_ )      ;  // 8
    CutVars1_.push_back( eop1_EB_ )            ;  // 9
    CutVars1_.push_back( trackIsoUser1_EB_ )   ;  // 10
    CutVars1_.push_back( ecalIsoUser1_EB_ )    ;  // 11
    CutVars1_.push_back( hcalIsoUser1_EB_ )    ;  // 12

    CutVars1_.push_back( trackIso1_EE_ )       ;  // 0
    CutVars1_.push_back( ecalIso1_EE_ )        ;  // 1
    CutVars1_.push_back( hcalIso1_EE_ )        ;  // 2
    CutVars1_.push_back( sihih1_EE_ )          ;  // 3
    CutVars1_.push_back( dphi1_EE_ )           ;  // 4
    CutVars1_.push_back( deta1_EE_ )           ;  // 5
    CutVars1_.push_back( hoe1_EE_ )            ;  // 6
    CutVars1_.push_back( cIso1_EE_ )           ;  // 7
    CutVars1_.push_back( tip_bspot1_EE_ )      ;  // 8
    CutVars1_.push_back( eop1_EE_ )            ;  // 9
    CutVars1_.push_back( trackIsoUser1_EE_ )   ;  // 10
    CutVars1_.push_back( ecalIsoUser1_EE_ )    ;  // 11
    CutVars1_.push_back( hcalIsoUser1_EE_ )    ;  // 12

    InvVars1_.push_back( trackIso1_EB_inv )    ;  // 0
    InvVars1_.push_back( ecalIso1_EB_inv )     ;  // 1
    InvVars1_.push_back( hcalIso1_EB_inv )     ;  // 2
    InvVars1_.push_back( sihih1_EB_inv )       ;  // 3
    InvVars1_.push_back( dphi1_EB_inv )        ;  // 4
    InvVars1_.push_back( deta1_EB_inv )        ;  // 5
    InvVars1_.push_back( hoe1_EB_inv )         ;  // 6
    InvVars1_.push_back( cIso1_EB_inv )        ;  // 7
    InvVars1_.push_back( tip_bspot1_EB_inv )   ;  // 8
    InvVars1_.push_back( eop1_EB_inv )         ;  // 9
    InvVars1_.push_back( trackIsoUser1_EB_inv );  // 10
    InvVars1_.push_back( ecalIsoUser1_EB_inv ) ;  // 11
    InvVars1_.push_back( hcalIsoUser1_EB_inv ) ;  // 12

    InvVars1_.push_back( trackIso1_EE_inv )    ;  // 0
    InvVars1_.push_back( ecalIso1_EE_inv )     ;  // 1
    InvVars1_.push_back( hcalIso1_EE_inv )     ;  // 2
    InvVars1_.push_back( sihih1_EE_inv )       ;  // 3
    InvVars1_.push_back( dphi1_EE_inv )        ;  // 4
    InvVars1_.push_back( deta1_EE_inv )        ;  // 5
    InvVars1_.push_back( hoe1_EE_inv )         ;  // 6
    InvVars1_.push_back( cIso1_EE_inv )        ;  // 7
    InvVars1_.push_back( tip_bspot1_EE_inv )   ;  // 8
    InvVars1_.push_back( eop1_EE_inv )         ;  // 9
    InvVars1_.push_back( trackIsoUser1_EE_inv );  // 10
    InvVars1_.push_back( ecalIsoUser1_EE_inv ) ;  // 11
    InvVars1_.push_back( hcalIsoUser1_EE_inv ) ;  // 12


    //  2nd Leg variables
    CutVars2_.push_back( trackIso2_EB_ )       ;  // 0
    CutVars2_.push_back( ecalIso2_EB_ )        ;  // 1
    CutVars2_.push_back( hcalIso2_EB_ )        ;  // 2
    CutVars2_.push_back( sihih2_EB_ )          ;  // 3
    CutVars2_.push_back( dphi2_EB_ )           ;  // 4
    CutVars2_.push_back( deta2_EB_ )           ;  // 5
    CutVars2_.push_back( hoe2_EB_ )            ;  // 6
    CutVars2_.push_back( cIso2_EB_ )           ;  // 7
    CutVars2_.push_back( tip_bspot2_EB_ )      ;  // 8
    CutVars2_.push_back( eop2_EB_ )            ;  // 9
    CutVars2_.push_back( trackIsoUser2_EB_ )   ;  // 10
    CutVars2_.push_back( ecalIsoUser2_EB_ )    ;  // 11
    CutVars2_.push_back( hcalIsoUser2_EB_ )    ;  // 12

    CutVars2_.push_back( trackIso2_EE_ )       ;  // 0
    CutVars2_.push_back( ecalIso2_EE_ )        ;  // 1
    CutVars2_.push_back( hcalIso2_EE_ )        ;  // 2
    CutVars2_.push_back( sihih2_EE_ )          ;  // 3
    CutVars2_.push_back( dphi2_EE_ )           ;  // 4
    CutVars2_.push_back( deta2_EE_ )           ;  // 5
    CutVars2_.push_back( hoe2_EE_ )            ;  // 6
    CutVars2_.push_back( cIso2_EE_ )           ;  // 7
    CutVars2_.push_back( tip_bspot2_EE_ )      ;  // 8
    CutVars2_.push_back( eop2_EE_ )            ;  // 9
    CutVars2_.push_back( trackIsoUser2_EE_ )   ;  // 10
    CutVars2_.push_back( ecalIsoUser2_EE_ )    ;  // 11
    CutVars2_.push_back( hcalIsoUser2_EE_ )    ;  // 12

    InvVars2_.push_back( trackIso2_EB_inv )    ;  // 0
    InvVars2_.push_back( ecalIso2_EB_inv )     ;  // 1
    InvVars2_.push_back( hcalIso2_EB_inv )     ;  // 2
    InvVars2_.push_back( sihih2_EB_inv )       ;  // 3
    InvVars2_.push_back( dphi2_EB_inv )        ;  // 4
    InvVars2_.push_back( deta2_EB_inv )        ;  // 5
    InvVars2_.push_back( hoe2_EB_inv )         ;  // 6
    InvVars2_.push_back( cIso2_EB_inv )        ;  // 7
    InvVars2_.push_back( tip_bspot2_EB_inv )   ;  // 8
    InvVars2_.push_back( eop2_EB_inv )         ;  // 9
    InvVars2_.push_back( trackIsoUser2_EB_inv );  // 10
    InvVars2_.push_back( ecalIsoUser2_EB_inv ) ;  // 11
    InvVars2_.push_back( hcalIsoUser2_EB_inv ) ;  // 12

    InvVars2_.push_back( trackIso2_EE_inv )    ;  // 0
    InvVars2_.push_back( ecalIso2_EE_inv )     ;  // 1
    InvVars2_.push_back( hcalIso2_EE_inv )     ;  // 2
    InvVars2_.push_back( sihih2_EE_inv )       ;  // 3
    InvVars2_.push_back( dphi2_EE_inv )        ;  // 4
    InvVars2_.push_back( deta2_EE_inv )        ;  // 5
    InvVars2_.push_back( hoe2_EE_inv )         ;  // 6
    InvVars2_.push_back( cIso2_EE_inv )        ;  // 7
    InvVars2_.push_back( tip_bspot2_EE_inv )   ;  // 8
    InvVars2_.push_back( eop2_EE_inv )         ;  // 9
    InvVars2_.push_back( trackIsoUser2_EE_inv );  // 10
    InvVars2_.push_back( ecalIsoUser2_EE_inv ) ;  // 11
    InvVars2_.push_back( hcalIsoUser2_EE_inv ) ;  // 12

    // ________________________________________________________________________
    //
    // The VBTF Root Tuples ---------------------------------------------------
    // ________________________________________________________________________
    //
    ZEE_VBTFselectionFile_ = new TFile(TString(ZEE_VBTFselectionFileName_) , "RECREATE");

    vbtfSele_tree = new TTree("vbtfSele_tree", "Tree to store the Z Candidates that pass the VBTF selection");

    vbtfSele_tree->Branch("runNumber", &runNumber, "runNumber/I");
    vbtfSele_tree->Branch("eventNumber", &eventNumber, "eventNumber/L");
    vbtfSele_tree->Branch("lumiSection", &lumiSection, "lumiSection/I");

    //  for ele 1
    vbtfSele_tree->Branch("ele1_sc_gsf_et", &ele1_sc_gsf_et,"ele1_sc_gsf_et/F");
    vbtfSele_tree->Branch("ele1_sc_energy", &ele1_sc_energy,"ele1_sc_energy/F");
    vbtfSele_tree->Branch("ele1_sc_eta", &ele1_sc_eta,"ele1_sc_eta/F");
    vbtfSele_tree->Branch("ele1_sc_phi", &ele1_sc_phi,"ele1_sc_phi/F");
    vbtfSele_tree->Branch("ele1_cand_et", &ele1_cand_et, "ele1_cand_et/F");
    vbtfSele_tree->Branch("ele1_cand_eta", &ele1_cand_eta,"ele1_cand_eta/F");
    vbtfSele_tree->Branch("ele1_cand_phi",&ele1_cand_phi,"ele1_cand_phi/F");
    vbtfSele_tree->Branch("ele1_iso_track",&ele1_iso_track,"ele1_iso_track/F");
    vbtfSele_tree->Branch("ele1_iso_ecal",&ele1_iso_ecal,"ele1_iso_ecal/F");
    vbtfSele_tree->Branch("ele1_iso_hcal",&ele1_iso_hcal,"ele1_iso_hcal/F");
    vbtfSele_tree->Branch("ele1_id_sihih",&ele1_id_sihih,"ele1_id_sihih/F");
    vbtfSele_tree->Branch("ele1_id_deta",&ele1_id_deta,"ele1_id_deta/F");
    vbtfSele_tree->Branch("ele1_id_dphi",&ele1_id_dphi,"ele1_id_dphi/F");
    vbtfSele_tree->Branch("ele1_id_hoe",&ele1_id_hoe,"ele1_id_hoe/F");
    vbtfSele_tree->Branch("ele1_cr_mhitsinner",&ele1_cr_mhitsinner,"ele1_cr_mhitsinner/I");
    vbtfSele_tree->Branch("ele1_cr_dcot",&ele1_cr_dcot,"ele1_cr_dcot/F");
    vbtfSele_tree->Branch("ele1_cr_dist",&ele1_cr_dist,"ele1_cr_dist/F");
    vbtfSele_tree->Branch("ele1_vx",&ele1_vx,"ele1_vx/F");
    vbtfSele_tree->Branch("ele1_vy",&ele1_vy,"ele1_vy/F");
    vbtfSele_tree->Branch("ele1_vz",&ele1_vz,"ele1_vz/F");
    vbtfSele_tree->Branch("ele1_gsfCharge",&ele1_gsfCharge,"ele1_gsfCharge/I");
    vbtfSele_tree->Branch("ele1_ctfCharge",&ele1_ctfCharge,"ele1_ctfCharge/I");
    vbtfSele_tree->Branch("ele1_scPixCharge",&ele1_scPixCharge,"ele1_scPixCharge/I");
    vbtfSele_tree->Branch("ele1_eop",&ele1_eop,"ele1_eop/F");
    vbtfSele_tree->Branch("ele1_tip_bs",&ele1_tip_bs,"ele1_tip_bs/F");
    vbtfSele_tree->Branch("ele1_tip_pv",&ele1_tip_pv,"ele1_tip_pv/F");

    //  for ele 2
    vbtfSele_tree->Branch("ele2_sc_gsf_et", &ele2_sc_gsf_et,"ele2_sc_gsf_et/F");
    vbtfSele_tree->Branch("ele2_sc_energy", &ele2_sc_energy,"ele2_sc_energy/F");
    vbtfSele_tree->Branch("ele2_sc_eta", &ele2_sc_eta,"ele2_sc_eta/F");
    vbtfSele_tree->Branch("ele2_sc_phi", &ele2_sc_phi,"ele2_sc_phi/F");
    vbtfSele_tree->Branch("ele2_cand_et", &ele2_cand_et, "ele2_cand_et/F");
    vbtfSele_tree->Branch("ele2_cand_eta", &ele2_cand_eta,"ele2_cand_eta/F");
    vbtfSele_tree->Branch("ele2_cand_phi",&ele2_cand_phi,"ele2_cand_phi/F");
    vbtfSele_tree->Branch("ele2_iso_track",&ele2_iso_track,"ele2_iso_track/F");
    vbtfSele_tree->Branch("ele2_iso_ecal",&ele2_iso_ecal,"ele2_iso_ecal/F");
    vbtfSele_tree->Branch("ele2_iso_hcal",&ele2_iso_hcal,"ele2_iso_hcal/F");
    vbtfSele_tree->Branch("ele2_id_sihih",&ele2_id_sihih,"ele2_id_sihih/F");
    vbtfSele_tree->Branch("ele2_id_deta",&ele2_id_deta,"ele2_id_deta/F");
    vbtfSele_tree->Branch("ele2_id_dphi",&ele2_id_dphi,"ele2_id_dphi/F");
    vbtfSele_tree->Branch("ele2_id_hoe",&ele2_id_hoe,"ele2_id_hoe/F");
    vbtfSele_tree->Branch("ele2_cr_mhitsinner",&ele2_cr_mhitsinner,"ele2_cr_mhitsinner/I");
    vbtfSele_tree->Branch("ele2_cr_dcot",&ele2_cr_dcot,"ele2_cr_dcot/F");
    vbtfSele_tree->Branch("ele2_cr_dist",&ele2_cr_dist,"ele2_cr_dist/F");
    vbtfSele_tree->Branch("ele2_vx",&ele2_vx,"ele2_vx/F");
    vbtfSele_tree->Branch("ele2_vy",&ele2_vy,"ele2_vy/F");
    vbtfSele_tree->Branch("ele2_vz",&ele2_vz,"ele2_vz/F");
    vbtfSele_tree->Branch("ele2_gsfCharge",&ele2_gsfCharge,"ele2_gsfCharge/I");
    vbtfSele_tree->Branch("ele2_ctfCharge",&ele2_ctfCharge,"ele2_ctfCharge/I");
    vbtfSele_tree->Branch("ele2_scPixCharge",&ele2_scPixCharge,"ele2_scPixCharge/I");
    vbtfSele_tree->Branch("ele2_eop",&ele2_eop,"ele2_eop/F");
    vbtfSele_tree->Branch("ele2_tip_bs",&ele2_tip_bs,"ele2_tip_bs/F");
    vbtfSele_tree->Branch("ele2_tip_pv",&ele2_tip_pv,"ele2_tip_pv/F");
    //
    vbtfSele_tree->Branch("pv_x1",&pv_x1,"pv_x1/F");
    vbtfSele_tree->Branch("pv_y1",&pv_y1,"pv_y1/F");
    vbtfSele_tree->Branch("pv_z1",&pv_z1,"pv_z1/F");
    //
    vbtfSele_tree->Branch("pv_x2",&pv_x2,"pv_x2/F");
    vbtfSele_tree->Branch("pv_y2",&pv_y2,"pv_y2/F");
    vbtfSele_tree->Branch("pv_z2",&pv_z2,"pv_z2/F");
    //
    vbtfSele_tree->Branch("event_caloMET",&event_caloMET,"event_caloMET/F");
    vbtfSele_tree->Branch("event_pfMET",&event_pfMET,"event_pfMET/F");
    vbtfSele_tree->Branch("event_tcMET",&event_tcMET,"event_tcMET/F");
    vbtfSele_tree->Branch("event_caloMET_phi",&event_caloMET_phi,"event_caloMET_phi/F");
    vbtfSele_tree->Branch("event_pfMET_phi",&event_pfMET_phi,"event_pfMET_phi/F");
    vbtfSele_tree->Branch("event_tcMET_phi",&event_tcMET_phi,"event_tcMET_phi/F");

    vbtfSele_tree->Branch("event_Mee",&event_Mee,"event_Mee/F");

    //
    // the extra jet variables:
    if ( includeJetInformationInNtuples_ ) {

        vbtfSele_tree->Branch("calojet_et",calojet_et,"calojet_et[5]/F");
        vbtfSele_tree->Branch("calojet_eta",calojet_eta,"calojet_eta[5]/F");
        vbtfSele_tree->Branch("calojet_phi",calojet_phi,"calojet_phi[5]/F");
        vbtfSele_tree->Branch("pfjet_et",pfjet_et,"pfjet_et[5]/F");
        vbtfSele_tree->Branch("pfjet_eta",pfjet_eta,"pfjet_eta[5]/F");
        vbtfSele_tree->Branch("pfjet_phi",pfjet_phi,"pfjet_phi[5]/F");

    }

    vbtfSele_tree->Branch("event_datasetTag",&event_datasetTag,"event_dataSetTag/I");

    // everything after preselection
    ZEE_VBTFpreseleFile_ = new TFile(TString(ZEE_VBTFpreseleFileName_) , "RECREATE");

    vbtfPresele_tree = new TTree("vbtfPresele_tree", "Tree to store the Z Candidates that pass the VBTF preselection");

    vbtfPresele_tree->Branch("runNumber", &runNumber, "runNumber/I");
    vbtfPresele_tree->Branch("eventNumber", &eventNumber, "eventNumber/L");
    vbtfPresele_tree->Branch("lumiSection", &lumiSection, "lumiSection/I");

    //  for ele 1
    vbtfPresele_tree->Branch("ele1_sc_gsf_et", &ele1_sc_gsf_et,"ele1_sc_gsf_et/F");
    vbtfPresele_tree->Branch("ele1_sc_energy", &ele1_sc_energy,"ele1_sc_energy/F");
    vbtfPresele_tree->Branch("ele1_sc_eta", &ele1_sc_eta,"ele1_sc_eta/F");
    vbtfPresele_tree->Branch("ele1_sc_phi", &ele1_sc_phi,"ele1_sc_phi/F");
    vbtfPresele_tree->Branch("ele1_cand_et", &ele1_cand_et, "ele1_cand_et/F");
    vbtfPresele_tree->Branch("ele1_cand_eta", &ele1_cand_eta,"ele1_cand_eta/F");
    vbtfPresele_tree->Branch("ele1_cand_phi",&ele1_cand_phi,"ele1_cand_phi/F");
    vbtfPresele_tree->Branch("ele1_iso_track",&ele1_iso_track,"ele1_iso_track/F");
    vbtfPresele_tree->Branch("ele1_iso_ecal",&ele1_iso_ecal,"ele1_iso_ecal/F");
    vbtfPresele_tree->Branch("ele1_iso_hcal",&ele1_iso_hcal,"ele1_iso_hcal/F");
    vbtfPresele_tree->Branch("ele1_id_sihih",&ele1_id_sihih,"ele1_id_sihih/F");
    vbtfPresele_tree->Branch("ele1_id_deta",&ele1_id_deta,"ele1_id_deta/F");
    vbtfPresele_tree->Branch("ele1_id_dphi",&ele1_id_dphi,"ele1_id_dphi/F");
    vbtfPresele_tree->Branch("ele1_id_hoe",&ele1_id_hoe,"ele1_id_hoe/F");
    vbtfPresele_tree->Branch("ele1_cr_mhitsinner",&ele1_cr_mhitsinner,"ele1_cr_mhitsinner/I");
    vbtfPresele_tree->Branch("ele1_cr_dcot",&ele1_cr_dcot,"ele1_cr_dcot/F");
    vbtfPresele_tree->Branch("ele1_cr_dist",&ele1_cr_dist,"ele1_cr_dist/F");
    vbtfPresele_tree->Branch("ele1_vx",&ele1_vx,"ele1_vx/F");
    vbtfPresele_tree->Branch("ele1_vy",&ele1_vy,"ele1_vy/F");
    vbtfPresele_tree->Branch("ele1_vz",&ele1_vz,"ele1_vz/F");
    vbtfPresele_tree->Branch("ele1_gsfCharge",&ele1_gsfCharge,"ele1_gsfCharge/I");
    vbtfPresele_tree->Branch("ele1_ctfCharge",&ele1_ctfCharge,"ele1_ctfCharge/I");
    vbtfPresele_tree->Branch("ele1_scPixCharge",&ele1_scPixCharge,"ele1_scPixCharge/I");
    vbtfPresele_tree->Branch("ele1_eop",&ele1_eop,"ele1_eop/F");
    vbtfPresele_tree->Branch("ele1_tip_bs",&ele1_tip_bs,"ele1_tip_bs/F");
    vbtfPresele_tree->Branch("ele1_tip_pv",&ele1_tip_pv,"ele1_tip_pv/F");

    //  for ele 2
    vbtfPresele_tree->Branch("ele2_sc_gsf_et", &ele2_sc_gsf_et,"ele2_sc_gsf_et/F");
    vbtfPresele_tree->Branch("ele2_sc_energy", &ele2_sc_energy,"ele2_sc_energy/F");
    vbtfPresele_tree->Branch("ele2_sc_eta", &ele2_sc_eta,"ele2_sc_eta/F");
    vbtfPresele_tree->Branch("ele2_sc_phi", &ele2_sc_phi,"ele2_sc_phi/F");
    vbtfPresele_tree->Branch("ele2_cand_et", &ele2_cand_et, "ele2_cand_et/F");
    vbtfPresele_tree->Branch("ele2_cand_eta", &ele2_cand_eta,"ele2_cand_eta/F");
    vbtfPresele_tree->Branch("ele2_cand_phi",&ele2_cand_phi,"ele2_cand_phi/F");
    vbtfPresele_tree->Branch("ele2_iso_track",&ele2_iso_track,"ele2_iso_track/F");
    vbtfPresele_tree->Branch("ele2_iso_ecal",&ele2_iso_ecal,"ele2_iso_ecal/F");
    vbtfPresele_tree->Branch("ele2_iso_hcal",&ele2_iso_hcal,"ele2_iso_hcal/F");
    vbtfPresele_tree->Branch("ele2_id_sihih",&ele2_id_sihih,"ele2_id_sihih/F");
    vbtfPresele_tree->Branch("ele2_id_deta",&ele2_id_deta,"ele2_id_deta/F");
    vbtfPresele_tree->Branch("ele2_id_dphi",&ele2_id_dphi,"ele2_id_dphi/F");
    vbtfPresele_tree->Branch("ele2_id_hoe",&ele2_id_hoe,"ele2_id_hoe/F");
    vbtfPresele_tree->Branch("ele2_cr_mhitsinner",&ele2_cr_mhitsinner,"ele2_cr_mhitsinner/I");
    vbtfPresele_tree->Branch("ele2_cr_dcot",&ele2_cr_dcot,"ele2_cr_dcot/F");
    vbtfPresele_tree->Branch("ele2_cr_dist",&ele2_cr_dist,"ele2_cr_dist/F");
    vbtfPresele_tree->Branch("ele2_vx",&ele2_vx,"ele2_vx/F");
    vbtfPresele_tree->Branch("ele2_vy",&ele2_vy,"ele2_vy/F");
    vbtfPresele_tree->Branch("ele2_vz",&ele2_vz,"ele2_vz/F");
    vbtfPresele_tree->Branch("ele2_gsfCharge",&ele2_gsfCharge,"ele2_gsfCharge/I");
    vbtfPresele_tree->Branch("ele2_ctfCharge",&ele2_ctfCharge,"ele2_ctfCharge/I");
    vbtfPresele_tree->Branch("ele2_scPixCharge",&ele2_scPixCharge,"ele2_scPixCharge/I");
    vbtfPresele_tree->Branch("ele2_eop",&ele2_eop,"ele2_eop/F");
    vbtfPresele_tree->Branch("ele2_tip_bs",&ele2_tip_bs,"ele2_tip_bs/F");
    vbtfPresele_tree->Branch("ele2_tip_pv",&ele2_tip_pv,"ele2_tip_pv/F");

    vbtfPresele_tree->Branch("pv_x1",&pv_x1,"pv_x1/F");
    vbtfPresele_tree->Branch("pv_y1",&pv_y1,"pv_y1/F");
    vbtfPresele_tree->Branch("pv_z1",&pv_z1,"pv_z1/F");

    vbtfPresele_tree->Branch("pv_x2",&pv_x2,"pv_x2/F");
    vbtfPresele_tree->Branch("pv_y2",&pv_y2,"pv_y2/F");
    vbtfPresele_tree->Branch("pv_z2",&pv_z2,"pv_z2/F");

    vbtfPresele_tree->Branch("event_caloMET",&event_caloMET,"event_caloMET/F");
    vbtfPresele_tree->Branch("event_pfMET",&event_pfMET,"event_pfMET/F");
    vbtfPresele_tree->Branch("event_tcMET",&event_tcMET,"event_tcMET/F");
    vbtfPresele_tree->Branch("event_caloMET_phi",&event_caloMET_phi,"event_caloMET_phi/F");
    vbtfPresele_tree->Branch("event_pfMET_phi",&event_pfMET_phi,"event_pfMET_phi/F");
    vbtfPresele_tree->Branch("event_tcMET_phi",&event_tcMET_phi,"event_tcMET_phi/F");

    vbtfPresele_tree->Branch("event_Mee",&event_Mee,"event_Mee/F");

    //
    // the extra jet variables:
    if ( includeJetInformationInNtuples_) {

        vbtfPresele_tree->Branch("calojet_et",calojet_et,"calojet_et[5]/F");
        vbtfPresele_tree->Branch("calojet_eta",calojet_eta,"calojet_eta[5]/F");
        vbtfPresele_tree->Branch("calojet_phi",calojet_phi,"calojet_phi[5]/F");
        vbtfPresele_tree->Branch("pfjet_et",pfjet_et,"pfjet_et[5]/F");
        vbtfPresele_tree->Branch("pfjet_eta",pfjet_eta,"pfjet_eta[5]/F");
        vbtfPresele_tree->Branch("pfjet_phi",pfjet_phi,"pfjet_phi[5]/F");

    }

    vbtfPresele_tree->Branch("event_datasetTag",&event_datasetTag,"event_dataSetTag/I");


    // _________________________________________________________________________
    //



}

// ------------ method called once each job just after ending the event loop  -
void ZeePlots::endJob() {
    TFile * newfile  =  new TFile(TString(outputFile_),"RECREATE");
    //
    // for consistency all the plots are in the root file
    // even though they may be empty (in the case when
    // usePrecalcID_ ==  true inverted and N-1 are empty)

    h_mee->Write();
    h_mee_EBEB->Write();
    h_mee_EBEE->Write();
    h_mee_EEEE->Write();
    h_Zcand_PT->Write();
    h_Zcand_Y->Write();

    h_e_PT->Write();
    h_e_ETA->Write();
    h_e_PHI->Write();

    h_EB_trkiso->Write();
    h_EB_ecaliso->Write();
    h_EB_hcaliso->Write();
    h_EB_sIetaIeta->Write();
    h_EB_dphi->Write();
    h_EB_deta->Write();
    h_EB_HoE->Write();

    h_EE_trkiso->Write();
    h_EE_ecaliso->Write();
    h_EE_hcaliso->Write();
    h_EE_sIetaIeta->Write();
    h_EE_dphi->Write();
    h_EE_deta->Write();
    h_EE_HoE->Write();

    //
    newfile->Close();

    // write the VBTF trees
    //
    ZEE_VBTFpreseleFile_->Write();
    ZEE_VBTFpreseleFile_->Close();

    ZEE_VBTFselectionFile_->Write();
    ZEE_VBTFselectionFile_->Close();

}


//define this as a plug-in
DEFINE_FWK_MODULE(ZeePlots);
