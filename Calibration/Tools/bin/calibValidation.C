{
        gSystem->Load("libFWCoreFWLite.so"); 
        AutoLibraryLoader::enable();
        gSystem->Load("libDataFormatsFWLite.so");
        printf("libDataFormatsFWLite.so loaded\n");

        gSystem->Load("libCalibrationTools.so");
        printf("libCalibrationTools.so loaded\n");
        DSIsBarrel isBarrel;
        DSIsEndcap isEndcap;
        DSIsEndcapPlus isEndcapPlus;
        DSIsEndcapMinus isEndcapMinus;
        DSAll all;
        IC ic;
        ////IC::readSimpleTextFile("ic_test.dat", ic);
        //IC::readTextFile("interCalibConstants.combinedPi0Eta.run178003to180252.EcalBarrel_corrected.txt", ic);
        //cout << "EB average " << IC::average(ic, isBarrel) << std::endl;
        //cout << "EE average " << IC::average(ic, isEndcap) << std::endl;
        //cout << "All average " << IC::average(ic, all) << std::endl;
        //IC::readCmscondXMLFile("MC_production/EcalIntercalibConstantsMC_2010_V3_Bon_mc.xml", ic);
        IC a, b, res;
        //IC::readXMLFile("MC_production/MC_startup_2011_V3withBiXtalEBCorr.xml", a);
        //IC::readXMLFile("MC_production/EcalIntercalibConstants_V20120109_Electrons_etaScale_withErrors.dat", a);
        //IC::readTextFile("EcalIntercalibConstants_V20120109_Electrons_etaScale_withErrors_smeared.dat", b);
        IC::readXMLFile("EcalIntercalibConstants_IDEAL.xml", a);
        //IC::dump("EcalIntercalibConstants_IDEAL.dat", a);
        IC::readXMLFile("/afs/cern.ch/user/r/ric/public/4federico/MC_ideal_2011_V3withBiXtalEBCorr.xml", b);
        //IC::dump("ric.dat", b);
        IC::multiply(a, -1, a);
        IC::add(a, b, res);
        TFile * fout = new TFile("fout_valid.root", "RECREATE");
        TProfile * p_eta = new TProfile("p_eta", "p_eta", 171+120, -145.5, 145.5);
        IC::profileEta(res, p_eta, all);
        TH1F * hEB = new TH1F("hEB", "hEB", 150, -0.5, 0.5);
        IC::constantDistribution(res, hEB, isBarrel);
        TH1F * hEE = new TH1F("hEE", "hEE", 150, -0.5, 0.5);
        IC::constantDistribution(res, hEE, isEndcap);

        TH2F * h2EE_p = new TH2F("h2EE_p", "h2EE_p", 101, -0.5, 100.5, 101, -0.5, 100.5);
        TH2F * h2EE_m = new TH2F("h2EE_m", "h2EE_m", 101, -0.5, 100.5, 101, -0.5, 100.5);
        IC::constantMap(res, h2EE_p, isEndcapPlus);
        IC::constantMap(res, h2EE_m, isEndcapMinus);

        TH2F * h2EB = new TH2F("h2EB", "h2EB", 361, -0.5, 360.5, 171, -85.5, 85.5);
        IC::constantMap(res, h2EB, isBarrel);


        fout->Write();
        //fout->Close();
}
