void readHistos(int nc=4) {
    std::cout << "\n  vvvvv--------  Lines for the Datacard  -----------------------------" << std::endl;
    std::cout << "observation " ;
    for (int i = 0; i < nc; ++i) { 
        std::cout << " " << ((TH1*)gFile->Get(TString::Format("datmc_cat%d",i)))->Integral(); 
    }
    std::cout << std::endl;

    std::cout << "rate ";
    for (int i = 0; i < nc; ++i) { 
        std::cout << " " << ((TH1*)gFile->Get(TString::Format("sig_cat%d",i)))->Integral()  
                  << " " << ((TH1*)gFile->Get(TString::Format("bkg_cat%d",i)))->Integral(); 
    }
    std::cout << std::endl;
    std::cout << "  ---------------------------------------  End of Lines  --------^^^^^" << std::endl;

    const int nsyst = 5;
    const char *systs[nsyst] = { "massresol", "scaleeta", "scaler9", "migr9", "migpt" };
    for (int s = 0; s < nsyst; ++s) {
        printf("  ===== %s =====\n", systs[s]);
        for (int i = 0; i < nc; ++i) { 
            TH1F *h0  = (TH1*) gFile->Get(TString::Format("sig_cat%d", i));
            TH1F *hup = (TH1*) gFile->Get(TString::Format("sig_%sUp_cat%d", systs[s],i));
            TH1F *hdn = (TH1*) gFile->Get(TString::Format("sig_%sDown_cat%d", systs[s],i));
            double m0 = h0->GetMean(), rms0 = h0->GetRMS(), norm0 = h0->Integral();
            double mup = hup->GetMean(), rmsup = hup->GetRMS(), normup = hup->Integral();
            double mdn = hdn->GetMean(), rmsdn = hdn->GetRMS(), normdn = hdn->Integral();
            printf("cat %d: norm %5.3f/%5.3f,  mean %+4.2f/%+4.2f,  rms %4.2f/%4.2f\n", i, 
                    normdn/norm0, normup/norm0,
                    mdn-m0, mup-m0,
                    rmsdn/rms0, rmsup/rms0);
        }
        printf("\n");
    }
}
