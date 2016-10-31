void runplot()
{
    gROOT->ProcessLine(" .L tdrstyle.C");
    gROOT->ProcessLine("setTDRStyle()");
    gROOT->ProcessLine(".L HIPplots.cc++");
    gROOT->ProcessLine(".L plotter.C++");

    //***********DEFINED BY USER******************

    char* path = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HIP/public/AlignmentCamp_2016RunB/CMSSW_8_0_7/src/Alignment/HIPAlignmentAlgorithm/";
    char* obj = "hp1704_2";

    //starting run number of alignment
    int iov = 272497;

    // choose plot type from : "cov","shift","chi2","param","hitmap"
    char* plotType[] = { "cov", "shift", "chi2", "param", "hitmap" };

    // plot all detectors together or individually
    bool MergeAllDet = 1;
    
    //the iteration to use for plotting fitted paramters
    int Niteration =9;

    //*******************************************

    int Nplots = sizeof(plotType) / sizeof(plotType[0]);

    //plotting all detectors together

    if (MergeAllDet == 1) {
        for (int i = 0; i < Nplots; i++) {
            char arg[500];
            sprintf(arg, "plotter(\"%s\",\"%s\",%d,\"%s\",%d)", path, obj, iov, plotType[i],Niteration);
            cout << arg << endl;
            gROOT->ProcessLine(arg);
        }
    }

    // plotting each detector separately, don't use this for hit-map.

    else {
        // det: 0=all,1=PXB, 2=PXF, 3=TIB, 4=TID, 5=TOB, 6=TEC
        for (int i = 0; i < Nplots; i++) {
            for (int det = 0; det < 6; det++) {
                char arg2[500];
                sprintf(arg2, "plotter(\"%s\",\"%s\",%d,\"%s\",%d,%d)", path, obj, iov, plotType[i],Niteration,det);
                gROOT->ProcessLine(arg2);
            }
        }
    }
}

