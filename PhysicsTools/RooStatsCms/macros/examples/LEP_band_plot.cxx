void LEP_band_plot(){

    gSystem->Load("libRooStatsCms.so");


    // The content of the plot
    // Fermilab-Pub-08-069-E, CDF Note 9290, D0 note 5645
    // PLEASE DO NOT CITE THIS PLOT FOR ANY OFFICIAL PURPOSE! IT IS JUST AN EXAMPLE!

    double x_vals[10]={110,120,130,140,150,160,170,180,190,200};

    double sb_vals[10]={-.416,-.25,-.166,-.25,-.583,-1.75,-1.416,-.75,-.25,-.083};

    double b_vals[10]={0.583,.466,.466,.583,.833,2.083,1.666,.916,.466,0.25};
    double b_rms[10]={1.333,1.083,1.,1.25,1.583,2.5,2.25,1.666,1.083,0.791};

    double exp_vals[10]={.833,-.666,-.333,.500,3.33,4.,3.41,1.33,1.34,0.166};

    // The plot object
    LEPBandPlot* plot=new LEPBandPlot("Tevatron Band Plot","Tevatron Band Plot",10,x_vals,sb_vals,b_vals,b_rms,exp_vals);
    plot->setXaxisTitle("m_{H} (GeV/c^{2})");
    plot->setTitle("Rsc Examples Plot (inspired by Tevatron HWG)");

    plot->draw();

    }