{
//look at higgs decays
Setup::decay_particle=23;
Setup::mass_power=1;
Setup::mass_scale_on=true;

// Setup histograms
int n_bins=60;
double default_min_bin=0.0;
double default_max_bin=1.1;
Setup::SetHistogramDefaults(n_bins,default_min_bin,default_max_bin);

// Setup User Histograms
//Setup::UserTreeAnalysis = "PiPiUserTreeAnalysis";

// Description
Setup::gen1_desc_1="Pythia + Tauola Interface Test";
Setup::gen1_desc_2=" $e^+ e^- \\rightarrow Z^0 \\rightarrow \\pi^+ \\pi^- \\nu_{\\tau} \\bar{\\nu_{\\tau}} $";
 Setup::gen1_desc_3="Center of Mass Energy = 500GeV. New";


if (Setup::stage==0)
    printf("Setup loaded from SETUP.C, ANALYSIS stage.\n");
else 
    printf("Setup loaded from SETUP.C, GENERATION stage %i.\n",Setup::stage);

Setup::SuppressDecay(111); // suppress pi0 decays

};

