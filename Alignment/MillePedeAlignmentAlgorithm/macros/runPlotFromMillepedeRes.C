{
  gROOT->ProcessLine(".L PlotFromMillepedeRes.C+");
  gStyle->SetPalette(1);
  gStyle->SetCanvasColor(kWhite); 

  //enum {PXB,PXF,TIB,TID,TOB,TEC};
  //enum {PULLS, NHITS, PARS, PARSwithERRvsLABEL};

  TString strVars;

  // uncomment two lines you need (check below)
    // one starts with strVars
    // the other starts with PlotFromMillepedeRes
  // pass appropriate unzipped millepede.res file
    // as the first argument to PlotFromMillepedeRes()
  // don't change strVars
  // set strOutdir where you want to save plots
    // by default, it's here, where you run this script
  // they will be saved in .png and .pdf formats
  // there will be also empty root file
  // run with
  // root -l runPlotMillepedeRes.C

  TString strOutdir="./"; 

  //###########################################################

  // plots par+-err as function of (label%700000) 
  // quoted in millepede.res 
  // to be used to plot from millepede.res with high level alignables 
  // ran in inversion mode (has to have errors, see strVars defined above)

  // uncomment two lines below if this is what you want to plot

//  strVars="label/I:smth2/F:smth1/F:parVal/F:parErr/F:Nhits/I";
//  PlotFromMillepedeRes("mp1720_millepede.res", strOutdir, strVars, PARSwithERRvsLABEL);

  //###########################################################

  // plots distributions of number of derivatives
  // associated with given module (last column in millepede.res)
  // for certain modules it's equal to number of tracks passed 
  // through the module
  // and for the others it's 2(x)tracks
  // the script takes this number from "u" coordinate
  // it introduces cut label<700000 to reject high level structures 
  // if they were allowed for IOV 
  // (if they were not then they also will contribute to distributions)
  // to be used to plot from millepede.res with module level alignables 
  // assumes that you didn't run in inversion mode and millepede.res doesn't have errors
  // if it does, please, add appropriate variable to strVars

  // uncomment two lines below if this is what you want to plot

//  strVars="label/I:smth2/F:smth1/F:parVal/F:Nhits/I";
//  PlotFromMillepedeRes("mp1700_millepede.res", strOutdir, strVars, NHITS);

  //###########################################################

  // plots distributions of pulls 
  // defined as parVal/(1.41 x parErr)
  // 1.41 factor appears because for pull distributions 
  // we run alignment in 2 iterations on independent same-size
  // sets of data and the error of the 1st iteration is ignored
  // has cut label<700000 to reject large structures if they were allowed for IOV
  // needs unzipped millepede.res ran in inversion mode 

  // uncomment two lines below if this is what you want to plot

//  strVars="label/I:smth2/F:smth1/F:parVal/F:parErr/F:Nhits/I";
//  PlotFromMillepedeRes("mp1587_mpInversion_TPBandTPE_jobData_jobm_millepede.res", strOutdir, strVars, PULLS);

  //###########################################################

  // plots distributions of parVal separately for each subdetector
  // it saved 6 different canvases, 9 plots on each (u,v,w,alpha,beta,gamma,def1,def2,def3)
  // has cut label<700000 to reject large structures if they were allowed for IOV
  // needs unzipped millepede.res
  // assumes that you didn't run in inversion mode and millepede.res doesn't have errors
  // if it does, please, add appropriate variable to strVars

  // uncomment two lines below if this is what you want to plot

//  strVars="label/I:smth2/F:smth1/F:parVal/F:Nhits/I";
//  PlotFromMillepedeRes("mp1700_millepede.res", strOutdir, strVars, PARS);

}
