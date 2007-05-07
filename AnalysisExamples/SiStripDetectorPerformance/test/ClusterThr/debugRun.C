int i,j;
void debugRun(){
  gSystem->Load("ClusterThr_C.so");

  char outfile[128];
  sprintf(outfile,"/tmp/giordano/ClusterThr/prova.root");

  cout << "OutputFiles " << outfile << endl;
 
  ClusterThr A;                                                      
      
  /* Set input file(s) separated by space*/
  //  A.setInputFiles("SingleMuon_Clusters","/data/local1/giordano/ClusterTh_Data");
  A.setInputFiles("6925","/tmp/giordano");
  //A.setInputFile("/castor/cern.ch/cms/store/TAC/RECO/2007/3/2/TAC-TOB-RecoPass0b-run2048/0000/06D72868-78C8-DB11-8AF9-00304876A0DB.root");
  //A.setInputFiles("rfio:/castor/cern.ch/cms/store/TAC/RECO/2007/3/31/TAC-TIBTOB-RecoPass0r-run6923/0000/123BA35D-95DF-DB11-81F2-001731AF67E1.root");

  /* set Output file */
  A.setOutputFile(outfile);                                        

  /* Set maximum number of events to be analyzed*/
  A.setMaxEvent(5);

  /* Set verbosity in a scale from 0 to 4 */
  A.setVerbosity(1);

  A.setBadStripsFile("badStrips.bin");
    
  /* Set definition of Noise used in S/N cut*/
  A.setNoiseMode(1);
    
  /* Set SiStripClusterInfo branch name, this could change from process to process */
  A.setBranchName("SiStripClusterInfoedmDetSetVector_siStripClusterInfoProducer__NewNoise.obj._sets");

  /* Set module to be skipped*/
  //    A.setSkippedModules("369199109 419561989 419627960");
  //    A.setSkippedModules("369199365 369199366 369199370 369199369 369199374 369199373 369329156 369329160 369329164 369264138");

  /* Set SubDets and Layer to be analyzed*/

  //A.setSubDets("TOB");
  //A.setLayers("1 2 3");

  /* Set cuts to separate background from signal*/
  A.setStoNThr(13,15);

  /* Set grid to scan the parameter space*/
  A.setThC(5,15,2); //ThC_min,ThC_max,ThC_step
  A.setThS(3,9,1); //ThS_min,ThS_max,ThS_step
  A.setThN(2,4,.25); //ThN_min,ThN_max,ThN_step

  //Single set of values
  //A.setThC(5,5.5,1); //ThC_min,ThC_max,ThC_step
  //A.setThS(3,3.5,1); //ThS_min,ThS_max,ThS_step
  //A.setThN(2,2.5,1); //ThN_min,ThN_max,ThN_step

  /* do loop on all events, 
     do clusterization for the different cuts,
     fill histograms for each set of cuts
  */
  A.Process();     

  /* do analysis on previously generated histograms
     can be done also later respect to Process(), in another subproc
  */
  //A.Analysis();
    
  cout << "j " << j << endl;
}
