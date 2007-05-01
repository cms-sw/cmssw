{
gSystem->Load("ClusterThr_C.so");

char outfile[128];
sprintf(outfile,"/tmp/giordano/SingleMuon_ClusterTh_out.root");
cout << outfile << endl;
 
ClusterThr A;                                                      

A.setInputFiles("SingleMuon_ClusterTh","/tmp/giordano/");

A.setOutputFile(outfile);                                        
    
/* Set maximum number of events to be analyzed*/
//A.setMaxEvent(100);

/* Set verbosity in a scale from 0 to 4 */
A.setVerbosity(0);
    
/* Set definition of Noise used in S/N cut*/
A.setNoiseMode(1);
  
/* Set SiStripClusterInfo branch name, this could change from process to process */
A.setBranchName("SiStripClusterInfoedmDetSetVector_siStripClusterInfoProducer__NewNoise.obj._sets");

/* Set SubDets and Layer to be analyzed*/

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
A.Analysis();


/*
A.setVerbosity(0);
A.setBranchName("SiStripClusterInfoedmDetSetVector_siStripClusterInfoProducer__TrackerValidationNoiseBug.obj._sets");
A.setInputFile("/tmp/giordano/NoiseBug/Muon_NoiseFix_new.root");


A.setOutputFile(outfile);                                        
    
A.setStoNThr(13,15);
A.setThC(5,15,2); //ThC_min,ThC_max,ThC_step
A.setThS(3,9,1); //ThS_min,ThS_max,ThS_step
A.setThN(2,4,.25); //ThN_min,ThN_max,ThN_step
//A.setThC(5,5.5,1); //ThC_min,ThC_max,ThC_step
//A.setThS(3,3.5,1); //ThS_min,ThS_max,ThS_step
//A.setThN(2,2.5,1); //ThN_min,ThN_max,ThN_step
A.Process();     
A.Analysis();
*/

}

