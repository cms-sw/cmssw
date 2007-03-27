{
gSystem->Load("ClusterThr_C.so");

char outfile[128];
sprintf(outfile,"/tmp/giordano/NoiseBug/Muon_Val.root");
cout << outfile << endl;
 
ClusterThr A;                                                      
A.setVerbosity(0);
A.setBranchName("SiStripClusterInfoedmDetSetVector_siStripClusterInfoProducer__TrackerValidationNoiseBug.obj._sets");
A.setInputFile("/tmp/giordano/NoiseBug/Muon_NoiseFix_new.root");
//A.setDetMap("../Map_TIB_TID.txt");

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
}

