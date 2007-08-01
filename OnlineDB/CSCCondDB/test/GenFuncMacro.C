void directoryCheck(){
  //check that the top directory is accurate. simple book-keeping. 
  TString this_dir = gSystem->pwd(); //returns linux directory where macro was executed 
  //change to be appropriate to the current directory
  if ( this_dir != "/afs/cern.ch/user/a/aroe/scratch0/CMSSW_1_1_1/src/OnlineDB/CSCCondDB/test" ){
    std::cout << "you aint in images! you in:" << std::endl;
    std::cout << this_dir << std::endl;
  }
}//directoryCheck()

void makeDirectory(TString directory_name){
  Int_t bool_dir = gSystem->mkdir(directory_name);//0 if there is no directory of this name, -1 if there is 
 if (bool_dir == 0){
  gSystem->mkdir(directory_name); 
  std::cout << "creating directory called " << directory_name << std::endl; 
 }
 if (bool_dir != 0){
   std::cout << "directory called " << directory_name << " already exists" << std::endl;
 }	 
}


void PrintAsGif(TCanvas *CanvasPointer, TString canvName){  
  if( (gROOT->IsBatch() ) == 1 ){

    /*
    TString CanvName = canvName + ".gif"; 
    CanvasPointer->Print(CanvName);     
    */
        
    TString CanvName = canvName + ".eps"; 
    CanvasPointer->Print(CanvName);       
    
    TString pstoString = "pstopnm -ppm -xborder 0 -yborder 0 -xsize 1050 -ysize 625 -portrait "+ CanvName; 
    TString togifString = "ppmtogif " + CanvName+ "001.ppm > " + canvName + ".gif"; 
    TString rmString1 = "rm " + CanvName+ "001.ppm"; 
    TString rmString2 = "rm " + CanvName; 
    
    gSystem->Exec(pstoString); 
    gSystem->Exec(togifString); 
    gSystem->Exec(rmString1); 
    gSystem->Exec(rmString2);     
    
  } 
}


