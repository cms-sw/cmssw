void correctLumiMap(char* inFile="lumiByLSMap_179828.icc", int nBX=10, int correctionMode=3){
  char str [80];
  float f;
  FILE * pFile;

  pFile = fopen (inFile,"r");
	if (pFile == NULL) {
		fprintf(stderr, "Can't open input file in.list!\n");
		return;
	}
	TString iFileName(inFile);
	int idx = iFileName.Index('.');
	TString oFileName(iFileName);
	oFileName.Resize(idx);
	oFileName+="_corr.icc";
  outFile = fopen (oFileName.Data(),"w");

 	cout << oFileName << endl;

	char c_junk[100];
	char c_run_ls[100];
	char c_instLumi[10];

	while (fscanf(pFile, "%s %s %s", c_run_ls, c_junk, c_instLumi) != EOF) {
// 		printf("%s = %s\n", c_run_ls, c_instLumi);
		TString s_instLumi(c_instLumi);
		TString s_Lumi(s_instLumi.Strip(TString::kTrailing,';'));
		double d_Lumi = s_Lumi.Atof();
		double d_corrLumi = correctedLumi(d_Lumi*23.3, nBX, correctionMode);
		printf("\t%s = %3.4lf ;\t%3.4lf ;\n", c_run_ls, d_Lumi,d_corrLumi/23.3);
		fprintf(outFile,"\t%s = %3.4lf ;\n", c_run_ls, d_corrLumi/23.3);
	}

}

double correctedLumi(double rawLumi, int nBX, int correctionMode){

   // rawLumi is assumed to be the lumi for one LS given in units of ub.

   // nBX is the number of colliding bunches (typically 1317 or 1331 for 2011)
   // nBX = 10 for Run 179828 (high PU run)

   // correction mode = 0 ==> no correction
   // correction mode = 2 ==> lumiCalc2.py  no good for high pile
   // correction mode = 3 ==> correction based on High PU scan

  

   double val = rawLumi;
   if(correctionMode == 0)return val;
   if(correctionMode != 2 && correctionMode != 3)return val;  

   double gamma = 1.141;
   double beta = 1.;
   if(nBX >= 1317){
      beta = 0.9748;
   } else if (nBX >= 1179) {
      beta = 0.9768;
   } else if (nBX >= 1041) {
      beta = 0.9788;
   } else if (nBX >= 873) {
      beta = 0.9788;
   } else if (nBX >= 700) {
      beta = 0.9836;
   } else if (nBX >= 597) {
      beta = 0.9851;
   } else if (nBX >= 423) {
      beta = 0.9881;
   } else if (nBX >= 321) {
      beta = 0.9902;
   } else if (nBX >= 213) {
      beta = 0.9916;
   }

   double fOrbit = 11246.;
   double secondsPerLS = pow(2, 18) / fOrbit;
   double Ldot = rawLumi/nBX/secondsPerLS;
   double alpha = 0.076;
   val = rawLumi*gamma*beta*(1. - alpha*Ldot); 
   if(correctionMode == 2)return val;

   // from vertex
   double alpha_1 = 0.0700;
   double alpha_2 = -0.00406;
   alpha_2 = -0.0045;
   val = rawLumi*gamma*beta/(1. + alpha_1*Ldot + alpha_2*Ldot*Ldot);
   return val;
}


   

