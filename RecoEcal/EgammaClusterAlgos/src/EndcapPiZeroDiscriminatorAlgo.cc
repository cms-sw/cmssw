
#include "RecoEcal/EgammaClusterAlgos/interface/EndcapPiZeroDiscriminatorAlgo.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <fstream>
#include <iostream>

using namespace std;

EndcapPiZeroDiscriminatorAlgo::EndcapPiZeroDiscriminatorAlgo(double stripEnergyCut, int nStripCut, const string& path,
                                 DebugLevel_pi0 debugLevel = pINFO ) :
   preshStripEnergyCut_(stripEnergyCut),  preshSeededNstr_(nStripCut), debugLevel_(debugLevel), pathToFiles_(path)
{   
   
     // Read all Weight files
     Nfiles_EB = 5;
     Nfiles_EE = 5;
     string file_pt[5] = {"20","30","40","50","60"};
     string file_barrel_pt[5] = {"20","30","40","50","60"};   

     string nn_paterns_file  = "";
     for(int j=0;j<Nfiles_EE;j++) {
       nn_paterns_file = "endcapPiZeroDiscriminatorWeights_et"+file_pt[j]+".wts";
       edm::FileInPath WFile(pathToFiles_+nn_paterns_file);
       readWeightFile(WFile.fullPath().c_str()); // read the weights' file

       EE_Layers = Layers;
       EE_Indim = Indim;
       EE_Hidden = Hidden;
       EE_Outdim = Outdim;
          
       for(int i=0;i<Indim*Hidden;i++)  I_H_Weight_all.push_back(I_H_Weight[i]);
       for(int i=0;i<Hidden;i++)  H_Thresh_all.push_back(H_Thresh[i]);
       for(int i=0;i<Outdim*Hidden;i++)  H_O_Weight_all.push_back(H_O_Weight[i]);
       for(int i=0;i<Outdim;i++)  O_Thresh_all.push_back(O_Thresh[i]);
     }

     for(int k=0;k<Nfiles_EB;k++) {
       nn_paterns_file = "barrelPiZeroDiscriminatorWeights_et"+file_barrel_pt[k]+".wts";
       edm::FileInPath WFile(pathToFiles_+nn_paterns_file);
       readWeightFile(WFile.fullPath().c_str()); // read the weights' file

       EB_Layers = Layers;
       EB_Indim = Indim;
       EB_Hidden = Hidden;
       EB_Outdim = Outdim;

       for(int i=0;i<Indim*Hidden;i++)  I_H_Weight_all.push_back(I_H_Weight[i]);
       for(int i=0;i<Hidden;i++)  H_Thresh_all.push_back(H_Thresh[i]);
       for(int i=0;i<Outdim*Hidden;i++)  H_O_Weight_all.push_back(H_O_Weight[i]);
       for(int i=0;i<Outdim;i++)  O_Thresh_all.push_back(O_Thresh[i]);
     }
   delete I_H_Weight;
   delete H_Thresh;
   delete H_O_Weight;
   delete O_Thresh;
}


vector<float> EndcapPiZeroDiscriminatorAlgo::findPreshVector(ESDetId strip,  RecHitsMap *rechits_map,
                                                     CaloSubdetectorTopology *topology_p)
{
  vector<float> vout_stripE;

  vout_stripE.clear();

  vector<ESDetId> road_2d;
  road_2d.clear();

  int plane = strip.plane();

  if ( debugLevel_ <= pDEBUG ) {
    cout << "findPreshVectors: Preshower Seeded Algorithm - looking for clusters" << "n"
              << "findPreshVectors: Preshower is intersected at strip " << strip.strip() << ", at plane " << plane << endl;
  }

  if ( strip == ESDetId(0) ) { //works in case of no intersected strip found
    for(int i=0;i<11;i++) {
       vout_stripE.push_back(-100.);
    }
  }

  // Add to the road the central strip
  road_2d.push_back(strip);

  //Make a navigator, and set it to the strip cell.
  EcalPreshowerNavigator navigator(strip, topology_p);
  navigator.setHome(strip);
 //search for neighbours in the central road
  findPi0Road(strip, navigator, plane, road_2d);
  if ( debugLevel_ <= pDEBUG ) cout << "findPreshVectors: Total number of strips in the central road: " << road_2d.size() << endl;

  // Find the energy of each strip
  RecHitsMap::iterator final_strip =  rechits_map->end();
  final_strip--;
  ESDetId last_stripID = final_strip->first;

  float E = 0;
  vector<ESDetId>::iterator itID;
  for (itID = road_2d.begin(); itID != road_2d.end(); itID++) {
    if ( debugLevel_ == pDEBUG ) cout << " findPreshVectors: ID = " << *itID << endl;
    RecHitsMap::iterator strip_it = rechits_map->find(*itID);
    if(goodPi0Strip(strip_it,last_stripID)) { // continue if strip not found in rechit_map
      E = strip_it->second.energy();
    } else  E = 0;
    vout_stripE.push_back(E);
    if ( debugLevel_ == pDEBUG ) cout << " findPreshVectors: E = " << E <<  endl;
  }

  return vout_stripE;

}

// returns true if the candidate strip fulfills the requirements to be added to the cluster:
bool EndcapPiZeroDiscriminatorAlgo::goodPi0Strip(RecHitsMap::iterator candidate_it, ESDetId lastID)
{
  RecHitsMap::iterator candidate_tmp = candidate_it;
  candidate_tmp--;
  if ( debugLevel_ == pDEBUG ) {
    if (candidate_tmp->first == lastID )
        cout << " goodPi0Strip No such a strip in rechits_map " << endl;
    if (candidate_it->second.energy() <= preshStripEnergyCut_)
        cout << " goodPi0Strip Strip energy " << candidate_it->second.energy() <<" is below threshold " << endl;
  }
  // crystal should not be included...
  if ( (candidate_tmp->first == lastID )                    ||       // ...if it corresponds to a hit
       (candidate_it->second.energy() <= preshStripEnergyCut_ ) )   // ...if it has a negative or zero energy
    {
      return false;
    }

  return true;
}

// find strips in the road of size +/- preshSeededNstr_ from the central strip
void EndcapPiZeroDiscriminatorAlgo::findPi0Road(ESDetId strip, EcalPreshowerNavigator& theESNav,
                                                                int plane, vector<ESDetId>& vout) {
  if ( strip == ESDetId(0) ) return;
   ESDetId next;
   theESNav.setHome(strip);

   if ( debugLevel_ <= pDEBUG ) cout << "findPi0Road: starts from strip " << strip << endl;
   if (plane == 1) {
     // east road
     int n_east= 0;
     if ( debugLevel_ == pDEBUG ) cout << " findPi0Road: Go to the East " <<  endl;
     while ( ((next=theESNav.east()) != ESDetId(0) && next != strip) ) {
        if ( debugLevel_ == pDEBUG ) cout << "findPi0Road: East: " << n_east << " current strip is " << next << endl;
        vout.push_back(next);
        ++n_east;
        if (n_east == preshSeededNstr_) break;
     }
     // west road
     int n_west= 0;
     if ( debugLevel_ == pDEBUG ) cout << " findPi0Road: Go to the West " <<  endl;
     theESNav.home();
     while ( ((next=theESNav.west()) != ESDetId(0) && next != strip )) {
        if ( debugLevel_ == pDEBUG ) cout << "findPi0Road: West: " << n_west << " current strip is " << next << endl;
        vout.push_back(next);
        ++n_west;
        if (n_west == preshSeededNstr_) break;
     }
     if ( debugLevel_ == pDEBUG ) cout << "findPi0Road: Total number of strips found in the road at 1-st plane is " << n_east+n_west << endl;
  }
  else if (plane == 2) {
    // north road
    int n_north= 0;
    if ( debugLevel_ == pDEBUG ) cout << " findPi0Road: Go to the North " <<  endl;
    while ( ((next=theESNav.north()) != ESDetId(0) && next != strip) ) {
       if ( debugLevel_ == pDEBUG ) cout << "findPi0Road: North: " << n_north << " current strip is " << next << endl;
       vout.push_back(next);
       ++n_north;
       if (n_north == preshSeededNstr_) break;
    }
    // south road
    int n_south= 0;
    if ( debugLevel_ == pDEBUG ) cout << " findPi0Road: Go to the South " <<  endl;
    theESNav.home();
    while ( ((next=theESNav.south()) != ESDetId(0) && next != strip) ) {
       if ( debugLevel_ == pDEBUG ) cout << "findPi0Road: South: " << n_south << " current strip is " << next << endl;
       vout.push_back(next);
       ++n_south;
       if (n_south == preshSeededNstr_) break;
    }
    if ( debugLevel_ == pDEBUG ) cout << "findPi0Road: Total number of strips found in the road at 2-nd plane is " << n_south+n_north << endl;
  }
  else {
    if ( debugLevel_ == pDEBUG ) cout << " findPi0Road: Wrong plane number, null cluster will be returned! " << endl;
  } // end of if

  theESNav.home();
}


//===================================================================
// EndcapPiZeroDiscriminatorAlgo::readWeightFile(...), a method that reads the weigths of the NN
// INPUT: Weights_file
// OUTPUT: I_H_Weight, H_Thresh, H_O_Weight, O_Thresh arrays
//===================================================================
void EndcapPiZeroDiscriminatorAlgo::readWeightFile(const char *Weights_file){
   FILE *weights;

   char *line;
   line = new char[80];

   bool checkinit=false;
// Open the weights file, generated by jetnet, and read
// in the nodes and weights
//*******************************************************
  weights = fopen(Weights_file, "r");
  if ( debugLevel_ <= pDEBUG ) cout << " I opeded the Weights file  = " << Weights_file << endl;

  while( !feof(weights) ){
	fscanf(weights, "%s", line);
  	if (line[0] == 'A') { //Read in ANN nodes: Layers, input , Hidden, Output
	   fscanf(weights, "%d", &Layers);	 // # of NN Layers  used
	   fscanf(weights, "%d", &Indim);	 // # of Inputs actually used
	   fscanf(weights, "%d", &Hidden);	 // # of hidden nodes
	   fscanf(weights, "%d", &Outdim);   // # of output nodes

           inp_var = Indim + 1;

   	   I_H_Weight = new float[Indim*Hidden];
	   H_Thresh = new float[Hidden];
	   H_O_Weight = new float[Hidden*Outdim];
	   O_Thresh = new float[Outdim];
	   checkinit=true;
	}else if (line[0] == 'B') { // read in weights between hidden and intput nodes
	    assert(checkinit);
	    for (int i = 0; i<Indim; i++){
                for (int j = 0; j<Hidden; j++){
                    fscanf(weights, "%f", &I_H_Weight[i*Hidden+j]);
                }
	    }
	}else if (line[0] == 'C'){	 // Read in the thresholds for hidden nodes
	    assert(checkinit);
	    for (int i = 0; i<Hidden; i++){
		fscanf(weights, "%f", &H_Thresh[i]);
	    }
	}else if (line[0] == 'D'){ // read in weights between hidden and output nodes
	    assert(checkinit);
	    for (int i = 0; i<Hidden*Outdim; i++){
		fscanf(weights, "%f", &H_O_Weight[i]);
	    }
	}else if (line[0] == 'E'){ // read in the threshold for the output nodes
	    assert(checkinit);
	    for (int i = 0; i<Outdim; i++){
		fscanf(weights, "%f", &O_Thresh[i]);

	    }
        }
          else{cout << "Not a Net file of Corrupted Net file " << endl;
        }
   }
   fclose(weights);
}

//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::getNNoutput(int sel_wfile), a method that calculated the NN output
// INPUT: sel_wfile -> Weight file selection
// OUTPUT : nnout -> the NN output
//=====================================================================================

float EndcapPiZeroDiscriminatorAlgo::getNNoutput(int sel_wfile)
{
 float* I_SUM;
 float* OUT;
 float nnout=0.0;
 int mij;

 I_SUM = new float[Hidden];
 OUT = new  float[Outdim];

 for(int k=0;k<Hidden;k++) I_SUM[k]=0.0;
 for(int k1=0;k1<Outdim;k1++) OUT[k1]=0.0;

 for (int h = 0; h<Hidden; h++){
     mij = h - Hidden;
     for (int i = 0; i<Indim; i++){
         mij = mij + Hidden;
         I_SUM[h] += I_H_Weight_all[mij+sel_wfile*Indim*Hidden + barrelstart*Nfiles_EE*EE_Indim*EE_Hidden] * input_var[i];
     }
     I_SUM[h] += H_Thresh_all[h+sel_wfile*Hidden + barrelstart*Nfiles_EE*EE_Hidden];
     for (int o1 = 0; o1<Outdim; o1++) {
        OUT[o1] += H_O_Weight_all[barrelstart*Nfiles_EE*EE_Outdim*EE_Hidden + h*Outdim+o1 + sel_wfile*Outdim*Hidden]*Activation_fun(I_SUM[h]); 

     }
 }
 for (int o2 = 0; o2<Outdim; o2++){      
	        OUT[o2] += O_Thresh_all[barrelstart*Nfiles_EE*EE_Outdim + o2 + sel_wfile*Outdim]; 
  }
  nnout = Activation_fun(OUT[0]);

  if ( debugLevel_ <= pDEBUG ) cout << "getNNoutput :: -> NNout = " <<  nnout << endl;

  delete I_SUM;
  delete OUT;
  delete input_var;
   
  return (nnout);
}


float EndcapPiZeroDiscriminatorAlgo::Activation_fun(float SUM){
        return( 1.0 / ( 1.0 + exp(-2.0*SUM) ) );
}
//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::calculateNNInputVariables(...), a method that calculates the 25 input variables
// INPUTS:
// vph1 -> vector of the stip energies in 1st Preshower plane
// vph2 -> vector of the stip energies in 2nd Preshower plane
// pS1_max -> E1
// pS9_max -> E9
// pS25_max -> E25
// OUTPUT: 
// input_var[25] -> the 25 input to the NN variables array
//=====================================================================================
bool EndcapPiZeroDiscriminatorAlgo::calculateNNInputVariables(vector<float>& vph1, vector<float>& vph2, 
                                          float pS1_max, float pS9_max, float pS25_max)
{
   input_var = new float[EE_Indim];

   bool valid_NNinput = true;
   
   if ( debugLevel_ <= pDEBUG ) {
     cout << "Energies of the Preshower Strips in X plane = ( "; 
     for(int i = 0; i<11;i++) {
        cout << " " << vph1[i];
     }
     cout << ")" << endl;
     cout << "Energies of the Preshower Strips in Y plane = ( "; 
     for(int i = 0; i<11;i++) {
        cout << " " << vph2[i];
     }
     cout << ")" << endl;
   }
   // check if all Preshower info is availabla - If NOT use remaning info
   for(int k = 0; k<11; k++) {
      if(vph1[k] < 0 ) {
         if ( debugLevel_ <= pDEBUG ) { 
	   cout  << "Oops!!! Preshower Info for strip : " << k << " of X plane Do not exists" << endl; 
	 }  
         vph1[k] = 0.0;
      } 	 
      if(vph2[k] < 0 ) { 
        if ( debugLevel_ <= pDEBUG ) { 
	  cout  << "Oops!!! Preshower Info for strip : " << k << " of Y plane Do not exists" << endl;
        }
        vph2[k] = 0.0;
      }
   }   
   if ( debugLevel_ <= pDEBUG ) {
     cout << "After: Energies of the Preshower Strips in X plane = ( "; 
     for(int i = 0; i<11;i++) {
        cout << " " << vph1[i];
     }
     cout << ")" << endl;
     cout << "After: Energies of the Preshower Strips in Y plane = ( "; 
     for(int i = 0; i<11;i++) {
        cout << " " << vph2[i];
     }
     cout << ")" << endl;
   }

// FIRST : Produce the 22 NN variables related with the Preshower 
// --------------------------------------------------------------
// New normalization of the preshower strip energies Aris 8/11/2004
   for(int kk=0;kk<11;kk++){
     input_var[kk] = fabs(vph1[kk]/0.01);
     input_var[kk + 11] = fabs(vph2[kk]/0.02);       
     if(input_var[kk] < 0.0001) input_var[kk] = 0.;
     if(input_var[kk + 11] < 0.0001) input_var[kk + 11] = 0.;
   }
   input_var[0] = fabs(input_var[0]/2.); 
   input_var[1] = fabs(input_var[1]/2.); 
   input_var[6] = fabs(input_var[6]/2.); 
   input_var[11] = fabs(input_var[11]/2.); 
   input_var[12] = fabs(input_var[12]/2.); 
   input_var[17] = fabs(input_var[17]/2.); 

// SECOND: Take the final NN variable related to the ECAL
// -----------------------------------------------
   input_var[22] = pS1_max/500.;
   input_var[23] = pS9_max/500.;
   input_var[24] = pS25_max/500.;

   if ( debugLevel_ <= pDEBUG ) {
     cout << "S1/500. = " << input_var[22] << endl;
     cout << "S9/500. = " << input_var[23] << endl;
     cout << "S25/500. = " << input_var[24] << endl;
   }
   for(int i=0;i<EE_Indim;i++){
     if(input_var[i] > 1.0e+00) {
       valid_NNinput = false;
       break;
     }
   }
   if ( debugLevel_ <= pDEBUG ) {
     cout << " valid_NNinput = " << valid_NNinput << endl; }
   return valid_NNinput;
}


//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::calculateBarrelNNInputVariables(...), a method that calculates
// the 12 barrel NN input
// OUTPUT:
// input_var[12] -> the 12 input to the barrel NN variables array
//=====================================================================================

void EndcapPiZeroDiscriminatorAlgo::calculateBarrelNNInputVariables(float et, double s1, double s9, double s25,
								    double m2, double cee, double cep,double cpp,
								    double s4, double s6, double ratio,
								    double xcog, double ycog)
{
  input_var = new float[EB_Indim];

  double lam, lam1, lam2;
  if(xcog < 0.)  input_var[0] = -xcog/s25;  else  input_var[0] = xcog/s25;
  if(ycog < 0.)  input_var[6] = -ycog/s25;  else  input_var[6] = ycog/s25;
  input_var[1] = cee/0.0004;
  if(cpp<.001)    input_var[2] = cpp/.001;  else    input_var[2] = 0.;
  if(s9!=0.) {   input_var[3] = s1/s9; input_var[8] = s6/s9; input_var[10] = (m2+s1)/s9; }
     else {    input_var[3] = 0.; input_var[8] = 0.;  input_var[10] = 0.; }
  if(s25-s1>0.) input_var[4] = (s9-s1)/(s25-s1);  else input_var[4] = 0.;
  if(s25>0.) input_var[5] = s4/s25;  else input_var[5] = 0.;
   lam=sqrt((cee -cpp)*(cee -cpp)+4*cep*cep); lam1=(cee + cpp + lam)/2; lam2=(cee + cpp - lam)/2;
   if(lam1 == 0) input_var[9] = .0; else input_var[9] = lam2/lam1;
  if(s4!=0.) input_var[11] = (m2+s1)/s4; else input_var[11] = 0.;
   input_var[7] = ratio;
}


//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::GetNNOutput(...), a method that calculates the NNoutput
// INPUTS: Super Cluster Energy
// OUTPUTS : NNoutput
//=====================================================================================
float EndcapPiZeroDiscriminatorAlgo::GetNNOutput(float EE_Et)
{
    Layers = EE_Layers; Indim = EE_Indim; Hidden = EE_Hidden; Outdim = EE_Outdim;  barrelstart = 0;  

    float nnout = -1;
// Print the  NN input variables that are related to the Preshower + ECAL
// ------------------------------------------------------------------------
     if ( debugLevel_ <= pDEBUG )cout << " EndcapPiZeroDiscriminatorAlgo::GetNNoutput :nn_invar_presh = " ;
     for(int k1=0;k1<Indim;k1++) {
        if ( debugLevel_ <= pDEBUG )cout << input_var[k1] << " " ;
     }
     if ( debugLevel_ <= pDEBUG )cout << endl;

     // select the appropriate Weigth file
     int sel_wfile;
     if(EE_Et<25.0)                     {sel_wfile = 0;}
     else if(EE_Et>=25.0 && EE_Et<35.0) {sel_wfile = 1;}
     else if(EE_Et>=35.0 && EE_Et<45.0) {sel_wfile = 2;}
     else if(EE_Et>=45.0 && EE_Et<55.0) {sel_wfile = 3;}
     else                               {sel_wfile = 4;} 

     if ( debugLevel_ <= pDEBUG ) {
         cout << " Et_SC = " << EE_Et << " and I select Weight file Number = " << sel_wfile << endl; 
     }

     nnout = getNNoutput(sel_wfile); // calculate the nnoutput for the given ECAL object
     if ( debugLevel_ <= pDEBUG ) cout << "===================> GetNNOutput : NNout = " << nnout <<  endl;
   return nnout;
}


//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::GetBarrelNNOutput(...), a method that calculates the barrel NNoutput
// INPUTS: Super Cluster Energy
// OUTPUTS : NNoutput
//=====================================================================================
float EndcapPiZeroDiscriminatorAlgo::GetBarrelNNOutput(float EB_Et)
{

    Layers = EB_Layers; Indim = EB_Indim; Hidden = EB_Hidden; Outdim = EB_Outdim;  barrelstart = 1;  

    float nnout = -1;
// Print the  NN input variables that are related to the ECAL Barrel
// ------------------------------------------------------------------------
     if ( debugLevel_ <= pDEBUG )cout << " EndcapPiZeroDiscriminatorAlgo::GetBarrelNNoutput :nn_invar_presh = " ;
     for(int k1=0;k1<Indim;k1++) {
        if ( debugLevel_ <= pDEBUG )cout << input_var[k1] << " " ;
     }
     if ( debugLevel_ <= pDEBUG )cout << 1 << endl;

     // select the appropriate Weigth file
     int sel_wfile;
     if(EB_Et<25.0)                     {sel_wfile = 0;}
     else if(EB_Et>=25.0 && EB_Et<35.0) {sel_wfile = 1;}
     else if(EB_Et>=35.0 && EB_Et<45.0) {sel_wfile = 2;}
     else if(EB_Et>=45.0 && EB_Et<55.0) {sel_wfile = 3;}
     else                               {sel_wfile = 4;}

     if ( debugLevel_ <= pDEBUG ) {
         cout << " E_SC = " << EB_Et << " and I select Weight file Number = " << sel_wfile << endl;
     }

     nnout = getNNoutput(sel_wfile); // calculate the nnoutput for the given ECAL object
     if ( debugLevel_ <= pDEBUG ) cout << "===================> GetNNOutput : NNout = " << nnout <<  endl;

   return nnout;
}

