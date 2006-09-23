
#include "RecoEcal/EgammaClusterAlgos/interface/EndcapPiZeroDiscriminatorAlgo.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
//#include "FWCore/MessageService/interface/MessageLogger.h"
//using edm::LogInfo;
//#include <sstream>
#include <fstream>
#include <iostream>


// Aris 10/7/2006
// ---------------
//void EndcapPiZeroDiscriminatorAlgo::findPreshVector(ESDetId strip,  RecHitsMap *rechits_map,
//                                                     CaloSubdetectorTopology *topology_p, std::vector<float>& vout_stripE)
std::vector<float> EndcapPiZeroDiscriminatorAlgo::findPreshVector(ESDetId strip,  RecHitsMap *rechits_map,
                                                     CaloSubdetectorTopology *topology_p)
{
  std::vector<float> vout_stripE;
  
  vout_stripE.clear();

  std::vector<ESDetId> road_2d;
  road_2d.clear();

  int plane = strip.plane();

  if ( debugLevel_ <= pDEBUG ) {
    std::cout << "findPreshVectors: Preshower Seeded Algorithm - looking for clusters" << "n"
              << "findPreshVectors: Preshower is intersected at strip " << strip.strip() << ", at plane " << plane << std::endl;
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
  if ( debugLevel_ <= pDEBUG ) std::cout << "findPreshVectors: Total number of strips in the central road: " << road_2d.size() << std::endl;

  // Find the energy of each strip 
  RecHitsMap::iterator final_strip =  rechits_map->end();
  final_strip--;
  ESDetId last_stripID = final_strip->first;

  float E = 0;
  std::vector<ESDetId>::iterator itID;
  for (itID = road_2d.begin(); itID != road_2d.end(); itID++) {
    if ( debugLevel_ == pDEBUG ) std::cout << " findPreshVectors: ID = " << *itID << std::endl;
    RecHitsMap::iterator strip_it = rechits_map->find(*itID);
    if(goodPi0Strip(strip_it,last_stripID)) { // continue if strip not found in rechit_map  
      E = strip_it->second.energy();
    } else  E = 0; 
    vout_stripE.push_back(E);
    if ( debugLevel_ == pDEBUG ) std::cout << " findPreshVectors: E = " << E <<  std::endl;
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
        std::cout << " goodPi0Strip No such a strip in rechits_map " << std::endl; 
    if (candidate_it->second.energy() <= preshStripEnergyCut_)
        std::cout << " goodPi0Strip Strip energy " << candidate_it->second.energy() <<" is below threshold " << std::endl; 
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
                                                                int plane, std::vector<ESDetId>& vout) {
  if ( strip == ESDetId(0) ) return;
//   int preshSeededNstr_ = 5; 
   ESDetId next;
   theESNav.setHome(strip);

   if ( debugLevel_ <= pDEBUG ) std::cout << "findPi0Road: starts from strip " << strip << std::endl;  
   if (plane == 1) {
     // east road
     int n_east= 0;
     if ( debugLevel_ == pDEBUG ) std::cout << " findPi0Road: Go to the East " <<  std::endl;   
     while ( ((next=theESNav.east()) != ESDetId(0) && next != strip) ) {
        if ( debugLevel_ == pDEBUG ) std::cout << "findPi0Road: East: " << n_east << " current strip is " << next << std::endl;  
        vout.push_back(next);   
        ++n_east;  
        if (n_east == preshSeededNstr_) break; 
     }
     // west road
     int n_west= 0;
     if ( debugLevel_ == pDEBUG ) std::cout << " findPi0Road: Go to the West " <<  std::endl;
     theESNav.home();
     while ( ((next=theESNav.west()) != ESDetId(0) && next != strip )) {
        if ( debugLevel_ == pDEBUG ) std::cout << "findPi0Road: West: " << n_west << " current strip is " << next << std::endl;  
        vout.push_back(next);   
        ++n_west;  
        if (n_west == preshSeededNstr_) break; 
     }
     if ( debugLevel_ == pDEBUG ) std::cout << "findPi0Road: Total number of strips found in the road at 1-st plane is " << n_east+n_west << std::endl;
  } 
  else if (plane == 2) {
    // north road
    int n_north= 0;
    if ( debugLevel_ == pDEBUG ) std::cout << " findPi0Road: Go to the North " <<  std::endl;
    while ( ((next=theESNav.north()) != ESDetId(0) && next != strip) ) {       
       if ( debugLevel_ == pDEBUG ) std::cout << "findPi0Road: North: " << n_north << " current strip is " << next << std::endl; 
       vout.push_back(next);   
       ++n_north;  
       if (n_north == preshSeededNstr_) break; 
    }
    // south road
    int n_south= 0;
    if ( debugLevel_ == pDEBUG ) std::cout << " findPi0Road: Go to the South " <<  std::endl;
    theESNav.home();
    while ( ((next=theESNav.south()) != ESDetId(0) && next != strip) ) {
       if ( debugLevel_ == pDEBUG ) std::cout << "findPi0Road: South: " << n_south << " current strip is " << next << std::endl;      
       vout.push_back(next);   
       ++n_south;  
       if (n_south == preshSeededNstr_) break; 
    }
    if ( debugLevel_ == pDEBUG ) std::cout << "findPi0Road: Total number of strips found in the road at 2-nd plane is " << n_south+n_north << std::endl;
  } 
  else {
    if ( debugLevel_ == pDEBUG ) std::cout << " findPi0Road: Wrong plane number, null cluster will be returned! " << std::endl;    
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
  if ( debugLevel_ <= pDEBUG ) std::cout << " I opeded the Weights file  = " << Weights_file << std::endl;
  while( !feof(weights) ){
	fscanf(weights, "%s", line);
  	if (line[0] == 'A') { //Read in ANN nodes: Layers, Input , Hidden, Output
//           cout << " The ANN parwmeters are : " << endl;
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
          else{std::cout << "Not a Net file of Corrupted Net file " << std::endl;
        }						
   } 
   fclose(weights); 
}

//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::getNNoutput(float *input), a method that calculated the NN output 
// INPUT: input[25] -> the 25 input to the NN variables
// OUTPUT : nnout -> the NN output
//=====================================================================================
float EndcapPiZeroDiscriminatorAlgo::getNNoutput(float *input)
{
 float I_SUM[26],OUT[1];
 float nnout=0.0;
 int mij;

 for(int k=0;k<Hidden;k++) I_SUM[k]=0.0;
 for(int k1=0;k1<Outdim;k1++) OUT[k1]=0.0;

 for (int h = 0; h<Hidden; h++){
     mij = h - Hidden;
     for (int i = 0; i<Indim; i++){
         mij = mij + Hidden;
         I_SUM[h] += I_H_Weight[mij] * input[i];
     }
     I_SUM[h] += H_Thresh[h];
     for (int o1 = 0; o1<Outdim; o1++) {
        OUT[o1] += H_O_Weight[h*Outdim+o1]*Activation_fun(I_SUM[h]);
     }
 }
 for (int o2 = 0; o2<Outdim; o2++){
        OUT[o2] += O_Thresh[o2];
  }
  nnout = Activation_fun(OUT[0]);

  if ( debugLevel_ <= pDEBUG ) std::cout << "getNNoutput :: -> NNout = " <<  nnout << std::endl;

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
// PhCand -> an iterator over the Photon Candidates
// OUTPUT: 
// nn_invar[25] -> the 25 input to the NN variables array
// f9 -> E9/Esc
// f25 -> E25/Esc
//=====================================================================================
void EndcapPiZeroDiscriminatorAlgo::calculateNNInputVariables(std::vector<float>& vph1, std::vector<float>& vph2, 
                                           float pS1_max, float pS9_max, float pS25_max,
                                           float *nn_invar)
{

   if ( debugLevel_ <= pDEBUG ) {
     std::cout << "Energies of the Preshower Strips in X plane = ( "; 
     for(int i = 0; i<11;i++) {
        std::cout << " " << vph1[i];
     }
     std::cout << std::endl;
     std::cout << "Energies of the Preshower Strips in Y plane = ( "; 
     for(int i = 0; i<11;i++) {
        std::cout << " " << vph2[i];
     }
     std::cout << std::endl;
   }

// FIRST : Produce the 22 NN variables related with the Preshower 
// --------------------------------------------------------------
// New normalization of the preshower strip energies Aris 8/11/2004
     for(int kk=0;kk<11;kk++)
      {
       nn_invar[kk] = fabs(vph1[kk]/0.01);
       nn_invar[kk + 11] = fabs(vph2[kk]/0.02);       
      }
      nn_invar[0] = fabs(nn_invar[0]/2.); 
      nn_invar[1] = fabs(nn_invar[1]/2.); 
      nn_invar[6] = fabs(nn_invar[6]/2.); 
      nn_invar[11] = fabs(nn_invar[11]/2.); 
      nn_invar[12] = fabs(nn_invar[12]/2.); 
      nn_invar[17] = fabs(nn_invar[17]/2.); 

// SECOND: Take the final NN variable related to the ECAL
// -----------------------------------------------
      nn_invar[22] = pS1_max/500.;
      nn_invar[23] = pS9_max/500.;
      nn_invar[24] = pS25_max/500.;

      if ( debugLevel_ <= pDEBUG ) {
        std::cout << "S1/500. = " << nn_invar[22] << std::endl;
        std::cout << "S9/500. = " << nn_invar[23] << std::endl;
        std::cout << "S25/500. = " << nn_invar[24] << std::endl;
      }
}

float EndcapPiZeroDiscriminatorAlgo::GetNNOutput(float Et_SE, float *nn_invar_presh) 
{
    float nnout = -10;
    float input_var[25]; // array with the 25 variables to be used as input in NN
// Print the 25 NN input variables that are related to the Preshower + ECAL
// ------------------------------------------------------------------------
     if ( debugLevel_ <= pDEBUG )std::cout << " PreshNNoutput :nn_invar_presh = " ;
     for(int k1=0;k1<25;k1++) {
        input_var[k1] = nn_invar_presh[k1];
        if ( debugLevel_ <= pDEBUG )std::cout << input_var[k1] << " " ;
     }
     if ( debugLevel_ <= pDEBUG )std::cout << std::endl;

     // Choose the correct file according to the cluster's energy
     std::string nn_paterns_file  = "";
     if(Et_SE<25.0)                     {nn_paterns_file = "endcapPiZeroDiscriminatorWeights_et20.wts"; }
     else if(Et_SE>=25.0 && Et_SE<35.0) {nn_paterns_file = "endcapPiZeroDiscriminatorWeights_et30.wts"; }
     else if(Et_SE>=35.0 && Et_SE<45.0) {nn_paterns_file = "endcapPiZeroDiscriminatorWeights_et40.wts"; }
     else if(Et_SE>=45.0 && Et_SE<55.0) {nn_paterns_file = "endcapPiZeroDiscriminatorWeights_et50.wts"; }
     else                               {nn_paterns_file = "endcapPiZeroDiscriminatorWeights_et60.wts"; }

     edm::FileInPath WFile(pathToFiles_+nn_paterns_file);
     readWeightFile(WFile.fullPath().c_str()); // read the weights' file

     nnout = getNNoutput(input_var); // calculate the nnoutput for the given ECAL object
     if ( debugLevel_ <= pDEBUG ) std::cout << "***************PreshNNoutput : NNout = " << nnout <<  std::endl;  
   return nnout;
}
