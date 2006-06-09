/** 
 * Analyzer for reading CFEB comparator information
 * author O.Boeriu 9/05/06 
 *   
 */

#include <iostream>
#include "OnlineDB/CSCCondDB/interface/condbc.h"
#include "OnlineDB/CSCCondDB/interface/cscmap.h"

class CSCCompThreshAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCCompThreshAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS_ct 9
#define LAYERS_ct 6
#define STRIPS_ct 80
#define TOTALSTRIPS_ct 480
#define DDU_ct 9
#define NUMMOD_ct 875
#define NUMBERPLOTTED_ct 25

  ~CSCCompThreshAnalyzer(){
 
    //create array (480 entries) for database transfer
    //condbc *cdb = new condbc();
    //cscmap *map = new cscmap();

    for(int myChamber=0; myChamber<NChambers; myChamber++){

      //float meanComp = 0.;
      std::string test1="CSC_slice";
      std::string test2="comparator_threshold";
      std::string answer;

      //print out result here
      for (int i=0; i<CHAMBERS_ct; i++){
	if (myChamber !=i) continue;
	
	for (int j=0; j<LAYERS_ct; j++){
	  for (int k=0; k<STRIPS_ct; k++){
	    //arrayMeanThresh[i][j][k]= meanmod[tmp][i_chamber][i_layer-1][mycompstrip];
	    fff = (j*80)+k;
	    //theMeanThresh  = arrayMeanThresh[i][j][k];
	    //newMeanThresh[fff]  = theRMS;
	    
	    //std::cout <<" chamber "<<i<<" layer "<<j<<" strip "<<fff<<"  ped "<<newPed[fff]<<" RMS "<<newRMS[fff]<<std::endl;
	  }
	}
      }
    }
  }

private:
  // variables persistent across events should be declared here.
  //
  
  std::string chamber_id;
  int eventNumber,evt,event,pedSum, strip, misMatch,fff,ret_code,NChambers,Nddu;
  int length,i_chamber,i_layer,reportedChambers,chamber_num,sector; 
  int timebin,mycompstrip,comparator,compstrip;
  int dmbID[CHAMBERS_ct],crateID[CHAMBERS_ct]; 
  float theMeanThresh[CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  float	arrayMeanThresh[CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  float	mean[CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  float	meanTot[CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  float meanmod[NUMMOD_ct][CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  float newThresh[TOTALSTRIPS_ct];

};
