/** 
 * Analyzer for reading CFEB comparator information
 * author O.Boeriu 9/05/06 
 *   
 */

#include <iostream>
#include "CalibMuon/CSCCalibration/interface/condbc.h"
#include "CalibMuon/CSCCalibration/interface/cscmap.h"

class CSCCompThreshAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCCompThreshAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS 9
#define LAYERS 6
#define STRIPS 80
#define TOTALSTRIPS 480
#define DDU 2
#define NUMMOD 875
#define NUMBERPLOTTED 25

  ~CSCCompThreshAnalyzer(){
 
    //create array (480 entries) for database transfer
    condbc *cdb = new condbc();
    cscmap *map = new cscmap();

    for(int myChamber=0; myChamber<NChambers; myChamber++){

      //float meanComp = 0.;
      std::string test1="CSC_slice";
      std::string test2="comparator_threshold";
      std::string answer;

      //print out result here
      for (int i=0; i<CHAMBERS; i++){
	if (myChamber !=i) continue;
	
	for (int j=0; j<LAYERS; j++){
	  for (int k=0; k<STRIPS; k++){
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
  int dmbID[CHAMBERS],crateID[CHAMBERS]; 
  float theMeanThresh[CHAMBERS][LAYERS][STRIPS];
  float	arrayMeanThresh[CHAMBERS][LAYERS][STRIPS];
  float	mean[CHAMBERS][LAYERS][STRIPS];
  float	meanTot[CHAMBERS][LAYERS][STRIPS];
  float meanmod[NUMMOD][CHAMBERS][LAYERS][STRIPS];
  float newThresh[TOTALSTRIPS];

};
