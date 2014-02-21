#include "../interface/PatternFinder.h"

PatternFinder::PatternFinder(int sp, int at, SectorTree* st, string f, string of){
  superStripSize = sp;
  active_threshold = at;
  sectors = st;
  eventsFilename = f;
  outputFileName = of;

  map< int, vector<int> > detector_config = Sector::readConfig("detector.cfg");

  //We add the layers corresponding to the sectors structure
  vector<Sector*> sectors = st->getAllSectors();
  if(sectors.size()>0){
    for(int i=0;i<sectors[0]->getNbLayers();i++){
      int layerID = sectors[0]->getLayerID(i);
      if(detector_config.size()>0){
	if(detector_config.find(layerID)!=detector_config.end())
	  tracker.addLayer(detector_config[layerID][0],detector_config[layerID][1],detector_config[layerID][2],detector_config[layerID][3], superStripSize);
	else
	  cout<<"WARNING : Layer "<<layerID<<" is used in the sector definition of the bank but is missing in the configuration of the virtual detector"<<endl;
      }
    }
  }

  //Link the patterns with the tracker representation
  cout<<"linking..."<<endl;
  st->link(tracker);
  cout<<"done."<<endl;
}

void PatternFinder::setSectorTree(SectorTree* s){
  sectors = s;
}

void PatternFinder::setEventsFile(string f){
  eventsFilename = f;
}

void PatternFinder::mergeFiles(string outputFile, string inputFile1, string inputFile2){

  const int MAX_NB_PATTERNS = 1500;
  const int MAX_NB_HITS = 100;
  const int MAX_NB_LADDERS_PER_LAYER = 16;
  const int MAX_NB_LAYERS = 8;

  /*********************INPUT 1 FILE *************************************/

  TChain *PATT1    = new TChain("Patterns"); // infos about patterns
  TChain *SEC1    = new TChain("Sectors");   //infos about sectors
  PATT1->Add(inputFile1.c_str());
  SEC1->Add(inputFile1.c_str());

  int input1_nb_layers;
  int input1_nb_patterns=0;
  int input1_ori_nb_stubs=0;
  int input1_sel_nb_stubs=0;
  int input1_event_id;
  int *input1_superStrip_layer_0 = new int[MAX_NB_PATTERNS];
  int *input1_superStrip_layer_1 = new int[MAX_NB_PATTERNS];
  int *input1_superStrip_layer_2 = new int[MAX_NB_PATTERNS];
  int *input1_superStrip_layer_3 = new int[MAX_NB_PATTERNS];
  int *input1_superStrip_layer_4 = new int[MAX_NB_PATTERNS];
  int *input1_superStrip_layer_5 = new int[MAX_NB_PATTERNS];
  int *input1_superStrip_layer_6 = new int[MAX_NB_PATTERNS];
  int *input1_superStrip_layer_7 = new int[MAX_NB_PATTERNS];
  int *input1_pattern_sector_id = new int[MAX_NB_PATTERNS];

  //Array containing the strips arrays
  int* input1_superStrips[MAX_NB_LAYERS];
  input1_superStrips[0]=input1_superStrip_layer_0;
  input1_superStrips[1]=input1_superStrip_layer_1;
  input1_superStrips[2]=input1_superStrip_layer_2;
  input1_superStrips[3]=input1_superStrip_layer_3;
  input1_superStrips[4]=input1_superStrip_layer_4;
  input1_superStrips[5]=input1_superStrip_layer_5;
  input1_superStrips[6]=input1_superStrip_layer_6;
  input1_superStrips[7]=input1_superStrip_layer_7;

  int input1_sector_id=0;
  int input1_sector_layers=0;
  int input1_nb_ladders_layer[MAX_NB_LAYERS];
  int input1_sector_layer_list[MAX_NB_LAYERS];
  int input1_sector_layer_0[MAX_NB_LADDERS_PER_LAYER];
  int input1_sector_layer_1[MAX_NB_LADDERS_PER_LAYER];
  int input1_sector_layer_2[MAX_NB_LADDERS_PER_LAYER];
  int input1_sector_layer_3[MAX_NB_LADDERS_PER_LAYER];
  int input1_sector_layer_4[MAX_NB_LADDERS_PER_LAYER];
  int input1_sector_layer_5[MAX_NB_LADDERS_PER_LAYER];
  int input1_sector_layer_6[MAX_NB_LADDERS_PER_LAYER];
  int input1_sector_layer_7[MAX_NB_LADDERS_PER_LAYER];

  int* input1_sector_layers_detail[MAX_NB_LAYERS];
  input1_sector_layers_detail[0]=input1_sector_layer_0;
  input1_sector_layers_detail[1]=input1_sector_layer_1;
  input1_sector_layers_detail[2]=input1_sector_layer_2;
  input1_sector_layers_detail[3]=input1_sector_layer_3;
  input1_sector_layers_detail[4]=input1_sector_layer_4;
  input1_sector_layers_detail[5]=input1_sector_layer_5;
  input1_sector_layers_detail[6]=input1_sector_layer_6;
  input1_sector_layers_detail[7]=input1_sector_layer_7;

  int *input1_nbHitPerPattern = new int[MAX_NB_PATTERNS];
  int input1_totalNbHits=0;
  int input1_nbTracks = 0;
  float *input1_track_pt = new float[MAX_NB_PATTERNS];
  float *input1_track_phi = new float[MAX_NB_PATTERNS];
  float *input1_track_d0 = new float[MAX_NB_PATTERNS];
  float *input1_track_eta = new float[MAX_NB_PATTERNS];
  float *input1_track_z0 = new float[MAX_NB_PATTERNS];
  short *input1_hit_layer = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  short *input1_hit_ladder = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  short *input1_hit_zPos = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  short *input1_hit_segment = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  short *input1_hit_strip = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  int *input1_hit_tp = new int[MAX_NB_PATTERNS*MAX_NB_HITS];
  int *input1_hit_idx = new int[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_ptGEN = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_etaGEN = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_phi0GEN = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_ip = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_x = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_y = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_z = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_X0 = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_Y0 = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input1_hit_Z0 = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  
  SEC1->SetBranchAddress("sectorID",            &input1_sector_id);//ID du secteur
  SEC1->SetBranchAddress("nbLayers",            &input1_sector_layers);// nombre de layers dans le secteur
  SEC1->SetBranchAddress("layer",                  input1_sector_layer_list);//layers ID
  SEC1->SetBranchAddress("nb_ladders_layer",       input1_nb_ladders_layer); // nombre de ladders pour chaque layer
  SEC1->SetBranchAddress("sectorLadders_layer_0",  input1_sector_layer_0);//liste des ladders pour layer 0
  SEC1->SetBranchAddress("sectorLadders_layer_1",  input1_sector_layer_1);
  SEC1->SetBranchAddress("sectorLadders_layer_2",  input1_sector_layer_2);
  SEC1->SetBranchAddress("sectorLadders_layer_3",  input1_sector_layer_3);
  SEC1->SetBranchAddress("sectorLadders_layer_4",  input1_sector_layer_4);
  SEC1->SetBranchAddress("sectorLadders_layer_5",  input1_sector_layer_5);
  
  PATT1->SetBranchAddress("nbLayers",            &input1_nb_layers);//nombre de layers pour les patterns
  PATT1->SetBranchAddress("nbPatterns",          &input1_nb_patterns); // nombre de patterns dans l'evenement
  PATT1->SetBranchAddress("nbStubsInEvt",        &input1_ori_nb_stubs); // nombre de stubs dans l'evenement initial
  PATT1->SetBranchAddress("nbStubsInPat",        &input1_sel_nb_stubs); // nombre de stubs uniques dans les patterns
  PATT1->SetBranchAddress("eventID",             &input1_event_id); // ID de l'evenement (le meme que dans le fichier de simulation)
  PATT1->SetBranchAddress("sectorID",            input1_pattern_sector_id);// ID du secteur du pattern (permet de retrouver le secteur dans le premier TTree)
  PATT1->SetBranchAddress("superStrip0",         input1_superStrip_layer_0);// tableau de superstrips pour le layer 0
  PATT1->SetBranchAddress("superStrip1",         input1_superStrip_layer_1);
  PATT1->SetBranchAddress("superStrip2",         input1_superStrip_layer_2);
  PATT1->SetBranchAddress("superStrip3",         input1_superStrip_layer_3);
  PATT1->SetBranchAddress("superStrip4",         input1_superStrip_layer_4);
  PATT1->SetBranchAddress("superStrip5",         input1_superStrip_layer_5);
  PATT1->SetBranchAddress("total_nb_stubs",      &input1_totalNbHits);
  PATT1->SetBranchAddress("nbTracks",            &input1_nbTracks);
  PATT1->SetBranchAddress("nbStubs",             input1_nbHitPerPattern); // nombre de stubs contenus dans chaque pattern

  PATT1->SetBranchAddress("track_pt",            input1_track_pt); // PT of fitted tracks
  PATT1->SetBranchAddress("track_phi",           input1_track_phi); // PHI0 of fitted tracks
  PATT1->SetBranchAddress("track_d0",            input1_track_d0); // D0 of fitted tracks
  PATT1->SetBranchAddress("track_eta",           input1_track_eta); // ETA of fitted tracks
  PATT1->SetBranchAddress("track_z0",            input1_track_z0); // Z0 of fitted tracks

  PATT1->SetBranchAddress("stub_layers",         input1_hit_layer);//layer du stub
  PATT1->SetBranchAddress("stub_ladders",        input1_hit_ladder);//ladder du stub
  PATT1->SetBranchAddress("stub_module",         input1_hit_zPos);//position en Z du module du stub
  PATT1->SetBranchAddress("stub_segment",        input1_hit_segment);//segment du stub
  PATT1->SetBranchAddress("stub_strip",          input1_hit_strip);//numero de strip du stub
  PATT1->SetBranchAddress("stub_tp",             input1_hit_tp);//numero de la particule du stub
  PATT1->SetBranchAddress("stub_idx",             input1_hit_idx);//index du stub dans l'evenement
  PATT1->SetBranchAddress("stub_ptGEN",          input1_hit_ptGEN);//PT de la particule du stub
  PATT1->SetBranchAddress("stub_etaGEN",         input1_hit_etaGEN);//ETA de la particule du stub
  PATT1->SetBranchAddress("stub_phi0GEN",        input1_hit_phi0GEN);//PHI0 de la particule du stub
  PATT1->SetBranchAddress("stub_IP",             input1_hit_ip);//distance avec l'IP
  PATT1->SetBranchAddress("stub_x",              input1_hit_x);
  PATT1->SetBranchAddress("stub_y",              input1_hit_y);
  PATT1->SetBranchAddress("stub_z",              input1_hit_z);
  PATT1->SetBranchAddress("stub_X0",             input1_hit_X0);
  PATT1->SetBranchAddress("stub_Y0",             input1_hit_Y0);
  PATT1->SetBranchAddress("stub_Z0",             input1_hit_Z0);

  /*********************************************/

  /*********************INPUT 2 FILE *************************************/


  TChain *PATT2    = new TChain("Patterns"); // infos about patterns
  TChain *SEC2    = new TChain("Sectors");   //infos about sectors
  PATT2->Add(inputFile2.c_str());
  SEC2->Add(inputFile2.c_str());

  int input2_nb_layers;
  int input2_nb_patterns=0;
  int input2_ori_nb_stubs=0;
  int input2_sel_nb_stubs=0;
  int input2_event_id;
  int *input2_superStrip_layer_0 = new int[MAX_NB_PATTERNS];
  int *input2_superStrip_layer_1 = new int[MAX_NB_PATTERNS];
  int *input2_superStrip_layer_2 = new int[MAX_NB_PATTERNS];
  int *input2_superStrip_layer_3 = new int[MAX_NB_PATTERNS];
  int *input2_superStrip_layer_4 = new int[MAX_NB_PATTERNS];
  int *input2_superStrip_layer_5 = new int[MAX_NB_PATTERNS];
  int *input2_superStrip_layer_6 = new int[MAX_NB_PATTERNS];
  int *input2_superStrip_layer_7 = new int[MAX_NB_PATTERNS];
  int *input2_pattern_sector_id = new int[MAX_NB_PATTERNS];

  //Array containing the strips arrays
  int* input2_superStrips[MAX_NB_LAYERS];
  input2_superStrips[0]=input2_superStrip_layer_0;
  input2_superStrips[1]=input2_superStrip_layer_1;
  input2_superStrips[2]=input2_superStrip_layer_2;
  input2_superStrips[3]=input2_superStrip_layer_3;
  input2_superStrips[4]=input2_superStrip_layer_4;
  input2_superStrips[5]=input2_superStrip_layer_5;
  input2_superStrips[6]=input2_superStrip_layer_6;
  input2_superStrips[7]=input2_superStrip_layer_7;

  int input2_sector_id=0;
  int input2_sector_layers=0;
  int input2_nb_ladders_layer[MAX_NB_LAYERS];
  int input2_sector_layer_list[MAX_NB_LAYERS];
  int input2_sector_layer_0[MAX_NB_LADDERS_PER_LAYER];
  int input2_sector_layer_1[MAX_NB_LADDERS_PER_LAYER];
  int input2_sector_layer_2[MAX_NB_LADDERS_PER_LAYER];
  int input2_sector_layer_3[MAX_NB_LADDERS_PER_LAYER];
  int input2_sector_layer_4[MAX_NB_LADDERS_PER_LAYER];
  int input2_sector_layer_5[MAX_NB_LADDERS_PER_LAYER];
  int input2_sector_layer_6[MAX_NB_LADDERS_PER_LAYER];
  int input2_sector_layer_7[MAX_NB_LADDERS_PER_LAYER];

  int* input2_sector_layers_detail[MAX_NB_LAYERS];
  input2_sector_layers_detail[0]=input2_sector_layer_0;
  input2_sector_layers_detail[1]=input2_sector_layer_1;
  input2_sector_layers_detail[2]=input2_sector_layer_2;
  input2_sector_layers_detail[3]=input2_sector_layer_3;
  input2_sector_layers_detail[4]=input2_sector_layer_4;
  input2_sector_layers_detail[5]=input2_sector_layer_5;
  input2_sector_layers_detail[6]=input2_sector_layer_6;
  input2_sector_layers_detail[7]=input2_sector_layer_7;

  int *input2_nbHitPerPattern = new int[MAX_NB_PATTERNS];
  int input2_totalNbHits=0;
  int input2_nbTracks = 0;
  float *input2_track_pt = new float[MAX_NB_PATTERNS];
  float *input2_track_phi = new float[MAX_NB_PATTERNS];
  float *input2_track_d0 = new float[MAX_NB_PATTERNS];
  float *input2_track_eta = new float[MAX_NB_PATTERNS];
  float *input2_track_z0 = new float[MAX_NB_PATTERNS];
  short *input2_hit_layer = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  short *input2_hit_ladder = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  short *input2_hit_zPos = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  short *input2_hit_segment = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  short *input2_hit_strip = new short[MAX_NB_PATTERNS*MAX_NB_HITS];
  int *input2_hit_tp = new int[MAX_NB_PATTERNS*MAX_NB_HITS];
  int *input2_hit_idx = new int[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_ptGEN = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_etaGEN = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_phi0GEN = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_ip = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_x = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_y = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_z = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_X0 = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_Y0 = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  float *input2_hit_Z0 = new float[MAX_NB_PATTERNS*MAX_NB_HITS];
  
  SEC2->SetBranchAddress("sectorID",            &input2_sector_id);//ID du secteur
  SEC2->SetBranchAddress("nbLayers",            &input2_sector_layers);// nombre de layers dans le secteur
  SEC2->SetBranchAddress("layer",                  input2_sector_layer_list);//layers ID
  SEC2->SetBranchAddress("nb_ladders_layer",       input2_nb_ladders_layer); // nombre de ladders pour chaque layer
  SEC2->SetBranchAddress("sectorLadders_layer_0",  input2_sector_layer_0);//liste des ladders pour layer 0
  SEC2->SetBranchAddress("sectorLadders_layer_1",  input2_sector_layer_1);
  SEC2->SetBranchAddress("sectorLadders_layer_2",  input2_sector_layer_2);
  SEC2->SetBranchAddress("sectorLadders_layer_3",  input2_sector_layer_3);
  SEC2->SetBranchAddress("sectorLadders_layer_4",  input2_sector_layer_4);
  SEC2->SetBranchAddress("sectorLadders_layer_5",  input2_sector_layer_5);
  
  PATT2->SetBranchAddress("nbLayers",            &input2_nb_layers);//nombre de layers pour les patterns
  PATT2->SetBranchAddress("nbPatterns",          &input2_nb_patterns); // nombre de patterns dans l'evenement
  PATT1->SetBranchAddress("nbStubsInEvt",        &input2_ori_nb_stubs); // nombre de stubs dans l'evenement initial
  PATT1->SetBranchAddress("nbStubsInPat",        &input2_sel_nb_stubs); // nombre de stubs uniques dans les patterns
  PATT2->SetBranchAddress("eventID",             &input2_event_id); // ID de l'evenement (le meme que dans le fichier de simulation)
  PATT2->SetBranchAddress("sectorID",            input2_pattern_sector_id);// ID du secteur du pattern (permet de retrouver le secteur dans le premier TTree)
  PATT2->SetBranchAddress("superStrip0",         input2_superStrip_layer_0);// tableau de superstrips pour le layer 0
  PATT2->SetBranchAddress("superStrip1",         input2_superStrip_layer_1);
  PATT2->SetBranchAddress("superStrip2",         input2_superStrip_layer_2);
  PATT2->SetBranchAddress("superStrip3",         input2_superStrip_layer_3);
  PATT2->SetBranchAddress("superStrip4",         input2_superStrip_layer_4);
  PATT2->SetBranchAddress("superStrip5",         input2_superStrip_layer_5);
  PATT2->SetBranchAddress("total_nb_stubs",      &input2_totalNbHits);
  PATT2->SetBranchAddress("nbTracks",            &input2_nbTracks);
  PATT2->SetBranchAddress("nbStubs",             input2_nbHitPerPattern); // nombre de stubs contenus dans chaque pattern

  PATT1->SetBranchAddress("track_pt",            input2_track_pt); // PT of fitted tracks
  PATT1->SetBranchAddress("track_phi",           input2_track_phi); // PHI0 of fitted tracks
  PATT1->SetBranchAddress("track_d0",            input2_track_d0); // D0 of fitted tracks
  PATT1->SetBranchAddress("track_eta",           input2_track_eta); // ETA of fitted tracks
  PATT1->SetBranchAddress("track_z0",            input2_track_z0); // Z0 of fitted tracks

  PATT2->SetBranchAddress("stub_layers",         input2_hit_layer);//layer du stub
  PATT2->SetBranchAddress("stub_ladders",        input2_hit_ladder);//ladder du stub
  PATT2->SetBranchAddress("stub_module",         input2_hit_zPos);//position en Z du module du stub
  PATT2->SetBranchAddress("stub_segment",        input2_hit_segment);//segment du stub
  PATT2->SetBranchAddress("stub_strip",          input2_hit_strip);//numero de strip du stub
  PATT2->SetBranchAddress("stub_tp",             input2_hit_tp);//numero de la particule du stub
  PATT2->SetBranchAddress("stub_idx",             input2_hit_idx);//index du stub dans l'evenement
  PATT2->SetBranchAddress("stub_ptGEN",          input2_hit_ptGEN);//PT de la particule du stub
  PATT2->SetBranchAddress("stub_etaGEN",         input2_hit_etaGEN);//PT de la particule du stub
  PATT2->SetBranchAddress("stub_phi0GEN",        input2_hit_phi0GEN);//PT de la particule du stub
  PATT2->SetBranchAddress("stub_IP",             input2_hit_ip);//distance avec l'IP
  PATT2->SetBranchAddress("stub_x",              input2_hit_x);
  PATT2->SetBranchAddress("stub_y",              input2_hit_y);
  PATT2->SetBranchAddress("stub_z",              input2_hit_z);
  PATT2->SetBranchAddress("stub_X0",             input2_hit_X0);
  PATT2->SetBranchAddress("stub_Y0",             input2_hit_Y0);
  PATT2->SetBranchAddress("stub_Z0",             input2_hit_Z0);

  /*********************OUTPUT FILE *************************************/
  TTree *PATTOUT    = new TTree("Patterns", "Active patterns");
  TTree *SECOUT     = new TTree("Sectors", "Used Sectors");
  TFile *t = new TFile(outputFile.c_str(),"recreate");

  const int MAX_NB_OUTPUT_PATTERNS = 100000;

  int nb_layers;
  int nb_patterns=0;
  int ori_nb_stubs=0;
  int sel_nb_stubs=0;
  int event_id;
  int *superStrip_layer_0 = new int[MAX_NB_OUTPUT_PATTERNS];
  int *superStrip_layer_1 = new int[MAX_NB_OUTPUT_PATTERNS];
  int *superStrip_layer_2 = new int[MAX_NB_OUTPUT_PATTERNS];
  int *superStrip_layer_3 = new int[MAX_NB_OUTPUT_PATTERNS];
  int *superStrip_layer_4 = new int[MAX_NB_OUTPUT_PATTERNS];
  int *superStrip_layer_5 = new int[MAX_NB_OUTPUT_PATTERNS];
  int *superStrip_layer_6 = new int[MAX_NB_OUTPUT_PATTERNS];
  int *superStrip_layer_7 = new int[MAX_NB_OUTPUT_PATTERNS];
  int *pattern_sector_id = new int[MAX_NB_OUTPUT_PATTERNS];

  //Array containing the strips arrays
  int* superStrips[MAX_NB_LAYERS];
  superStrips[0]=superStrip_layer_0;
  superStrips[1]=superStrip_layer_1;
  superStrips[2]=superStrip_layer_2;
  superStrips[3]=superStrip_layer_3;
  superStrips[4]=superStrip_layer_4;
  superStrips[5]=superStrip_layer_5;
  superStrips[6]=superStrip_layer_6;
  superStrips[7]=superStrip_layer_7;

  int sector_id=0;
  int sector_layers=0;
  int nb_ladders_layer[MAX_NB_LAYERS];
  int sector_layer_list[MAX_NB_LAYERS];
  int sector_layer_0[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_1[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_2[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_3[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_4[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_5[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_6[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_7[MAX_NB_LADDERS_PER_LAYER];

  int* sector_layers_detail[MAX_NB_LAYERS];
  sector_layers_detail[0]=sector_layer_0;
  sector_layers_detail[1]=sector_layer_1;
  sector_layers_detail[2]=sector_layer_2;
  sector_layers_detail[3]=sector_layer_3;
  sector_layers_detail[4]=sector_layer_4;
  sector_layers_detail[5]=sector_layer_5;
  sector_layers_detail[6]=sector_layer_6;
  sector_layers_detail[7]=sector_layer_7;

  int *nbHitPerPattern = new int[MAX_NB_OUTPUT_PATTERNS];
  int totalNbHits=0;
  int nbTracks = 0;

  float *track_pt = new float[MAX_NB_OUTPUT_PATTERNS];
  float *track_phi = new float[MAX_NB_OUTPUT_PATTERNS];
  float *track_d0 = new float[MAX_NB_OUTPUT_PATTERNS];
  float *track_eta = new float[MAX_NB_OUTPUT_PATTERNS];
  float *track_z0 = new float[MAX_NB_OUTPUT_PATTERNS];

  short *hit_layer = new  short[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  short *hit_ladder = new short[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  short *hit_zPos = new short[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  short *hit_segment = new  short[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  short *hit_strip = new  short[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  int *hit_tp = new int[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  int *hit_idx = new int[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_ptGEN = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_etaGEN = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_phi0GEN = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_ip = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_x = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_y = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_z = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_X0 = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_Y0 = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  float *hit_Z0 = new float[MAX_NB_OUTPUT_PATTERNS*MAX_NB_HITS];
  
  SECOUT->Branch("sectorID",            &sector_id);
  SECOUT->Branch("nbLayers",            &sector_layers);
  SECOUT->Branch("layer",                  sector_layer_list, "layer[nbLayers]/I");
  SECOUT->Branch("nb_ladders_layer",       nb_ladders_layer, "nb_ladders_layer[nbLayers]/I");
  SECOUT->Branch("sectorLadders_layer_0",  sector_layer_0, "sectorLadders_layer_0[16]/I");
  SECOUT->Branch("sectorLadders_layer_1",  sector_layer_1, "sectorLadders_layer_1[16]/I");
  SECOUT->Branch("sectorLadders_layer_2",  sector_layer_2, "sectorLadders_layer_2[16]/I");
  SECOUT->Branch("sectorLadders_layer_3",  sector_layer_3, "sectorLadders_layer_3[16]/I");
  SECOUT->Branch("sectorLadders_layer_4",  sector_layer_4, "sectorLadders_layer_4[16]/I");
  SECOUT->Branch("sectorLadders_layer_5",  sector_layer_5, "sectorLadders_layer_5[16]/I");
  SECOUT->Branch("sectorLadders_layer_6",  sector_layer_6, "sectorLadders_layer_6[16]/I");
  SECOUT->Branch("sectorLadders_layer_7",  sector_layer_7, "sectorLadders_layer_7[16]/I");


  PATTOUT->Branch("nbLayers",            &nb_layers);
  PATTOUT->Branch("nbPatterns",          &nb_patterns);
  PATTOUT->Branch("nbStubsInEvt",        &ori_nb_stubs);
  PATTOUT->Branch("nbStubsInPat",        &sel_nb_stubs);
  PATTOUT->Branch("eventID",             &event_id);
  PATTOUT->Branch("sectorID",            pattern_sector_id, "sectorID[nbPatterns]/I");
  PATTOUT->Branch("superStrip0",         superStrip_layer_0, "superStrip0[nbPatterns]/I");
  PATTOUT->Branch("superStrip1",         superStrip_layer_1, "superStrip1[nbPatterns]/I");
  PATTOUT->Branch("superStrip2",         superStrip_layer_2, "superStrip2[nbPatterns]/I");
  PATTOUT->Branch("superStrip3",         superStrip_layer_3, "superStrip3[nbPatterns]/I");
  PATTOUT->Branch("superStrip4",         superStrip_layer_4, "superStrip4[nbPatterns]/I");
  PATTOUT->Branch("superStrip5",         superStrip_layer_5, "superStrip5[nbPatterns]/I");
  PATTOUT->Branch("superStrip6",         superStrip_layer_6, "superStrip6[nbPatterns]/I");
  PATTOUT->Branch("superStrip7",         superStrip_layer_7, "superStrip7[nbPatterns]/I");
  PATTOUT->Branch("nbStubs",             nbHitPerPattern, "nbStubs[nbPatterns]/I");
  PATTOUT->Branch("total_nb_stubs",      &totalNbHits, "total_nb_stubs/I");
  PATTOUT->Branch("nbTracks",            &nbTracks, "nbTracks/I");
  PATTOUT->Branch("track_pt",            track_pt, "track_pt[nbPatterns]/F");
  PATTOUT->Branch("track_phi",           track_phi, "track_phi[nbPatterns]/F");
  PATTOUT->Branch("track_d0",            track_d0, "track_d0[nbPatterns]/F");
  PATTOUT->Branch("track_eta",           track_eta, "track_eta[nbPatterns]/F");
  PATTOUT->Branch("track_z0",            track_z0, "track_z0[nbPatterns]/F");
  PATTOUT->Branch("stub_layers",         hit_layer,"stub_layers[total_nb_stubs]/S");
  PATTOUT->Branch("stub_ladders",        hit_ladder, "stub_ladders[total_nb_stubs]/S");
  PATTOUT->Branch("stub_module",         hit_zPos, "stub_module[total_nb_stubs]/S");
  PATTOUT->Branch("stub_segment",        hit_segment, "stub_segment[total_nb_stubs]/S");
  PATTOUT->Branch("stub_strip",          hit_strip, "stub_strip[total_nb_stubs]/S");
  PATTOUT->Branch("stub_tp",             hit_tp,    "stub_tp[total_nb_stubs]/I");
  PATTOUT->Branch("stub_idx",            hit_idx,    "stub_idx[total_nb_stubs]/I");
  PATTOUT->Branch("stub_ptGEN",          hit_ptGEN, "stub_ptGEN[total_nb_stubs]/F");
  PATTOUT->Branch("stub_etaGEN",         hit_etaGEN, "stub_etaGEN[total_nb_stubs]/F");
  PATTOUT->Branch("stub_phi0GEN",        hit_phi0GEN, "stub_phi0GEN[total_nb_stubs]/F");
  PATTOUT->Branch("stub_IP",             hit_ip, "stub_IP[total_nb_stubs]/F");
  PATTOUT->Branch("stub_x",              hit_x, "stub_x[total_nb_stubs]/F");
  PATTOUT->Branch("stub_y",              hit_y, "stub_y[total_nb_stubs]/F");
  PATTOUT->Branch("stub_z",              hit_z, "stub_z[total_nb_stubs]/F");
  PATTOUT->Branch("stub_X0",             hit_X0, "stub_X0[total_nb_stubs]/F");
  PATTOUT->Branch("stub_Y0",             hit_Y0, "stub_Y0[total_nb_stubs]/F");
  PATTOUT->Branch("stub_Z0",             hit_Z0, "stub_Z0[total_nb_stubs]/F");
  

  /*********************************************/

  /******************** MERGING SECTOR DATA************************/
  vector<int> sectors;
  //copy all the content of the first file
  int nb_entries1 = SEC1->GetEntries();
  for(int i=0;i<nb_entries1;i++){
    SEC1->GetEntry(i);
    sector_id = input1_sector_id;
    sectors.push_back(sector_id);
    sector_layers=input1_sector_layers;
    memcpy(sector_layer_list, input1_sector_layer_list, sector_layers*sizeof(int));
    memcpy(nb_ladders_layer, input1_nb_ladders_layer, sector_layers*sizeof(int));
    for(int j=0;j<sector_layers;j++){
      memcpy(sector_layers_detail[j],input1_sector_layers_detail[j],input1_nb_ladders_layer[j]*sizeof(int));
    }
    SECOUT->Fill();
  }

  //copy the content of the second file (only the unknown sectors)
  int nb_entries2 = SEC2->GetEntries();
  for(int i=0;i<nb_entries2;i++){
    SEC2->GetEntry(i);
    if(std::find(sectors.begin(),sectors.end(), input2_sector_id)==sectors.end()){ // We do not know this sector
      sector_id = input2_sector_id;
      sectors.push_back(sector_id);
      sector_layers=input2_sector_layers;
      memcpy(sector_layer_list, input2_sector_layer_list, sector_layers*sizeof(int));
      memcpy(nb_ladders_layer, input2_nb_ladders_layer, sector_layers*sizeof(int));
      for(int j=0;j<sector_layers;j++){
	memcpy(sector_layers_detail[j],input2_sector_layers_detail[j],input2_nb_ladders_layer[j]*sizeof(int));
      }
      SECOUT->Fill();
    }
  }
  SECOUT->Write();

  /******************** MERGING PATTERN DATA************************/
  //Loop on the events of the first file
  nb_entries1 = PATT1->GetEntries();
  nb_entries2 = PATT2->GetEntries();

  if(nb_entries1!=nb_entries2){
    cout<<"The 2 files do not have the same number of events -> CANCELED"<<endl;
    return;
  }
  
  for(int i=0;i<nb_entries1;i++){
    PATT1->GetEntry(i);
    PATT2->GetEntry(i); 

    event_id = input1_event_id;
    nb_layers = input1_nb_layers;

    if(input2_event_id!=event_id){
      cout<<"Cannot find event "<<event_id<<" in file "<<inputFile2<<" -> DROP EVENT"<<endl;
      break;
    }

    memset(pattern_sector_id,0,MAX_NB_OUTPUT_PATTERNS*sizeof(int));

    nb_patterns = input1_nb_patterns+input2_nb_patterns;
    ori_nb_stubs = input1_ori_nb_stubs+input2_ori_nb_stubs;
    sel_nb_stubs = input1_sel_nb_stubs+input2_sel_nb_stubs;

    memcpy(pattern_sector_id,input1_pattern_sector_id,input1_nb_patterns*sizeof(int));
    memcpy(pattern_sector_id+input1_nb_patterns,input2_pattern_sector_id,input2_nb_patterns*sizeof(int));
    
    for(int j=0;j<nb_layers;j++){
      memcpy(superStrips[j],input1_superStrips[j],input1_nb_patterns*sizeof(int));
      memcpy(superStrips[j]+input1_nb_patterns,input2_superStrips[j],input2_nb_patterns*sizeof(int));
    }
    
    memcpy(nbHitPerPattern,input1_nbHitPerPattern,input1_nb_patterns*sizeof(int));
    memcpy(nbHitPerPattern+input1_nb_patterns,input2_nbHitPerPattern,input2_nb_patterns*sizeof(int));

    totalNbHits=input1_totalNbHits+input2_totalNbHits;
    
    nbTracks=input1_nbTracks+input2_nbTracks;

    memcpy(track_pt,input1_track_pt,input1_nbTracks*sizeof(float));
    memcpy(track_pt+input1_nbTracks,input2_track_pt,input2_nbTracks*sizeof(float));

    memcpy(track_phi,input1_track_phi,input1_nbTracks*sizeof(float));
    memcpy(track_phi+input1_nbTracks,input2_track_phi,input2_nbTracks*sizeof(float));

    memcpy(track_d0,input1_track_d0,input1_nbTracks*sizeof(float));
    memcpy(track_d0+input1_nbTracks,input2_track_d0,input2_nbTracks*sizeof(float));

    memcpy(track_eta,input1_track_eta,input1_nbTracks*sizeof(float));
    memcpy(track_eta+input1_nbTracks,input2_track_eta,input2_nbTracks*sizeof(float));

    memcpy(track_z0,input1_track_z0,input1_nbTracks*sizeof(float));
    memcpy(track_z0+input1_nbTracks,input2_track_z0,input2_nbTracks*sizeof(float));

    memcpy(hit_layer,input1_hit_layer,input1_totalNbHits*sizeof(short));
    memcpy(hit_layer+input1_totalNbHits,input2_hit_layer,input2_totalNbHits*sizeof(short));

    memcpy(hit_ladder,input1_hit_ladder,input1_totalNbHits*sizeof(short));
    memcpy(hit_ladder+input1_totalNbHits,input2_hit_ladder,input2_totalNbHits*sizeof(short));

    memcpy(hit_zPos,input1_hit_zPos,input1_totalNbHits*sizeof(short));
    memcpy(hit_zPos+input1_totalNbHits,input2_hit_zPos,input2_totalNbHits*sizeof(short));

    memcpy(hit_segment,input1_hit_segment,input1_totalNbHits*sizeof(short));
    memcpy(hit_segment+input1_totalNbHits,input2_hit_segment,input2_totalNbHits*sizeof(short));

    memcpy(hit_strip,input1_hit_strip,input1_totalNbHits*sizeof(short));
    memcpy(hit_strip+input1_totalNbHits,input2_hit_strip,input2_totalNbHits*sizeof(short));

    memcpy(hit_tp,input1_hit_tp,input1_totalNbHits*sizeof(int));
    memcpy(hit_tp+input1_totalNbHits,input2_hit_tp,input2_totalNbHits*sizeof(int));

    memcpy(hit_idx,input1_hit_idx,input1_totalNbHits*sizeof(int));
    memcpy(hit_idx+input1_totalNbHits,input2_hit_idx,input2_totalNbHits*sizeof(int));

    memcpy(hit_ptGEN,input1_hit_ptGEN,input1_totalNbHits*sizeof(float));
    memcpy(hit_ptGEN+input1_totalNbHits,input2_hit_ptGEN,input2_totalNbHits*sizeof(float));

    memcpy(hit_etaGEN,input1_hit_etaGEN,input1_totalNbHits*sizeof(float));
    memcpy(hit_etaGEN+input1_totalNbHits,input2_hit_etaGEN,input2_totalNbHits*sizeof(float));

    memcpy(hit_phi0GEN,input1_hit_phi0GEN,input1_totalNbHits*sizeof(float));
    memcpy(hit_phi0GEN+input1_totalNbHits,input2_hit_phi0GEN,input2_totalNbHits*sizeof(float));

    memcpy(hit_ip,input1_hit_ip,input1_totalNbHits*sizeof(float));
    memcpy(hit_ip+input1_totalNbHits,input2_hit_ip,input2_totalNbHits*sizeof(float));

    memcpy(hit_x,input1_hit_x,input1_totalNbHits*sizeof(float));
    memcpy(hit_x+input1_totalNbHits,input2_hit_x,input2_totalNbHits*sizeof(float));

    memcpy(hit_y,input1_hit_y,input1_totalNbHits*sizeof(float));
    memcpy(hit_y+input1_totalNbHits,input2_hit_y,input2_totalNbHits*sizeof(float));

    memcpy(hit_z,input1_hit_z,input1_totalNbHits*sizeof(float));
    memcpy(hit_z+input1_totalNbHits,input2_hit_z,input2_totalNbHits*sizeof(float));

    memcpy(hit_X0,input1_hit_X0,input1_totalNbHits*sizeof(float));
    memcpy(hit_X0+input1_totalNbHits,input2_hit_X0,input2_totalNbHits*sizeof(float));

    memcpy(hit_Y0,input1_hit_Y0,input1_totalNbHits*sizeof(float));
    memcpy(hit_Y0+input1_totalNbHits,input2_hit_Y0,input2_totalNbHits*sizeof(float));

    memcpy(hit_Z0,input1_hit_Z0,input1_totalNbHits*sizeof(float));
    memcpy(hit_Z0+input1_totalNbHits,input2_hit_Z0,input2_totalNbHits*sizeof(float));

    PATTOUT->Fill();
  }

  PATTOUT->Write();

  t->Close();

  delete [] input1_superStrip_layer_0;
  delete [] input1_superStrip_layer_1;
  delete [] input1_superStrip_layer_2;
  delete [] input1_superStrip_layer_3;
  delete [] input1_superStrip_layer_4;
  delete [] input1_superStrip_layer_5;
  delete [] input1_superStrip_layer_6;
  delete [] input1_superStrip_layer_7;
  delete [] input1_pattern_sector_id;

  delete [] input1_nbHitPerPattern;
  delete [] input1_track_pt;
  delete [] input1_track_phi;
  delete [] input1_track_d0;
  delete [] input1_track_eta;
  delete [] input1_track_z0;
  delete [] input1_hit_layer;
  delete [] input1_hit_ladder;
  delete [] input1_hit_zPos;
  delete [] input1_hit_segment;
  delete [] input1_hit_strip;
  delete [] input1_hit_tp;
  delete [] input1_hit_idx;
  delete [] input1_hit_ptGEN;
  delete [] input1_hit_etaGEN;
  delete [] input1_hit_phi0GEN;
  delete [] input1_hit_ip;
  delete [] input1_hit_x;
  delete [] input1_hit_y;
  delete [] input1_hit_z;
  delete [] input1_hit_X0;
  delete [] input1_hit_Y0;
  delete [] input1_hit_Z0;
  
  delete [] input2_superStrip_layer_0;
  delete [] input2_superStrip_layer_1;
  delete [] input2_superStrip_layer_2;
  delete [] input2_superStrip_layer_3;
  delete [] input2_superStrip_layer_4;
  delete [] input2_superStrip_layer_5;
  delete [] input2_superStrip_layer_6;
  delete [] input2_superStrip_layer_7;
  delete [] input2_pattern_sector_id;

  delete [] input2_nbHitPerPattern;
  delete [] input2_track_pt;
  delete [] input2_track_phi;
  delete [] input2_track_d0;
  delete [] input2_track_eta;
  delete [] input2_track_z0;
  delete [] input2_hit_layer;
  delete [] input2_hit_ladder;
  delete [] input2_hit_zPos;
  delete [] input2_hit_segment;
  delete [] input2_hit_strip;
  delete [] input2_hit_tp;
  delete [] input2_hit_idx;
  delete [] input2_hit_ptGEN;
  delete [] input2_hit_etaGEN;
  delete [] input2_hit_phi0GEN;
  delete [] input2_hit_ip;
  delete [] input2_hit_x;
  delete [] input2_hit_y;
  delete [] input2_hit_z;
  delete [] input2_hit_X0;
  delete [] input2_hit_Y0;
  delete [] input2_hit_Z0;

  delete [] superStrip_layer_0;
  delete [] superStrip_layer_1;
  delete [] superStrip_layer_2;
  delete [] superStrip_layer_3;
  delete [] superStrip_layer_4;
  delete [] superStrip_layer_5;
  delete [] superStrip_layer_6;
  delete [] superStrip_layer_7;
  delete [] pattern_sector_id;
  delete [] track_pt;
  delete [] track_phi;
  delete [] track_d0;
  delete [] track_eta;
  delete [] track_z0;
  delete [] nbHitPerPattern;
  delete [] hit_layer;
  delete [] hit_ladder;
  delete [] hit_zPos;
  delete [] hit_segment;
  delete [] hit_strip;
  delete [] hit_tp;
  delete [] hit_idx;
  delete [] hit_ptGEN;
  delete [] hit_etaGEN;
  delete [] hit_phi0GEN;
  delete [] hit_ip;
  delete [] hit_x;
  delete [] hit_y;
  delete [] hit_z;
  delete [] hit_X0;
  delete [] hit_Y0;
  delete [] hit_Z0;
 
  delete PATT1;
  delete PATT2;
  delete SEC1;
  delete SEC2;
  delete PATTOUT;
  delete SECOUT;

}

void PatternFinder::find(int start, int& stop){
  /**************** OUTPUT FILE ****************/
  TTree* Out = new TTree("Patterns", "Active patterns");
  TTree* SectorOut = new TTree("Sectors", "Used Sectors");
  TFile *t = new TFile(outputFileName.c_str(),"recreate");

  const int MAX_NB_PATTERNS = 1500;
  const int MAX_NB_HITS = 100;
  const int MAX_NB_LADDERS_PER_LAYER = 16;
  const int MAX_NB_LAYERS = 8;

  int nb_layers;
  int nb_patterns=0;
  int ori_nb_stubs=0;
  int sel_nb_stubs=0;
  int nb_tracks=0;
  int event_id;
  int superStrip_layer_0[MAX_NB_PATTERNS];
  int superStrip_layer_1[MAX_NB_PATTERNS];
  int superStrip_layer_2[MAX_NB_PATTERNS];
  int superStrip_layer_3[MAX_NB_PATTERNS];
  int superStrip_layer_4[MAX_NB_PATTERNS];
  int superStrip_layer_5[MAX_NB_PATTERNS];
  int superStrip_layer_6[MAX_NB_PATTERNS];
  int superStrip_layer_7[MAX_NB_PATTERNS];
  int pattern_sector_id[MAX_NB_PATTERNS];

  //Array containing the strips arrays
  int* superStrips[MAX_NB_LAYERS];
  superStrips[0]=superStrip_layer_0;
  superStrips[1]=superStrip_layer_1;
  superStrips[2]=superStrip_layer_2;
  superStrips[3]=superStrip_layer_3;
  superStrips[4]=superStrip_layer_4;
  superStrips[5]=superStrip_layer_5;
  superStrips[6]=superStrip_layer_6;
  superStrips[7]=superStrip_layer_7;

  int sector_id=0;
  int sector_layers=0;
  int nb_ladders_layer[MAX_NB_LAYERS];
  int sector_layer_list[MAX_NB_LAYERS];
  int sector_layer_0[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_1[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_2[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_3[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_4[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_5[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_6[MAX_NB_LADDERS_PER_LAYER];
  int sector_layer_7[MAX_NB_LADDERS_PER_LAYER];

  int* sector_layers_detail[MAX_NB_LAYERS];
  sector_layers_detail[0]=sector_layer_0;
  sector_layers_detail[1]=sector_layer_1;
  sector_layers_detail[2]=sector_layer_2;
  sector_layers_detail[3]=sector_layer_3;
  sector_layers_detail[4]=sector_layer_4;
  sector_layers_detail[5]=sector_layer_5;
  sector_layers_detail[6]=sector_layer_6;
  sector_layers_detail[7]=sector_layer_7;

  int nbHitPerPattern[MAX_NB_PATTERNS];

  float track_pt[MAX_NB_PATTERNS];
  float track_phi[MAX_NB_PATTERNS];
  float track_d0[MAX_NB_PATTERNS];
  float track_eta[MAX_NB_PATTERNS];
  float track_z0[MAX_NB_PATTERNS];

  int totalNbHits=0;
  short hit_layer[MAX_NB_PATTERNS*MAX_NB_HITS];
  short hit_ladder[MAX_NB_PATTERNS*MAX_NB_HITS];
  short hit_zPos[MAX_NB_PATTERNS*MAX_NB_HITS];
  short hit_segment[MAX_NB_PATTERNS*MAX_NB_HITS];
  short hit_strip[MAX_NB_PATTERNS*MAX_NB_HITS];
  int hit_tp[MAX_NB_PATTERNS*MAX_NB_HITS];
  int hit_idx[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_ptGEN[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_etaGEN[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_phi0GEN[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_ip[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_x[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_y[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_z[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_x0[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_y0[MAX_NB_PATTERNS*MAX_NB_HITS];
  float hit_z0[MAX_NB_PATTERNS*MAX_NB_HITS];
  
  SectorOut->Branch("sectorID",            &sector_id);
  SectorOut->Branch("nbLayers",            &sector_layers);
  SectorOut->Branch("layer",                  sector_layer_list, "layer[nbLayers]/I");
  SectorOut->Branch("nb_ladders_layer",       nb_ladders_layer, "nb_ladders_layer[nbLayers]/I");
  SectorOut->Branch("sectorLadders_layer_0",  sector_layer_0, "sectorLadders_layer_0[16]/I");
  SectorOut->Branch("sectorLadders_layer_1",  sector_layer_1, "sectorLadders_layer_1[16]/I");
  SectorOut->Branch("sectorLadders_layer_2",  sector_layer_2, "sectorLadders_layer_2[16]/I");
  SectorOut->Branch("sectorLadders_layer_3",  sector_layer_3, "sectorLadders_layer_3[16]/I");
  SectorOut->Branch("sectorLadders_layer_4",  sector_layer_4, "sectorLadders_layer_4[16]/I");
  SectorOut->Branch("sectorLadders_layer_5",  sector_layer_5, "sectorLadders_layer_5[16]/I");
  SectorOut->Branch("sectorLadders_layer_6",  sector_layer_6, "sectorLadders_layer_6[16]/I");
  SectorOut->Branch("sectorLadders_layer_7",  sector_layer_7, "sectorLadders_layer_7[16]/I");

  Out->Branch("nbLayers",            &nb_layers);
  Out->Branch("nbPatterns",          &nb_patterns);
  Out->Branch("nbStubsInEvt",        &ori_nb_stubs);
  Out->Branch("nbStubsInPat",        &sel_nb_stubs);

  Out->Branch("nbTracks",            &nb_tracks);

  Out->Branch("eventID",             &event_id);
  Out->Branch("sectorID",            pattern_sector_id, "sectorID[nbPatterns]/I");
  Out->Branch("superStrip0",         superStrip_layer_0, "superStrip0[nbPatterns]/I");
  Out->Branch("superStrip1",         superStrip_layer_1, "superStrip1[nbPatterns]/I");
  Out->Branch("superStrip2",         superStrip_layer_2, "superStrip2[nbPatterns]/I");
  Out->Branch("superStrip3",         superStrip_layer_3, "superStrip3[nbPatterns]/I");
  Out->Branch("superStrip4",         superStrip_layer_4, "superStrip4[nbPatterns]/I");
  Out->Branch("superStrip5",         superStrip_layer_5, "superStrip5[nbPatterns]/I");
  Out->Branch("superStrip6",         superStrip_layer_6, "superStrip6[nbPatterns]/I");
  Out->Branch("superStrip7",         superStrip_layer_7, "superStrip7[nbPatterns]/I");
  Out->Branch("nbStubs",             nbHitPerPattern, "nbStubs[nbPatterns]/I");

  Out->Branch("track_pt",             track_pt, "track_pt[nbTracks]/F");
  Out->Branch("track_phi",            track_phi, "track_phi[nbTracks]/F");
  Out->Branch("track_d0",             track_d0, "track_d0[nbTracks]/F");
  Out->Branch("track_eta",            track_eta, "track_eta[nbTracks]/F");
  Out->Branch("track_z0",             track_z0, "track_z0[nbTracks]/F");

  Out->Branch("total_nb_stubs",      &totalNbHits, "total_nb_stubs/I");
  Out->Branch("stub_layers",         hit_layer,"stub_layers[total_nb_stubs]/S");
  Out->Branch("stub_ladders",        hit_ladder, "stub_ladders[total_nb_stubs]/S");
  Out->Branch("stub_module",         hit_zPos, "stub_module[total_nb_stubs]/S");
  Out->Branch("stub_segment",        hit_segment, "stub_segment[total_nb_stubs]/S");
  Out->Branch("stub_strip",          hit_strip, "stub_strip[total_nb_stubs]/S");
  Out->Branch("stub_tp",             hit_tp,    "stub_tp[total_nb_stubs]/I");
  Out->Branch("stub_idx",            hit_idx,    "stub_idx[total_nb_stubs]/I");
  Out->Branch("stub_ptGEN",          hit_ptGEN, "stub_ptGEN[total_nb_stubs]/F");
  Out->Branch("stub_etaGEN",         hit_etaGEN, "stub_etaGEN[total_nb_stubs]/F");
  Out->Branch("stub_phi0GEN",        hit_phi0GEN, "stub_phi0GEN[total_nb_stubs]/F");
  Out->Branch("stub_IP",             hit_ip, "stub_IP[total_nb_stubs]/F");
  Out->Branch("stub_x",              hit_x, "stub_x[total_nb_stubs]/F");
  Out->Branch("stub_y",              hit_y, "stub_y[total_nb_stubs]/F");
  Out->Branch("stub_z",              hit_z, "stub_z[total_nb_stubs]/F");
  Out->Branch("stub_X0",             hit_x0, "stub_X0[total_nb_stubs]/F");
  Out->Branch("stub_Y0",             hit_y0, "stub_Y0[total_nb_stubs]/F");
  Out->Branch("stub_Z0",             hit_z0, "stub_Z0[total_nb_stubs]/F");
  

  /*********************************************/

  /******************* SAVING SECTORS **************/
  map<string,int> sectors_ids;
  map<string, Sector*> sectors_map;
  vector<Sector*> all_sectors = sectors->getAllSectors();
  for(unsigned int i=0;i<all_sectors.size();i++){
    Sector* tmpSec = all_sectors[i];
    sector_id=tmpSec->getOfficialID();
    if(sector_id==-1)
      sector_id=tmpSec->getKey();
    sector_layers = tmpSec->getNbLayers();
    for(int j=0;j<sector_layers;j++){
      vector<int> sec_l = tmpSec->getLadders(j);
      sector_layer_list[j] = tmpSec->getLayerID(j);
      nb_ladders_layer[j] = sec_l.size();
      for(unsigned int l=0;l<sec_l.size();l++){
	sector_layers_detail[j][l]=sec_l[l];
      }

    }
    cout<<tmpSec->getIDString()<<" -> "<<sector_id<<endl;
    sectors_ids[tmpSec->getIDString()]=sector_id;
    sectors_map[tmpSec->getIDString()]=tmpSec;
    SectorOut->Fill();
  }
  SectorOut->Write();
  delete SectorOut;

  /*************************************************/

  /***************** INPUT FILE ****************/
  TChain* TT = new TChain("L1TrackTrigger");
  TT->Add(eventsFilename.c_str());
  
  int               n_evt;

  int m_stub;

  vector<int>           m_stub_layer;  // Layer du stub (5 a 10 pour les 6 layers qui nous interessent)
  vector<int>           m_stub_module; // Position en Z du module contenant le stub
  vector<int>           m_stub_ladder; // Position en PHI du module contenant le stub
  vector<int>           m_stub_seg;    // Segment du module contenant le stub
  vector<int>           m_stub_strip;  // Strip du cluster interne du stub
  vector<int>           m_stub_tp;     // particule du stub
  vector<float>         m_stub_px_gen; // pt initial de la particule ayant genere le stub
  vector<float>         m_stub_py_gen; // pt initial de la particule ayant genere le stub
  vector<float>         m_stub_x0;     // utilise pour calculer la distance au point d'interaction
  vector<float>         m_stub_y0;     // utilise pour calculer la distance au point d'interaction
  vector<float>         m_stub_z0;
  vector<float>         m_stub_phi0;
  vector<float>         m_stub_eta_gen;
  vector<float>         m_stub_x;      // x coordinate of the hit
  vector<float>         m_stub_y;      // y coordinate of the hit
  vector<float>         m_stub_z;      // z coordinate of the hit

  vector<int>           *p_m_stub_layer =  &m_stub_layer;
  vector<int>           *p_m_stub_module = &m_stub_module;
  vector<int>           *p_m_stub_ladder = &m_stub_ladder;
  vector<int>           *p_m_stub_seg =    &m_stub_seg;
  vector<int>           *p_m_stub_strip =  &m_stub_strip;
  vector<int>           *p_m_stub_tp =     &m_stub_tp;
  vector<float>         *p_m_stub_pxGEN = &m_stub_px_gen;  
  vector<float>         *p_m_stub_pyGEN = &m_stub_py_gen;  
  vector<float>         *p_m_stub_x0 =     &m_stub_x0;
  vector<float>         *p_m_stub_y0 =     &m_stub_y0;
  vector<float>         *p_m_stub_z0 =     &m_stub_z0;
  vector<float>         *p_m_stub_phi0 =   &m_stub_phi0;
  vector<float>         *p_m_stub_etaGEN = &m_stub_eta_gen;
  vector<float>         *p_m_stub_x =      &m_stub_x;
  vector<float>         *p_m_stub_y =      &m_stub_y;
  vector<float>         *p_m_stub_z =      &m_stub_z;


  TT->SetBranchAddress("evt",            &n_evt);
  TT->SetBranchAddress("STUB_n",         &m_stub);
  TT->SetBranchAddress("STUB_layer",     &p_m_stub_layer);
  TT->SetBranchAddress("STUB_module",    &p_m_stub_module);
  TT->SetBranchAddress("STUB_ladder",    &p_m_stub_ladder);
  TT->SetBranchAddress("STUB_seg",       &p_m_stub_seg);
  TT->SetBranchAddress("STUB_strip",     &p_m_stub_strip);
  TT->SetBranchAddress("STUB_tp",        &p_m_stub_tp);
  TT->SetBranchAddress("STUB_X0",        &p_m_stub_x0);
  TT->SetBranchAddress("STUB_Y0",        &p_m_stub_y0);
  TT->SetBranchAddress("STUB_Z0",        &p_m_stub_z0);
  TT->SetBranchAddress("STUB_PHI0",      &p_m_stub_phi0);
  TT->SetBranchAddress("STUB_etaGEN",    &p_m_stub_etaGEN);
  TT->SetBranchAddress("STUB_pxGEN",     &p_m_stub_pxGEN);
  TT->SetBranchAddress("STUB_pyGEN",     &p_m_stub_pyGEN);
  TT->SetBranchAddress("STUB_x",         &p_m_stub_x);
  TT->SetBranchAddress("STUB_y",         &p_m_stub_y);
  TT->SetBranchAddress("STUB_z",         &p_m_stub_z);

  /*******************************************************/

  int n_entries_TT = TT->GetEntries();
  int num_evt = start;
  if(stop>n_entries_TT){
    stop=n_entries_TT-1;
    cout<<"Last event index too high : reset to "<<stop<<endl;
  }

  while(num_evt<n_entries_TT && num_evt<=stop){
    TT->GetEntry(num_evt);

    cout<<"Event "<<n_evt<<endl;

    vector<Hit*> hits;

    for(int i=0;i<m_stub;i++){
      int layer = m_stub_layer[i];
      int module = -1;
      module = CMSPatternLayer::getModuleCode(layer, m_stub_module[i]);
      if(module<0)// the stub is on the third Z position on the other side of the tracker -> out of range
	continue;
      int ladder = CMSPatternLayer::getLadderCode(layer, m_stub_ladder[i]);
      int segment =  CMSPatternLayer::getSegmentCode(layer, ladder, m_stub_seg[i]);
      if(segment<0 || segment>1){
	cout<<"Invalid segment on event "<<n_evt<<endl;
	continue;
      }
      int strip = m_stub_strip[i];
      int tp = m_stub_tp[i];
      float eta = m_stub_eta_gen[i];
      float phi0 = m_stub_phi0[i];
      float spt = sqrt(m_stub_px_gen[i]*m_stub_px_gen[i]+m_stub_py_gen[i]*m_stub_py_gen[i]);
      float x = m_stub_x[i];
      float y = m_stub_y[i];
      float z = m_stub_z[i];
      float x0 = m_stub_x0[i];
      float y0 = m_stub_y0[i];
      float z0 = m_stub_z0[i];
      
      //cout<<layer<<" "<<module<<" "<<ladder<<" "<<segment<<" "<<strip<<endl;

      float ip = sqrt(m_stub_x0[i]*m_stub_x0[i]+m_stub_y0[i]*m_stub_y0[i]);

      Hit* h = new Hit(layer,ladder, module, segment, strip, i, tp, spt, ip, eta, phi0, x, y, z, x0, y0, z0);

      if(sectors->getSector(*h)!=NULL)
	hits.push_back(h);
      else
	delete(h);
    }

    vector<Sector*> pattern_list = find(hits);

    //Traitement des patterns actif : enregistrement, affichage...
    nb_layers = tracker.getNbLayers();
    event_id=num_evt;//we use the index in the file as event_id (most of our input files do not have a valid event_id)
    nb_patterns = 0;
    ori_nb_stubs = (int)hits.size();
    
    nb_tracks = 0;
    for(int i=0;i<nb_layers;i++){
      memset(superStrips[i],0,MAX_NB_PATTERNS*sizeof(int));
    }
    memset(hit_layer,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(short));
    memset(hit_ladder,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(short));
    memset(hit_tp,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(int));
    memset(hit_idx,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(int));
    memset(hit_ptGEN,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));
    memset(hit_etaGEN,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));
    memset(hit_phi0GEN,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));
    memset(hit_ip,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));
    memset(hit_x,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));
    memset(hit_y,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));
    memset(hit_z,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));
    memset(hit_x0,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));
    memset(hit_y0,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));
    memset(hit_z0,0,MAX_NB_PATTERNS*MAX_NB_HITS*sizeof(float));

    int patternIndex=0;
    int trackIndex=0;
    int stubIndex = 0;
    totalNbHits = 0;
    // loop on sectors
    for(unsigned int i=0;i<pattern_list.size();i++){
      vector<GradedPattern*> pl = pattern_list[i]->getPatternTree()->getLDPatterns();
      vector<Track*> tracks;

      /////////FITTER////
      TrackFitter* fitter = (sectors_map[pattern_list[i]->getIDString()])->getFitter();
      if(fitter!=NULL){
	fitter->setSectorID(pattern_list[i]->getOfficialID());
	for(unsigned int l=0;l<pl.size();l++){
	  fitter->addPattern(pl[l]);
	}
	fitter->mergePatterns();
	fitter->fit();
	fitter->mergeTracks();
	tracks = fitter->getTracks();
	fitter->clean();
      }
      else{
	cout<<"No fitter found for this sector!"<<endl;
      }
      ///////////////////////

      nb_patterns+=(int)pl.size();
      nb_tracks+=(int)tracks.size();
      int sec_id = sectors_ids[pattern_list[i]->getIDString()];
      nbHitPerPattern[patternIndex]=0;
      //loop on the patterns
      unsigned int max_patterns = pl.size();
      unsigned int max_tracks = tracks.size();
      if((int)max_patterns>MAX_NB_PATTERNS){
          max_patterns=MAX_NB_PATTERNS;
	  nb_patterns=MAX_NB_PATTERNS;
	  if(max_tracks>max_patterns)
	    max_tracks=max_patterns;
          cout<<"WARNING : Too may patterns in event "<<n_evt<<" : "<<pl.size()<<" -> keep only the first "<<MAX_NB_PATTERNS<<"."<<endl;
      }

      set<short> stub_ids;

      for(unsigned int j=0;j<max_patterns;j++){
	//loop on layers
	for(int k=0;k<nb_layers;k++){
	  int sstripValue = pl[j]->getLayerStrip(k)->getIntValue();
	  superStrips[k][patternIndex]=sstripValue;
	}
	//sector of the pattern
	pattern_sector_id[patternIndex]=sec_id;
	
	//stubs of the patterns
	vector<Hit*> active_hits = pl[j]->getHits();
	int nbHits = active_hits.size();
	if(nbHits>MAX_NB_HITS) // if we have too many hits, we keep only the MAX_NB_HITS first
	  nbHits=MAX_NB_HITS;
	nbHitPerPattern[patternIndex]=nbHits;
	totalNbHits+=nbHits;
	for(int k=0;k<nbHits;k++){
	  //cout<<*active_hits[k]<<endl;
	  
	  hit_layer[stubIndex]=active_hits[k]->getLayer();
	  hit_ladder[stubIndex]=active_hits[k]->getLadder();
	  hit_zPos[stubIndex]=active_hits[k]->getModule();
	  hit_segment[stubIndex]=active_hits[k]->getSegment();
	  hit_strip[stubIndex]=active_hits[k]->getStripNumber();
	  hit_tp[stubIndex]=active_hits[k]->getParticuleID();
	  hit_idx[stubIndex]=active_hits[k]->getID();
	  hit_ptGEN[stubIndex]=active_hits[k]->getParticulePT();
	  hit_etaGEN[stubIndex]=active_hits[k]->getParticuleETA();
	  hit_phi0GEN[stubIndex]=active_hits[k]->getParticulePHI0();
	  hit_ip[stubIndex]=active_hits[k]->getParticuleIP();
	  hit_x[stubIndex]=active_hits[k]->getX();
	  hit_y[stubIndex]=active_hits[k]->getY();
	  hit_z[stubIndex]=active_hits[k]->getZ();
	  hit_x0[stubIndex]=active_hits[k]->getX0();
	  hit_y0[stubIndex]=active_hits[k]->getY0();
	  hit_z0[stubIndex]=active_hits[k]->getZ0();

	  stub_ids.insert(active_hits[k]->getID());
	  
	  stubIndex++;
	}
	
	patternIndex++;
	delete pl[j];
      }

      sel_nb_stubs = stub_ids.size();

      for(unsigned int j=0;j<max_tracks;j++){
	track_pt[trackIndex] = tracks[j]->getCurve();
	track_phi[trackIndex]= tracks[j]->getPhi0();
	track_d0[trackIndex] = tracks[j]->getD0();
	track_eta[trackIndex]= tracks[j]->getEta0();
	track_z0[trackIndex] = tracks[j]->getZ0();
	delete tracks[j];
	trackIndex++;
      }
    }
    Out->Fill();

    //////////////////////////////////
    for(unsigned int i=0;i<hits.size();i++){
      delete hits[i];
    }

    for(unsigned int i=0;i<pattern_list.size();i++){
      delete pattern_list[i];
    }
    num_evt++;
  }
  //Out->Print();
  Out->Write();
  t->Close();
  delete Out;
  delete t;

  delete TT;
}

vector<Sector*> PatternFinder::find(vector<Hit*> hits){
  tracker.clear();
  for(unsigned int i=0;i<hits.size();i++){
    //cout<<*hits[i]<<endl;
    tracker.receiveHit(*hits[i]);
  }
  return sectors->getActivePatternsPerSector(active_threshold);
}

