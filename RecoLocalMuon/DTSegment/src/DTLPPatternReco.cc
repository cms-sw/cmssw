/** \class DTLPPatternReco
 *
 * Algo for reconstructing 2d segment in DT using a linear programming approach
 *  
 * $Date: 2009/08/14 14:15:52 $
 * $Revision: 1.2 $
 * \author Enzo Busseti - SNS Pisa <enzo.busseti@sns.it>
 * 
 */

/* C++ Headers*/
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <math.h>
#include <iostream>

/* Collaborating class headers*/
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"


/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTLPPatternReco.h"

/*The algorithm*/
#include "RecoLocalMuon/DTSegment/src/DTLPPatternRecoAlgorithm.h"

using namespace std;


//used by print_gnuplot
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

//Constructor
DTLPPatternReco::DTLPPatternReco(const edm::ParameterSet& pset): DTRecSegment2DBaseAlgo(pset){
  std::cout << "DTLPPatternReco Constructor Called" << std::endl; 
  //getting parameters from python
  theDeltaFactor = pset.getParameter<double>("DeltaFactor");//multiplier of sigmas in LP algo
  theMaxAlphaTheta = pset.getParameter<double>("maxAlphaTheta");
  theMaxAlphaPhi = pset.getParameter<double>("maxAlphaPhi");
  theMinimumQ = pset.getParameter<double>("min_q");
  theMaximumQ = pset.getParameter<double>("max_q");
  theBigM = pset.getParameter<double>("bigM");//"Big M" used in LP
  theUpdator = new DTSegmentUpdator(pset);  
  //event counter used by gnuplot macro producer to name the files
  event_counter = 0;
}


//Destructor
DTLPPatternReco::~DTLPPatternReco() {
  std::cout << "DTLPPatternReco Destructor Called" << std::endl;
 delete theUpdator;
}

void DTLPPatternReco::setES(const edm::EventSetup& setup){
  
  /* Get the DT Geometry */

  setup.get<MuonGeometryRecord>().get(theDTGeometry);
  theUpdator->setES(setup);
  
  }

edm::OwnVector<DTSLRecSegment2D> DTLPPatternReco::reconstruct(const DTSuperLayer* sl,
							      const std::vector<DTRecHit1DPair>& pairs){

  /*That is just a driver to the common reconstructSegmentOrSupersegment templated method */
  ReconstructInSLOrChamber flagSLChamber = ReconstructInSL;
  return * (edm::OwnVector<DTSLRecSegment2D> * ) reconstructSegmentOrSupersegment(pairs, flagSLChamber);
}

edm::OwnVector<DTChamberRecSegment2D> DTLPPatternReco::reconstructSupersegment( const std::vector<DTRecHit1DPair>& pairs){

  /*That is just a driver to the common reconstructSegmentOrSupersegment templated method */

  ReconstructInSLOrChamber flagSLChamber = ReconstructInChamber;
  return *(edm::OwnVector<DTChamberRecSegment2D> * ) reconstructSegmentOrSupersegment( pairs, flagSLChamber);
}

void *  DTLPPatternReco::reconstructSegmentOrSupersegment(const std::vector<DTRecHit1DPair>& pairs,
						       ReconstructInSLOrChamber& sl_chamber ){
  //two geometry objects
  int counter = 0;
  const DTSuperLayer * sl;
  const DTChamber * chamber;
  void * theResults = NULL;  //this is the pointer that I return
  if (sl_chamber == ReconstructInSL)
    theResults = (void*)new edm::OwnVector<DTSLRecSegment2D>;
  if (sl_chamber == ReconstructInChamber)
    theResults = (void*)new edm::OwnVector<DTChamberRecSegment2D>;

  //I get the SL object 
  sl = theDTGeometry -> superLayer( pairs.begin()->wireId().superlayerId() );
  //I get a chamber object
  chamber = theDTGeometry -> chamber( pairs.begin()->wireId().chamberId() );
  
  /*Create the pairPointers array, and fill it with pointers to the RecHitPair objects*/ 
  std::vector<const DTRecHit1DPair *> pairPointers;
  for (std::vector<DTRecHit1DPair>::const_iterator it = pairs.begin(); it != pairs.end(); ++it)
    pairPointers.push_back(&(*it));
  
  std::list<double> pz; //positions of hits along z
  std::list<double> px; //along x (left and right the wires)
  std::list<double> pex; //errors (sigmas) on x
  ResultLPAlgo theAlgoResults;//datastructure containing all useful results from perform_fit
  
  /*I populate the coordinates lists, in SL or chamber ref frame. */
  populateCoordinatesLists(pz, px, pex,  sl, chamber, pairPointers, sl_chamber); 
  
  /*lpAlgo returns true as long as it manages to fit meaningful straight lines*/
  while(lpAlgorithm(theAlgoResults, pz, px, pex, -10, 10, theMinimumQ, theMaximumQ, theBigM, theDeltaFactor) ){
    counter++;
    std::cout << "Creating 2Dsegment" << std::endl;
    
    /*Now I create the actual new segment*/ 
    LocalPoint seg2Dposition( (float)theAlgoResults.qVar, 0. , 0. );
    LocalVector seg2DDirection ((float) theAlgoResults.mVar,  0. , 1.  );//don't know if I need to normalize the vector
    AlgebraicSymMatrix seg2DCovMatrix;
    double chi2 = theAlgoResults.chi2Var;
    std::vector<DTRecHit1D> hits1D;
    for(unsigned int i = 0; i < theAlgoResults.lambdas.size(); i++){
      if(theAlgoResults.lambdas[i]%2)
	hits1D.push_back( * pairPointers[(theAlgoResults.lambdas[i] - 1)/2]->componentRecHit(DTEnums::Right));
      else hits1D.push_back(* pairPointers[theAlgoResults.lambdas[i]/2]->componentRecHit(DTEnums::Left));
    }
    DTSLRecSegment2D * SLPointer = NULL;
    DTChamberRecSegment2D * chamberPointer = NULL;
    if (sl_chamber == ReconstructInSL)
      SLPointer = new DTSLRecSegment2D(sl->id(), seg2Dposition,seg2DDirection, seg2DCovMatrix, chi2, hits1D);
    if (sl_chamber == ReconstructInChamber)
      chamberPointer = new DTChamberRecSegment2D(chamber->id(),seg2Dposition,seg2DDirection, seg2DCovMatrix, chi2, hits1D);
    
    /*I call the Updator to update the segment just created. */
    std::cout << "2D segment created, updating it" << std::endl;
    if (sl_chamber == ReconstructInSL)
      theUpdator->update(SLPointer);
    if (sl_chamber == ReconstructInChamber)
      theUpdator->update(chamberPointer); 
    std::cout << "2D segment update, adding it to the result vector" << std::endl;
    
    /*I add it to the result vector.*/
    if (sl_chamber == ReconstructInSL)
      (( edm::OwnVector<DTSLRecSegment2D> *)theResults) ->push_back(SLPointer);
    if (sl_chamber == ReconstructInChamber)
      (( edm::OwnVector<DTChamberRecSegment2D> *)theResults) -> push_back(chamberPointer);

    /*this removes the used hits from the vector of pointers to pairs*/
    std::cout << "removing used hits" << std::endl;
    removeUsedHits(theAlgoResults, pairPointers);
    std::cout << "Used hits removed" << std::endl;

    /*And again populate the coordinates list*/
    
    populateCoordinatesLists(pz, px, pex,  sl, chamber, pairPointers, sl_chamber);
    theAlgoResults.lambdas.clear();     
  }
  //px.clear();
  //pz.clear();
  //pex.clear();
  
  //to disable the gnuplot macros production, just comment these two lines
  //printGnuplot((edm::OwnVector<DTSLRecSegment2D>*)theResults, pairs);
  // event_counter++;
  return theResults;
}



void DTLPPatternReco::populateCoordinatesLists(std::list<double>& pz, std::list<double>& px, std::list<double>& pex,
					       const DTSuperLayer* sl, const DTChamber* chamber,
					       const std::vector<const DTRecHit1DPair*>& pairsPointers,
					       const ReconstructInSLOrChamber sl_chamber) {

  /*Populate the pz, px and pex lists with coordinates taken from the actual rec hit pairs.
    The sl_chamber flag tells this function to use SL or chamber coordinate ref. frame. */

  px.clear(); pex.clear(); pz.clear();
  //iterate on pairs of rechits
  for (std::vector<const DTRecHit1DPair*>::const_iterator it = pairsPointers.begin(); it!=pairsPointers.end(); ++it){
    
    DTWireId theWireId = (*it)->wireId();
    const DTLayer * theLayer = (DTLayer*)theDTGeometry->layer(theWireId.layerId());
    LocalPoint thePosition;

    //extract a pair of rechits from the pairs vector
    std::pair<const DTRecHit1D*, const DTRecHit1D*> theRecHitPair = (*it) -> componentRecHits();

    //left hit
    if(sl_chamber == ReconstructInSL)
      thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.first-> localPosition()));
    if(sl_chamber == ReconstructInChamber)
      thePosition = chamber -> toLocal(theLayer->toGlobal(theRecHitPair.first -> localPosition()));
    pex.push_back( (double)std::sqrt(theRecHitPair.first -> localPositionError().xx()));
    pz.push_back( thePosition.z() );
    px.push_back( thePosition.x() );
    std::cout << pz.back() << " " <<  px.back() << " "<< pex.back() << std::endl;
    
    //and right hit
    if(sl_chamber == ReconstructInSL) 
      thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.second->localPosition()));
    if(sl_chamber == ReconstructInChamber)
      thePosition = chamber -> toLocal(theLayer->toGlobal(theRecHitPair.second -> localPosition()));
    pex.push_back( (double)std::sqrt(theRecHitPair.second-> localPositionError().xx()));
    pz.push_back( thePosition.z() );
    px.push_back( thePosition.x() );
    std::cout << pz.back() << " " <<  px.back() << " "<< pex.back() << std::endl;
  }
std::cout << "DTLPPatternReco:: : px, pz and pex lists populated " << std::endl;
}


void DTLPPatternReco::removeUsedHits(const ResultLPAlgo& theAlgoResults,
				     std::vector<const DTRecHit1DPair*>& pairsPointers){

  /* The used hits are characterized by the corresponding lambda in  theAlgoResults.lambdas[]
     being equal to zero. I remove them, deleting their references in the pairsPointers vector.
     This is done by declaring a temp vector, copying in it only the references to be re-used, and 
     assigning the original vector equal to it. */
  
  std::vector<const DTRecHit1DPair*> temp;  
  for(unsigned int i =0; i !=  pairsPointers.size(); ++i){
    std::cout << "RemoveUsedHits: Iterating on the " << i << "pair on the pointer vector." <<std::endl;
    if( theAlgoResults.lambdas[i*2] == 1 &&  theAlgoResults.lambdas[i*2 + 1] == 1 ) temp.push_back(pairsPointers[i]);
    else std::cout << "not copied a pair" << std::endl;
    std::cout << "We have lambdas: " << theAlgoResults.lambdas[i*2] << " and "
	      << theAlgoResults.lambdas[i*2 + 1] << std::endl;
  }
  pairsPointers = temp;
}
	
    

void DTLPPatternReco::printGnuplot( edm::OwnVector<DTSLRecSegment2D> * theResults, const std::vector<DTRecHit1DPair>& pairs)
{

  std::string directory = "/afs/cern.ch/user/e/ebusseti/CMSSW_3_1_1/src/RecoLocalMuon/DTSegment/gnuplot/";
  unsigned int num_seg = theResults -> size();
  std::stringstream file_name;  
  file_name << directory << num_seg << "segs_event" << event_counter << "dati.txt";
  std::string file_name_str = file_name.str();
  std::ofstream data (file_name_str.c_str() );
  if (!data.is_open()) std::cerr << "file could not be opened"<< std::endl;
  //writing coordinates on dati file
  const DTSuperLayer * sl;
  sl = theDTGeometry -> superLayer( pairs.begin()->wireId().superlayerId() );
  for (std::vector<DTRecHit1DPair>::const_iterator it = pairs.begin(); it!=pairs.end(); ++it){
    DTWireId theWireId = (it)->wireId();
    const DTLayer * theLayer = (DTLayer*)theDTGeometry->layer(theWireId.layerId());
    //transform the Local position in Layer-rf in a SL local position, or chamber
    std::pair<const DTRecHit1D*, const DTRecHit1D*> theRecHitPair = (it) -> componentRecHits();
    LocalPoint thePosition;
    //left hit
    thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.first-> localPosition()));
    data <<  thePosition.z() << " "
	 << thePosition.x()  << " "
	 << (double)std::sqrt(theRecHitPair.first-> localPositionError().xx()) << std::endl;
    //and right hit
    thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.second->localPosition()));
    data <<  thePosition.z() << " "
	 << thePosition.x()  << " "
	 << (double)std::sqrt(theRecHitPair.second-> localPositionError().xx()) << std::endl;
  }
  data.close();
  std::stringstream macro_name;
  macro_name << directory << num_seg << "segs_event" << event_counter << "macro.gpt";
  std::string macro_name_str = macro_name.str();
  std::ofstream macro(macro_name_str.c_str() );
  if (!macro.is_open()) std::cerr << "file could not be opened"<< std::endl;
  //writing the macro
  for (unsigned int i =0; i < num_seg; i++){
    std::stringstream mname;
    std::stringstream qname;
    std::stringstream funname;
    mname << "m"<< i;
    qname << "q"<< i;
    funname << "f"<< i;
    macro << qname.str() << "=" << (*theResults)[i].localPosition().x() << ";"
	  << mname.str() << "=" << (*theResults)[i].localDirection().x() << ";"
	  << funname.str() << "(x)=" << "x/ " << mname.str() <<  "-" << qname.str() << "/" << mname.str() << ";"
	  << std::endl;
  }
  macro << "set yrange [-3:3]; plot \"" << file_name_str << "\" using 2:1:($3 * 4) with xerrorbars";
  for (unsigned int i =0; i < num_seg; i++){
    std::stringstream funname;
    funname << "f"<< i << "(x)";
    macro << ", " << funname.str();
  }
  macro << std::endl;    
  macro.close();
}


/*
  void DTLPPatternReco::createSegment(void * theResults, ResultLPAlgo& theAlgoResults, const std::vector<const DTRecHit1DPair*>& pairPointers,const  DTSuperLayer * sl, const DTChamber * chamber, ReconstructInSLOrChamber sl_chamber ){
  //std::cout << "Creating 2Dsegment" << std::endl;
  LocalPoint seg2Dposition( (float)theAlgoResults.qVar, 0. , 0. );
  LocalVector seg2DDirection ((float) theAlgoResults.mVar,  0. , 1.  );//don't know if I need to normalize the vector
  AlgebraicSymMatrix seg2DCovMatrix;
  std::vector<DTRecHit1D> hits1D;
  for(unsigned int i = 0; i < theAlgoResults.lambdas.size(); i++){
    if(theAlgoResults.lambdas[i]%2) hits1D.push_back( * pairPointers[(theAlgoResults.lambdas[i] - 1)/2]->componentRecHit(DTEnums::Right));
    else hits1D.push_back(* pairPointers[theAlgoResults.lambdas[i]/2]->componentRecHit(DTEnums::Left));
    }
  if(sl_chamber == ReconstructInSL){
    edm::OwnVector<DTSLRecSegment2D> * result = (edm::OwnVector<DTSLRecSegment2D>*)theResults;
    DTSLRecSegment2D * candidate;
    candidate = new DTSLRecSegment2D(sl->id(),seg2Dposition,seg2DDirection, seg2DCovMatrix, theAlgoResults.chi2Var, hits1D);
    std::cout << "Updating 2D segment" << std::endl;
    theUpdator->update(candidate);
    result ->push_back(candidate);
  }
  if(sl_chamber == ReconstructInChamber){
    edm::OwnVector<DTChamberRecSegment2D> * result = (edm::OwnVector<DTChamberRecSegment2D>*)theResults;
    DTChamberRecSegment2D * candidate;
    candidate = new DTChamberRecSegment2D(chamber->id(),seg2Dposition,seg2DDirection, seg2DCovMatrix, theAlgoResults.chi2Var, hits1D);   
    std::cout << "Updating 2D segment" << std::endl;
    theUpdator->update(candidate);
    result ->push_back(candidate);
  }
}
*/


/*

void * DTLPPatternReco::reconstructSegmentOrSupersegment(const std::vector<DTRecHit1DPair>& pairs, ReconstructInSLOrChamber sl_chamber ){
  //two geometry objects
  int counter = 0;
  const DTSuperLayer * sl;
  const DTChamber * chamber;
  //this is the void pointer that I return
  void * theResults;
  //I get the SL object if I'm reconstructing a SL segment, and create the result Vector
  if (sl_chamber == ReconstructInSL) {
    sl = theDTGeometry -> superLayer( pairs.begin()->wireId().superlayerId() );
    theResults =(void*) new edm::OwnVector<DTSLRecSegment2D>;
  }
  //otherwise I get a chamber object, and create the result vector
  if (sl_chamber == ReconstructInChamber){
    chamber = theDTGeometry -> chamber( pairs.begin()->wireId().chamberId() );
    theResults =(void*) new edm::OwnVector<DTChamberRecSegment2D>;
  }
  std::vector<const DTRecHit1DPair *> pairPointers;
  for (std::vector<DTRecHit1DPair>::const_iterator it = pairs.begin(); it != pairs.end(); ++it){
    pairPointers.push_back(&(*it));
  }
  std::list<double> pz; //positions of hits along z
  std::list<double> px; //along x (left and right the wires)
  std::list<double> pex; //errors (sigmas) on x
  ResultLPAlgo theAlgoResults;//datastructure containing all useful results from perform_fit
  //I populate the coordinates lists, in SL or chamber ref frame
  populateCoordinatesLists(pz, px, pex,  sl, chamber, pairPointers, sl_chamber);
 
 //lpAlgo returns true as long as it manages to fit meaningful straight lines
  while(lpAlgorithm(theAlgoResults, pz, px, pex, -10, 10, theMinimumQ, theMaximumQ, theBigM, theDeltaFactor) ){
    counter++;
    std::cout << "Creating 2Dsegment" << std::endl;
    //this method populates the result vector with segments
    createSegment(theResults, theAlgoResults, pairPointers, sl, chamber,  sl_chamber);
    //now i update the recsegment
    std::cout << "2D segment created, removing used hits" << std::endl;
    //this removes the used hits from the vector of pointers to pairs
    removeUsedHits(theAlgoResults, pairPointers);
    populateCoordinatesLists(pz, px, pex,  sl, chamber, pairPointers, sl_chamber);
    //std::cout << "Used hits removed" << std::endl;
    theAlgoResults.lambdas.clear();
    // std::cout << "Checking if I need to reiterate perform_fit" << std::endl;
     }
  px.clear();
  pz.clear();
  pex.clear();
  std::cout << "---------------------------------------------" << std::endl;
  std::cout << " WE HAVE FITTED " << counter  << " SEGMENTS." << std::endl;
std::cout << "---------------------------------------------" << std::endl;
 
//to disable the gnuplot macros production, just comment these two lines
//printGnuplot((edm::OwnVector<DTSLRecSegment2D>*)theResults, pairs);
// event_counter++;

  return theResults;
}

*/
