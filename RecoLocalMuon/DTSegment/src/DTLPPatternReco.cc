/** \class DTLPPatternReco
 *
 * Algo for reconstructing 2d segment in DT using a linear programming approach
 *  
 * $Date: 2009/09/04 08:27:56 $
 * $Revision: 1.6 $
 * \author Enzo Busseti - SNS Pisa <enzo.busseti@sns.it>
 * 
 */

/* C++ Headers*/
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <cmath>
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
using namespace lpAlgo;

//used by print_gnuplot
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

//Constructor
DTLPPatternReco::DTLPPatternReco(const edm::ParameterSet& pset): DTRecSegment2DBaseAlgo(pset){

  //getting parameters from python
  theDeltaFactor = pset.getParameter<double>("DeltaFactor");//multiplier of sigmas in LP algo
  theMaxAlphaTheta = pset.getParameter<double>("maxAlphaTheta");
  theMaxAlphaPhi = pset.getParameter<double>("maxAlphaPhi");
  theMinimumQ = pset.getParameter<double>("min_q");
  theMaximumQ = pset.getParameter<double>("max_q");
  theBigM = pset.getParameter<double>("bigM");//"Big M" used in LP
  theUpdator = new DTSegmentUpdator(pset);
  debug = pset.getUntrackedParameter<bool>("debug", false);
  //event counter used by gnuplot macro producer to name the files
  event_counter = 0;
  if(debug) cout << "DTLPPatternReco Constructor Called" << endl; 
  //FIXME
  debug=true;
 
}


//Destructor
DTLPPatternReco::~DTLPPatternReco() {
  if(debug) cout << "DTLPPatternReco Destructor Called" << endl;
  delete theUpdator;
}

void DTLPPatternReco::setES(const edm::EventSetup& setup){
  
  /* Get the DT Geometry */

  setup.get<MuonGeometryRecord>().get(theDTGeometry);
  theUpdator->setES(setup);
  
  }

edm::OwnVector<DTSLRecSegment2D> DTLPPatternReco::reconstruct(const DTSuperLayer* sl,
							      const vector<DTRecHit1DPair>& pairs){

  /*That is just a driver to the common reconstructSegmentOrSupersegment templated method */
  ReconstructInSLOrChamber flagSLChamber = ReconstructInSL;
  return * (edm::OwnVector<DTSLRecSegment2D> * ) reconstructSegmentOrSupersegment(pairs, flagSLChamber);
}

edm::OwnVector<DTChamberRecSegment2D> DTLPPatternReco::reconstructSupersegment( const vector<DTRecHit1DPair>& pairs){

  /*That is just a driver to the common reconstructSegmentOrSupersegment templated method */

  ReconstructInSLOrChamber flagSLChamber = ReconstructInChamber;
  return *(edm::OwnVector<DTChamberRecSegment2D> * ) reconstructSegmentOrSupersegment( pairs, flagSLChamber);
}

void *  DTLPPatternReco::reconstructSegmentOrSupersegment(const vector<DTRecHit1DPair>& pairs,
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
  vector<const DTRecHit1DPair *> pairPointers;
  for (vector<DTRecHit1DPair>::const_iterator it = pairs.begin(); it != pairs.end(); ++it){
    DTWireId theWireId = it->wireId();
    const DTLayer * theLayer = (DTLayer*)theDTGeometry->layer(theWireId.layerId());
    LocalPoint uno  = sl -> toLocal(theLayer->toGlobal(it->componentRecHits().first->localPosition()));
    LocalPoint due = sl -> toLocal(theLayer->toGlobal(it->componentRecHits().second->localPosition()));
    if(uno.x() != due.x())//FIXME we have to change this!! FIXME
      pairPointers.push_back(&(*it));
  }
  
  list<double> pz; //positions of hits along z
  list<double> px; //along x (left and right the wires)
  list<double> pex; //errors (sigmas) on x
  list<int> layers;
  ResultLPAlgo theAlgoResults;//datastructure containing all useful results from perform_fit
  
  /*I populate the coordinates lists, in SL or chamber ref frame. */
  populateCoordinatesLists(pz, px, pex, layers,  sl, chamber, pairPointers, sl_chamber);

  if(debug) {
    if (sl_chamber ==  ReconstructInSL)
      cout << "DTLPPatternReco::reconstruct in SL: " << sl->id() << endl; 
    else if (sl_chamber ==  ReconstructInChamber) cout << "DTLPPatternReco::reconstruct in Ch: " << chamber->id() << endl;
  }


  findAngleConstraints( theMinimumM, theMaximumM, sl, chamber, sl_chamber );
  
  /*lpAlgo returns true as long as it manages to fit meaningful straight lines*/
  while(lpAlgorithm(theAlgoResults, pz, px, pex, layers, theMinimumM, theMaximumM, theMinimumQ, theMaximumQ, theBigM, theDeltaFactor) ){
    counter++;
    if(debug) cout << "Creating 2Dsegment" << endl;
    
    /*Now I create the actual new segment*/ 
    LocalPoint seg2Dposition( (float)theAlgoResults.qVar, 0. , 0. );
    float normalizer = sqrt( (float)theAlgoResults.mVar * (float)theAlgoResults.mVar + 1);
    LocalVector seg2DDirection (-(float) theAlgoResults.mVar / normalizer,  0. , -1. / normalizer  );
    AlgebraicSymMatrix seg2DCovMatrix;
    double chi2 = theAlgoResults.chi2Var;
    vector<DTRecHit1D> hits1D;
    for(unsigned int i = 0; i < theAlgoResults.lambdas.size(); i++){
      if(theAlgoResults.lambdas[i] == 0){
	if (i%2 == 0)
	  hits1D.push_back(*pairPointers[i/2]->componentRecHit(DTEnums::Left));
	else 
	  hits1D.push_back(*pairPointers[(i-1)/2]->componentRecHit(DTEnums::Right));
      }
    }
    DTSLRecSegment2D * SLPointer = NULL;
    DTChamberRecSegment2D * chamberPointer = NULL;
    if (sl_chamber == ReconstructInSL)
      SLPointer = new DTSLRecSegment2D(sl->id(), seg2Dposition,seg2DDirection, seg2DCovMatrix, chi2, hits1D);
    if (sl_chamber == ReconstructInChamber)
      chamberPointer = new DTChamberRecSegment2D(chamber->id(),seg2Dposition,seg2DDirection, seg2DCovMatrix, chi2, hits1D);
    
    /*I call the Updator to update the segment just created. */
    if (sl_chamber == ReconstructInSL)
      theUpdator->update(SLPointer);
    if (sl_chamber == ReconstructInChamber)
      theUpdator->update(chamberPointer); 

    /*DEBUGGING*/
    /* if (atan(SLPointer->localDirection().x()/SLPointer->localDirection().z() > 1.9) ) cout
										  << "###############################"
										  << endl << "Found wrong segment"
										  << endl;*/
				    
    
    /*I add it to the result vector.*/
    if(debug) cout << "2D segment created and updated, adding it to the result vector" << endl;
    if (sl_chamber == ReconstructInSL)
      (( edm::OwnVector<DTSLRecSegment2D> *)theResults) ->push_back(SLPointer);
    if (sl_chamber == ReconstructInChamber)
      (( edm::OwnVector<DTChamberRecSegment2D> *)theResults) -> push_back(chamberPointer);

    /*this removes the used hits from the vector of pointers to pairs*/
    removeUsedHits(theAlgoResults, pairPointers);

    /*And again populate the coordinates list*/
    // FIXME: can this operation be merged with removeUsedHits to gain on # of loops?
    populateCoordinatesLists(pz, px, pex, layers,  sl, chamber, pairPointers, sl_chamber); 
    theAlgoResults.lambdas.clear();     
  }
  //px.clear();
  //pz.clear();
  //pex.clear();
  
  //to disable the gnuplot macros production, just comment these two lines
  //printGnuplot((edm::OwnVector<DTSLRecSegment2D>*)theResults, pairs);
  // event_counter++;
    if(debug) cout <<  sl->id() << " reconstructed " << counter << " segments" << endl;
  return theResults;
}



void DTLPPatternReco::populateCoordinatesLists(list<double>& pz, list<double>& px, list<double>& pex, list<int>& layers,
					       const DTSuperLayer* sl, const DTChamber* chamber,
					       const vector<const DTRecHit1DPair*>& pairsPointers,
					       const ReconstructInSLOrChamber sl_chamber) {

  /*Populate the pz, px and pex lists with coordinates taken from the actual rec hit pairs.
    The sl_chamber flag tells this function to use SL or chamber coordinate ref. frame. */

  px.clear(); pex.clear(); pz.clear(); layers.clear();
  //iterate on pairs of rechits
  for (vector<const DTRecHit1DPair*>::const_iterator it = pairsPointers.begin(); it!=pairsPointers.end(); ++it){
    
    DTWireId theWireId = (*it)->wireId();
    const DTLayer * theLayer = (DTLayer*)theDTGeometry->layer(theWireId.layerId());
    LocalPoint thePosition;

    //extract a pair of rechits from the pairs vector
    pair<const DTRecHit1D*, const DTRecHit1D*> theRecHitPair = (*it) -> componentRecHits();

    //left hit
    if(sl_chamber == ReconstructInSL)
      thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.first-> localPosition()));
    if(sl_chamber == ReconstructInChamber)
      thePosition = chamber -> toLocal(theLayer->toGlobal(theRecHitPair.first -> localPosition()));
    pex.push_back( (double)sqrt(theRecHitPair.first -> localPositionError().xx()));
    pz.push_back( thePosition.z() );
    px.push_back( thePosition.x() );
    layers.push_back(theWireId.layer()+ theWireId.superlayer() * 10);

    if (debug) cout << pz.back() << " " <<  px.back() << " "<< pex.back() <<" " << layers.back()<< endl;
    
    //and right hit
    if(sl_chamber == ReconstructInSL)
      thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.second->localPosition()));
     if(sl_chamber == ReconstructInChamber)
      thePosition = chamber -> toLocal(theLayer->toGlobal(theRecHitPair.second -> localPosition()));
    pex.push_back( (double)sqrt(theRecHitPair.second-> localPositionError().xx()));
    pz.push_back( thePosition.z() );
    px.push_back( thePosition.x() );
    layers.push_back(theWireId.layer()+ theWireId.superlayer() * 10);
 
    if (debug) cout << pz.back() << " " <<  px.back() << " "<< pex.back()<< " " << layers.back() << endl;
  }
  if(debug) cout << "DTLPPatternReco:: : px, pz and pex lists populated " << endl;
}


void DTLPPatternReco::removeUsedHits(const ResultLPAlgo& theAlgoResults,
				     vector<const DTRecHit1DPair*>& pairsPointers){

  /* The used hits are characterized by the corresponding lambda in  theAlgoResults.lambdas[]
     being equal to zero. I remove them, deleting their references in the pairsPointers vector.
     This is done by declaring a temp vector, copying in it only the references to be re-used, and 
     assigning the original vector equal to it. */
  vector<const DTRecHit1DPair*> temp;  
  for(unsigned int i =0; i !=  pairsPointers.size(); ++i){
    if(debug)cout << "The point " << i << " has lambdas: " << theAlgoResults.lambdas[i*2] << " and "
		  << theAlgoResults.lambdas[i*2 + 1] << endl;
    if( theAlgoResults.lambdas[i*2] == 1 &&  theAlgoResults.lambdas[i*2 + 1] == 1 ) temp.push_back(pairsPointers[i]);
    else if(debug) cout << "not copied a pair" << endl;
  }
  pairsPointers = temp;
}
	
void DTLPPatternReco::findAngleConstraints(double & m_min, double & m_max,
					   const  DTSuperLayer * sl, const DTChamber * chamber,
					   ReconstructInSLOrChamber sl_chamber){
  if (sl_chamber == ReconstructInSL) {
    GlobalPoint superLayerPosition = sl->position();//globalpoint and localpoint differ only by a tag
    if(sl->id().superLayer() == 2){
      /*We are in a theta SL: compute the tangent of angle at its
      start and end: the chamber is 260 cm long in z direction */      
      float firstTan =(abs(superLayerPosition.z()) + 150. )/sqrt(superLayerPosition.x() * superLayerPosition.x() +
								      superLayerPosition.y() * superLayerPosition.y());
      float secondTan =(abs(superLayerPosition.z()) - 150.)/sqrt( superLayerPosition.x() * superLayerPosition.x() +
								      superLayerPosition.y() * superLayerPosition.y());
      m_max = firstTan;
      m_min = secondTan;
 
      //cout << "M min: " << m_min << " M max: " << m_max << endl;
    }

    else if(sl->id().superLayer() == 1 || sl->id().superLayer() == 3 ){
      m_min = -1.6;//-1 rad
      m_max =  1.6;// 1 rad 
    }
  }	
  else {     //FIXME placeholder for the 4D (phi supersegment) code
    m_min = -1.6;//-1 rad
    m_max =  1.6;// 1 rad 
  }
  if(debug) cout << "Angular constraints: " << m_min << " < m < " << m_max << endl;

}  

void DTLPPatternReco::printGnuplot( edm::OwnVector<DTSLRecSegment2D> * theResults, const vector<DTRecHit1DPair>& pairs)
{

  string directory = "/afs/cern.ch/user/e/ebusseti/CMSSW_3_1_1/src/RecoLocalMuon/DTSegment/gnuplot/";
  unsigned int num_seg = theResults -> size();
  stringstream file_name;  
  file_name << directory << num_seg << "segs_event" << event_counter << "dati.txt";
  string file_name_str = file_name.str();
  ofstream data (file_name_str.c_str() );
  if (!data.is_open()) cerr << "file could not be opened"<< endl;
  //writing coordinates on dati file
  const DTSuperLayer * sl;
  sl = theDTGeometry -> superLayer( pairs.begin()->wireId().superlayerId() );
  for (vector<DTRecHit1DPair>::const_iterator it = pairs.begin(); it!=pairs.end(); ++it){
    DTWireId theWireId = (it)->wireId();
    const DTLayer * theLayer = (DTLayer*)theDTGeometry->layer(theWireId.layerId());
    //transform the Local position in Layer-rf in a SL local position, or chamber
    pair<const DTRecHit1D*, const DTRecHit1D*> theRecHitPair = (it) -> componentRecHits();
    LocalPoint thePosition;
    //left hit
    thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.first-> localPosition()));
    data <<  thePosition.z() << " "
	 << thePosition.x()  << " "
	 << (double)sqrt(theRecHitPair.first-> localPositionError().xx()) << endl;
    //and right hit
    thePosition = sl -> toLocal(theLayer->toGlobal(theRecHitPair.second->localPosition()));
    data <<  thePosition.z() << " "
	 << thePosition.x()  << " "
	 << (double)sqrt(theRecHitPair.second-> localPositionError().xx()) << endl;
  }
  data.close();
  stringstream macro_name;
  macro_name << directory << num_seg << "segs_event" << event_counter << "macro.gpt";
  string macro_name_str = macro_name.str();
  ofstream macro(macro_name_str.c_str() );
  if (!macro.is_open()) cerr << "file could not be opened"<< endl;
  //writing the macro
  for (unsigned int i =0; i < num_seg; i++){
    stringstream mname;
    stringstream qname;
    stringstream funname;
    mname << "m"<< i;
    qname << "q"<< i;
    funname << "f"<< i;
    macro << qname.str() << "=" << (*theResults)[i].localPosition().x() << ";"
	  << mname.str() << "=" << (*theResults)[i].localDirection().x() << ";"
	  << funname.str() << "(x)=" << "x/ " << mname.str() <<  "-" << qname.str() << "/" << mname.str() << ";"
	  << endl;
  }
  macro << "set yrange [-3:3]; plot \"" << file_name_str << "\" using 2:1:($3 * 4) with xerrorbars";
  for (unsigned int i =0; i < num_seg; i++){
    stringstream funname;
    funname << "f"<< i << "(x)";
    macro << ", " << funname.str();
  }
  macro << endl;    
  macro.close();
}


/*
  void DTLPPatternReco::createSegment(void * theResults, ResultLPAlgo& theAlgoResults, const vector<const DTRecHit1DPair*>& pairPointers,const  DTSuperLayer * sl, const DTChamber * chamber, ReconstructInSLOrChamber sl_chamber ){
  //cout << "Creating 2Dsegment" << endl;
  LocalPoint seg2Dposition( (float)theAlgoResults.qVar, 0. , 0. );

  LocalVector seg2DDirection ((float) theAlgoResults.mVar,  0. , 1.  );//don't know if I need to normalize the vector
  AlgebraicSymMatrix seg2DCovMatrix;
  vector<DTRecHit1D> hits1D;
  for(unsigned int i = 0; i < theAlgoResults.lambdas.size(); i++){
    if(theAlgoResults.lambdas[i]%2) hits1D.push_back( * pairPointers[(theAlgoResults.lambdas[i] - 1)/2]->componentRecHit(DTEnums::Right));
    else hits1D.push_back(* pairPointers[theAlgoResults.lambdas[i]/2]->componentRecHit(DTEnums::Left));
    }
  if(sl_chamber == ReconstructInSL){
    edm::OwnVector<DTSLRecSegment2D> * result = (edm::OwnVector<DTSLRecSegment2D>*)theResults;
    DTSLRecSegment2D * candidate;
    candidate = new DTSLRecSegment2D(sl->id(),seg2Dposition,seg2DDirection, seg2DCovMatrix, theAlgoResults.chi2Var, hits1D);
    cout << "Updating 2D segment" << endl;
    theUpdator->update(candidate);
    result ->push_back(candidate);
  }
  if(sl_chamber == ReconstructInChamber){
    edm::OwnVector<DTChamberRecSegment2D> * result = (edm::OwnVector<DTChamberRecSegment2D>*)theResults;
    DTChamberRecSegment2D * candidate;
    candidate = new DTChamberRecSegment2D(chamber->id(),seg2Dposition,seg2DDirection, seg2DCovMatrix, theAlgoResults.chi2Var, hits1D);   
    cout << "Updating 2D segment" << endl;
    theUpdator->update(candidate);
    result ->push_back(candidate);
  }
}
*/


/*

void * DTLPPatternReco::reconstructSegmentOrSupersegment(const vector<DTRecHit1DPair>& pairs, ReconstructInSLOrChamber sl_chamber ){
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
  vector<const DTRecHit1DPair *> pairPointers;
  for (vector<DTRecHit1DPair>::const_iterator it = pairs.begin(); it != pairs.end(); ++it){
    pairPointers.push_back(&(*it));
  }
  list<double> pz; //positions of hits along z
  list<double> px; //along x (left and right the wires)
  list<double> pex; //errors (sigmas) on x
  ResultLPAlgo theAlgoResults;//datastructure containing all useful results from perform_fit
  //I populate the coordinates lists, in SL or chamber ref frame
  populateCoordinatesLists(pz, px, pex,  sl, chamber, pairPointers, sl_chamber);
 
 //lpAlgo returns true as long as it manages to fit meaningful straight lines
  while(lpAlgorithm(theAlgoResults, pz, px, pex, -10, 10, theMinimumQ, theMaximumQ, theBigM, theDeltaFactor) ){
    counter++;
    cout << "Creating 2Dsegment" << endl;
    //this method populates the result vector with segments
    createSegment(theResults, theAlgoResults, pairPointers, sl, chamber,  sl_chamber);
    //now i update the recsegment
    cout << "2D segment created, removing used hits" << endl;
    //this removes the used hits from the vector of pointers to pairs
    removeUsedHits(theAlgoResults, pairPointers);
    populateCoordinatesLists(pz, px, pex,  sl, chamber, pairPointers, sl_chamber);
    //cout << "Used hits removed" << endl;
    theAlgoResults.lambdas.clear();
    // cout << "Checking if I need to reiterate perform_fit" << endl;
     }
  px.clear();
  pz.clear();
  pex.clear();
  cout << "---------------------------------------------" << endl;
  cout << " WE HAVE FITTED " << counter  << " SEGMENTS." << endl;
cout << "---------------------------------------------" << endl;
 
//to disable the gnuplot macros production, just comment these two lines
//printGnuplot((edm::OwnVector<DTSLRecSegment2D>*)theResults, pairs);
// event_counter++;

  return theResults;
}

*/
