//-------------------------------------------------------------
//
//   Class: DTTracoChip
//
//   Description: Implementation of TRACO
//                trigger algorithm
//
//
//   Author List:
//   SV 4/II/2003
//   Modifications: 
//   22/VI/04 SV : last trigger code update
//   16/I/07  SV : new DTConfig update
//   3/IV/07  SV : setTracoAcceptance moved from card to chip
//   30/10/09 SV : lut parameters from DB are used in code
//   110208 SV   : TRACO hardware bug included
//------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTraco/interface/DTTracoChip.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTTraco/interface/DTTracoCard.h"
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrig.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTTSTheta.h"
#include "L1Trigger/DTTraco/interface/DTTracoCand.h"
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>

//----------------
// Constructors --
//----------------

DTTracoChip::DTTracoChip(DTTracoCard* card, int n, DTConfigTraco* conf) :
                                _card(card), _config(conf) {

  _geom = _card->geom();

  // n=traco number 1,2,...
  if(config()->debug()==4){
    std::cout << "DTTracoChip constructor called for TRACO number " << n << std::endl;
  }

  // set acceptances from CMSSW geometry
  setTracoAcceptances();
  
  // reserve the appropriate amount of space for vectors
  int i=0;
  for(i=0;i<DTConfigTraco::NSTEPL - DTConfigTraco::NSTEPF;i++) {
    _innerCand[i].reserve(DTConfigTraco::NBTITC);
    _outerCand[i].reserve(3*DTConfigTraco::NBTITC);
    _tracotrig[i].reserve(2);
  }
 
  // the identifier
  DTChamberId sid = _geom->statId();
  _id = DTTracoId(sid,n);
 
  // Flags for LTS
  _bxlts.zero();
  for(int is=0;is<DTConfigTraco::NSTEPL-DTConfigTraco::NSTEPF+1;is++){
    _flag[is].zero();
  }

  // debugging
  if(config()->debug()==4){
    std::cout << "CMS position:" << CMSPosition() << std::endl;
    std::cout << " psiRad=" << psiRad() << " KRad=" << KRad() << std::endl;
  }

  //init traco parameters from traco config file
  _krad = config()->KRAD();
  _btic=config()->BTIC(); 

  //offset from geometry (x1-x3 FE view): converted from cm to ST units (0.9999 for rounding)
  _ibtioff=static_cast<int>(config()->BTIC()/(_geom->cellPitch())*(_geom->phiSLOffset()/0.9999));  

  // 091030 SV lut parameters from DB
  // SV 08/12/12 : added flag for computing luts from DB parameters
  if( _card->lutFromDBFlag()==1 )
  {
    //int board = int( (n-1)/4 );
    //int traco = int(fmod( double(n-1),4.));
    // 110208 SV for TRACO hardware bug included SL_shift
    // SL shift
    float xBTI1_3 	= _geom->localPosition( DTBtiId(DTSuperLayerId(sid.wheel(),sid.station(),sid.sector(),3),1) ).x();
    float xBTI1_1 	= _geom->localPosition( DTBtiId(DTSuperLayerId(sid.wheel(),sid.station(),sid.sector(),1),1) ).x();
    float SL_shift 	= xBTI1_3 - xBTI1_1;

    _lutsCCB = new Lut(_card->config_luts(),n,SL_shift);
    _luts = 0;
  }
  else
  //this is always the case with new DTConfig SV 15/I/2007
  //if( config()->trigSetupGeom()==0 ){
  {
    _luts = 0;
    _lutsCCB = 0;
  }
/*
  //SV 21/V/03 for testbeam purpose: parameters from hardware setup
  if(config()->trigSetupGeom()==1){
    //init traco parameters
    _dd=config()->DD();
    _krad=config()->KRAD();
    _btic=config()->BTIC();
    _ibtioff=config()->IBTIOFF();

    //const char* testfile = "traco";   //FIXTB 
    std::string  testfile = "traco_";
    int it = number() - 4 - 1;
    if(it<10)
      testfile += it+'0';
    if(it>9){
      testfile += int(float(it)/10.0) + '0';                //add decimal char
      testfile += int(fmod(float(it),float(10))) + '0';     //add unit char
    }

    //const char* name = testfile;
    if(config()->debug()==4)
      std::cout << "Loading " << testfile << " luts for traco " << number() << std::endl;
    _luts = new DTTracoLUTs(testfile);
    _luts->reset();
    _luts->load();
    if(config()->debug()==4)
      _luts->print();
  }//end hardware setup

  //TB 2004 setup : luts from minicrate ccb equations
  if(config()->trigSetupGeom()==2){
    int board = int( (n-1)/4 );
    int traco = fmod( double(n-1),4. );
    _lutsCCB = new Lut(sid.station(),board,traco);
    // 091030 SV this constructur is obsolete now, use setForTestBeam instead
  }//end TB2004
*/
}


DTTracoChip::DTTracoChip(const DTTracoChip& traco) : 
  _geom(traco._geom), _id(traco._id), _card(traco._card), _luts(traco._luts) {
  int i=0;
  for(i=0;i<DTConfigTraco::NSTEPL - DTConfigTraco::NSTEPF;i++) {
    _innerCand[i].reserve(DTConfigTraco::NBTITC);
    std::vector<DTTracoCand>::const_iterator p;
    for(p=traco._innerCand[i].begin();p<traco._innerCand[i].end();p++) {
      _innerCand[i].push_back(*p);
    }
    _outerCand[i].reserve(3*DTConfigTraco::NBTITC);
    for(p=traco._outerCand[i].begin();p<traco._outerCand[i].end();p++) {
      _outerCand[i].push_back(*p);
    }
    _tracotrig[i].reserve(2);
    std::vector<DTTracoTrig*>::const_iterator p1;
    for(p1=traco._tracotrig[i].begin();p1<traco._tracotrig[i].end();p1++) {
      _tracotrig[i].push_back(*p1);
    }
  }
  _bxlts = traco._bxlts;
  for(int is=0;is<DTConfigTraco::NSTEPL-DTConfigTraco::NSTEPF+1;is++){
    _flag[is] = traco._flag[is];
  }

}


//--------------
// Destructor --
//--------------
DTTracoChip::~DTTracoChip(){
  clear();
  /*
  if(config()->trigSetupGeom()==1){
    _luts->reset();
    delete _luts;
  }

  if(config()->trigSetupGeom()==2)
    delete _lutsCCB;
  */

  if( _card->lutFromDBFlag()==1 )
    delete _lutsCCB;

}

//--------------
// Operations --
//--------------

DTTracoChip&
DTTracoChip::operator=(const DTTracoChip& traco) {
  if(this != &traco){
    _geom = traco._geom;
    _id = traco._id;
    _card = traco._card;
    int i=0;
    for(i=0;i<DTConfigTraco::NSTEPL - DTConfigTraco::NSTEPF;i++) {
      _innerCand[i].reserve(DTConfigTraco::NBTITC);
      std::vector<DTTracoCand>::const_iterator p;
      for(p=traco._innerCand[i].begin();p<traco._innerCand[i].end();p++) {
	_innerCand[i].push_back(*p);
      }
      _outerCand[i].reserve(3*DTConfigTraco::NBTITC);
      for(p=traco._outerCand[i].begin();p<traco._outerCand[i].end();p++) {
	_outerCand[i].push_back(*p);
      }
      _tracotrig[i].reserve(2);
      std::vector<DTTracoTrig*>::const_iterator p1;
      for(p1=traco._tracotrig[i].begin();p1<traco._tracotrig[i].end();p1++) {
	_tracotrig[i].push_back(*p1);
      }
    }
    _bxlts = traco._bxlts;
    for(int is=0;is<DTConfigTraco::NSTEPL-DTConfigTraco::NSTEPF+1;is++){
      _flag[is] = traco._flag[is];
    }
  }
  return *this;
}


void
DTTracoChip::clear() {

  std::vector<DTTracoTrig*>::iterator p1;
  for(int is=0;is<DTConfigTraco::NSTEPL-DTConfigTraco::NSTEPF+1;is++){
    for(p1=_tracotrig[is].begin();p1<_tracotrig[is].end();p1++){
      delete (*p1);
    }
    _tracotrig[is].clear();
    _innerCand[is].clear();
    _outerCand[is].clear();
    _flag[is].zero();
  }
  _bxlts.zero();
}



void
DTTracoChip::run() {

  // Debugging...
  if(config()->debug()>1){
    std::cout << "DTTracoChip::run: Processing TRACO " << _id.traco() << std::endl;
  }
  // End debugging

  int maxtc = static_cast<int>(ceil( float(geom()->nCell(1)) / float(DTConfigTraco::NBTITC) ));

  if( _id.traco()<1 || _id.traco()>maxtc ) {
    if(config()->debug()==4)
      std::cout << "DTTracoChip::run: wrong TRACO number " << _id.traco() << std::endl;
    return;
  }

  // Loop on step
  for(int is=DTConfigTraco::NSTEPF; is<=DTConfigTraco::NSTEPL;is++) {
    if(config()->debug()>1){
      std::cout << "\n STEP: " << is << std::endl;
      std::cout << " ================" << std::endl;
    }

    // skip if no cand. at this step
    if(_innerCand[is-DTConfigTraco::NSTEPF].size()<1 &&
       _outerCand[is-DTConfigTraco::NSTEPF].size()<1 ) 
      continue;

    // Debugging...
    if(config()->debug()==4){
      std::cout << " --> " << _innerCand[is-DTConfigTraco::NSTEPF].size()+
                         _outerCand[is-DTConfigTraco::NSTEPF].size();
      std::cout << " candidates " << std::endl;
    }
    // End debugging

    // Multiple trigger detection between consecutive TRACO's  
    setFlag(is);

    //check if there is a H in bx for LVALIDIFH flag
    if(config()->LVALIDIFH()){
      for(unsigned int e=0; e<_innerCand[is-DTConfigTraco::NSTEPF].size(); e++) {
        if(_innerCand[is-DTConfigTraco::NSTEPF][e].BtiTrig()->code()==8){
          _flag[is-DTConfigTraco::NSTEPF].set(9);
          break;
        }
      }
      for(unsigned int e=0; e<_outerCand[is-DTConfigTraco::NSTEPF].size(); e++) {
        if(_outerCand[is-DTConfigTraco::NSTEPF][e].BtiTrig()->code()==8){
          _flag[is-DTConfigTraco::NSTEPF].set(9);
          break;
        }
      }
    }
   
    // Loop over first/second tracks
    //for(int itk=0; itk < DTConfigTraco::NMAXCAND; itk++){ 
    // FIX this hardcoded 2!! 
    for(int itk=0; itk < 2; itk++){ 
    
      // Get the best inner and outer segments 
      if(config()->debug()==4)
        std::cout << "Inner:" << std::endl;
      DTTracoCand* inner = bestCand(itk,_innerCand[is-DTConfigTraco::NSTEPF]);
      if(config()->debug()==4)
        std::cout << "Outer:" << std::endl;
      DTTracoCand* outer = bestCand(itk,_outerCand[is-DTConfigTraco::NSTEPF]);

      //debug
      if(config()->debug()>1){
        if(inner || outer)
           std::cout<<"Best candidates for track " << itk+1 << " are:"<<std::endl;
        if(inner){ 
          std::cout<<"inner->";
          inner->print();
        }
        if(outer){
          std::cout<<"outer->";
          outer->print();
        }
      }

      // Skip to next step if no suitable candidates found
      if(inner==0&&outer==0)
        break;

      // suppression of LTRIG on BTI close to selected HTRIG
      // SV 24/IX/03 : AND suppression of LTRIG close to H in adiacent Traco
      // SV 31/III/03 : test : only if htprf is off--> NO, it's worse
      // if( config()->prefHtrig(0) && config()->prefHtrig(1) ){
        if(inner){
          DoAdjBtiLts( inner, _innerCand[is-DTConfigTraco::NSTEPF] );
        }
        if(outer){
          DoAdjBtiLts( outer, _outerCand[is-DTConfigTraco::NSTEPF] );
        }
      //}

      // set candidates unusable by further steps
      if(inner)inner->setUsed(); 
      if(outer)outer->setUsed(); 
      // Create a new TRACO trigger with preview for TS
      DTTracoTrig* tct = setPV(itk,inner,outer);

      // skip if no TRACO trigger has been created with this trigger
      if(!tct) break; // something nasty happened. Go to next step

      // try a correlation between segments
      int stored = 0;
      if(inner && outer) {
        stored = storeCorr(tct,inner,outer,itk);
      }

      if (!stored) { 
	// No suitable candidate found or no correlation possible
	// Fill single trigger
	stored = storeUncorr(tct,inner,outer,itk);
      }

      // if trigger has been filled store it in TRACO, otherway delete it
      if (stored) {
	addTrig(is,tct);
      } else {
	delete tct;
        //break;-> II track is computed even if no I track found...
      }

    } // end loop on first/second track

    // Inhibit second track at previous bunch crossing
    if(config()->debug()==4)
      std::cout<<"Checking overlap I-II track..." <<std::endl;
    if(_tracotrig[is-DTConfigTraco::NSTEPF].size()>0 && is>DTConfigTraco::NSTEPF
      && (_tracotrig[is-DTConfigTraco::NSTEPF])[0]->isFirst() ) {    //I track at bx
      if(nTrig(is-1)>0) {                                           //there is a track at bx-1
        if( !(trigger(is-1,1)->isFirst())  ||                       //trig 1 is II track
             ( nTrig(is-1)==2 && !(trigger(is-1,2)->isFirst()) )) { //trig 2 is II track
          raiseOverlap(is);
	  if(config()->debug()==4){
            std::cout << "II track at step " << std::hex << is-1 <<std::dec << "marked rej."<< std::endl;
            std::cout << "I track overlap flag at step " << std::hex << is << std::dec << " setted" << std::endl;
          }
        }
      }
    }
    //debug...
    for(int isd=0;isd<=DTConfigTraco::NSTEPL-DTConfigTraco::NSTEPF+1;isd++)
      if(config()->debug()==4){
        std::cout << "overlap flag step = " << isd+DTConfigTraco::NSTEPF << 
           "  " << _flag[isd].element(1) << std::endl;
      }
    // debugging...
    if(config()->debug()>0){    
      if(nTrig(is)>0) {
	for(int cc=1;cc<=nTrig(is);cc++){
	  trigger(is,cc)->print();
	}
      }
    }// end debugging
  }// end loop on step
}


void
DTTracoChip::raiseOverlap(int step){
    _flag[step-DTConfigTraco::NSTEPF].set(1);                    //overlap flag raised
    _flag[step-DTConfigTraco::NSTEPF-1].set(2);                  //mark II rej.
}


void
DTTracoChip::setFlag(int step, int ext) {

  if(ext==0){
    //this is the original: flags from card
    DTTracoChip* prevTraco = _card->getTRACO(_id.traco()-1);
    if(prevTraco!=0){
      if(prevTraco->edgeBTI(step,1,2))
        _flag[step-DTConfigTraco::NSTEPF].set(3);
      if(prevTraco->edgeBTI(step,2,2))
        _flag[step-DTConfigTraco::NSTEPF].set(5);
    }
    DTTracoChip* nextTraco = _card->getTRACO(_id.traco()+1);
    if(nextTraco!=0){
      if(nextTraco->edgeBTI(step,1,1))
        _flag[step-DTConfigTraco::NSTEPF].set(4);
      if(nextTraco->edgeBTI(step,2,1))
        _flag[step-DTConfigTraco::NSTEPF].set(6);
    }
  }
  else{
    //SV III/03: flags from input EXT: only for testing purpose
    for(int i=0;i<6;i++){
      int ibit = ext >> i;
      if(ibit & 0x01)   // bit i+1 -> flag 3,4,5,6 : IL,IR,OL,OR
        _flag[step-DTConfigTraco::NSTEPF].set(i+1 + 2);
    }
  }

  //debug:
  if(config()->debug()==4){
    std::cout << "Flags set for bx=" << step << std::endl;
    std::cout << _flag[step-DTConfigTraco::NSTEPF].element(1)<< "  ";
    std::cout << _flag[step-DTConfigTraco::NSTEPF].element(2)<< "  ";
    std::cout << _flag[step-DTConfigTraco::NSTEPF].element(3)<< "  ";
    std::cout << _flag[step-DTConfigTraco::NSTEPF].element(4)<< "  ";
    std::cout << _flag[step-DTConfigTraco::NSTEPF].element(5)<< "  ";
    std::cout << _flag[step-DTConfigTraco::NSTEPF].element(6)<< "  ";
    std::cout << _flag[step-DTConfigTraco::NSTEPF].element(7)<< "  ";
    std::cout << _flag[step-DTConfigTraco::NSTEPF].element(8)<< "  ";
    std::cout << _flag[step-DTConfigTraco::NSTEPF].element(9)<< "  "<<std::endl;
  } //end debugging
}


DTTracoCand*
DTTracoChip::bestCand(int itk, std::vector<DTTracoCand> & tclist) {

  // Return if no candidates
  if(tclist.size()<1) return 0;

  // stl function: sort in Ktc ascending or descending order according 
  // to user request comparing by default with user-defined <
  // NB don't reverse if candidates are two with same K
  stable_sort( tclist.begin(),tclist.end() ); //0=K ascending, 1=K descending
  if(config()->sortKascend(itk) &&
     !(tclist.size()==2 && tclist[0].K()==tclist[1].K()) ) {
     reverse( tclist.begin(),tclist.end() );
     if(config()->debug()==4)
       std::cout << "Reversing order of sorted candidate list..." << std::endl;
  }

  /*
  if(!config()->sortKascend(itk)){
    stable_sort( tclist.begin(),tclist.end(),DTTracoCand::closer );
  } else {
    stable_sort( tclist.begin(),tclist.end(),DTTracoCand::wider );
  }
 */

 // debugging...
 if(config()->debug()==4){
    std::cout << "DTTracoChip::findBest - Looking for track number " << itk+1 << std::endl ;
    std::cout << "Sorted std::vector of usable track candidates is:" << std::endl;
    int i = 1;
    for(std::vector<DTTracoCand>::iterator p=tclist.begin();p<tclist.end();p++){
      if((*p).usable()){
        std::cout << " DTTracoChip Candidate # " << i++;
        (*p).print();
      }
    }
    std::cout << "--------------------------------------------------" << std::endl;
  }
  // end debugging

  // return the best candidate
  int i=0;
  DTTracoCand* bestltrig = 0;
  std::vector<DTTracoCand>::iterator p;
  for ( p = tclist.begin(); p < tclist.end(); ++p ) {
    i++;
    // candidate must be usable and not suppressed by LTS
    if(AdjBtiLTSuppressed(&(*p))) 
      if(config()->debug()==4) 
        std::cout << "Candidate # " << i << " supp. because next to H in adiacent Tracos" << std::endl;
    if ( (*p).usable() && !AdjBtiLTSuppressed(&(*p)) ) {
      // check if preference to HTRIG is set and return first trigger
      if( !config()->prefHtrig(itk) ) return &(*p);
      if( (*p).BtiTrig()->code()==8 ) return &(*p);
      if( bestltrig==0 ) bestltrig=&(*p);
    }
  }
  return bestltrig;

}

void
DTTracoChip::DoAdjBtiLts(DTTracoCand* candidate, std::vector<DTTracoCand> & tclist) {
  // If requested, do suppression of LTRIG on BTI close to selected HTRIG in same traco
  //if(!(config()->adjBtiLts()) && candidate->BtiTrig()->code()==8) {
  // SV this is done always, not parametrized !! 
  if(candidate->BtiTrig()->code()==8) {
    std::vector<DTTracoCand>::iterator p;
    for(p=tclist.begin();p<tclist.end();p++){
      if( (*p).BtiTrig()->code()<8 &&
	  abs(       (*p).BtiTrig()->btiNumber() -
	       candidate->BtiTrig()->btiNumber() ) < 2 ) {
	(*p).setUsed();
        if(config()->debug()==4){
          std::cout << "Candidate :";
          (*p).print();
          std::cout << "Suppressed because adiacent to H trig" <<std::endl;
        } // end debug
      } // end if
    } // end candidate loop 
  } // end if H
}

int
DTTracoChip::AdjBtiLTSuppressed(DTTracoCand* candidate) {
  // If requested, do suppression of LTRIG on adjacent BTI -> obsolete!
  //if(!(config()->adjBtiLts()) && candidate->BtiTrig()->code()<8) {
  //SV: Ltrig always suppressed in hardware if Htrig in adj traco!
    if(candidate->BtiTrig()->code()<8) {
      if( _flag[candidate->step()-DTConfigTraco::NSTEPF].element(3) &&
        candidate->position()==1 )                                   return 1;
      if( _flag[candidate->step()-DTConfigTraco::NSTEPF].element(4) &&
        candidate->position()==DTConfigTraco::NBTITC )                return 1;
      if( _flag[candidate->step()-DTConfigTraco::NSTEPF].element(5) &&
        candidate->position()== DTConfigTraco::NBTITC+1)              return 1;
      if( _flag[candidate->step()-DTConfigTraco::NSTEPF].element(6) &&
        candidate->position()==DTConfigTraco::NBTITC*4 )              return 1;
    }
  //}
  return 0;
}

DTTracoTrig*
DTTracoChip::setPV(int itk, DTTracoCand* inner, DTTracoCand* outer) {

  // debugging...
  if(config()->debug()==4){
    std::cout << "DTTracoChip::setPV called for candidates : " << std::endl;
    if(inner)inner->print();
    if(outer)outer->print();
    std::cout << "--------------------------------------------------" << std::endl;
  }
  //end debugging

  // first or second track. This is tricky:
  // itk=0 menas first track  ==> first= true=1
  // itk=1 menas second track ==> first=false=0
  int first = (itk==0) ? 1 : 0;

  //preview selector: the same as priority selector I !!
  // select which of the inner/outer segments should be used

  DTTracoCand* candidate=0;  
  if(inner!=0&&outer!=0) {
//    if(config()->prefHtrig(itk)){   
//    ---> BUG! selection is ALWAYS for H trigs
//    ---> fixed by Sara Vanini 25/III/03
      if(inner->BtiTrig()->code()==8 && outer->BtiTrig()->code()<8 ){
        candidate=inner;
      } else if(inner->BtiTrig()->code()<8  && outer->BtiTrig()->code()==8){
        candidate=outer;
      } else { //for same quality tracks, pref. to in/out selection
        if(!config()->prefInner(itk)) {
          candidate=inner;
        } else {
          candidate=outer;
	}
      }
//    } //end if(config()->prefHtrig...
/*
    else {
      if(!config()->prefInner(itk)) {
	candidate=inner;
      } else {
	candidate=outer;
      }
    }
*/
  } else if(inner==0&&outer!=0) {
    candidate=outer;
  } else if(inner!=0&&outer==0) {
    candidate=inner;
  } else {
    return 0; // no candidates 
  }

  // create new trigger with this candidate
  DTTracoTrig* tct = new DTTracoTrig(this, candidate->step());
  // store preview for TS
  int cod = candidate->BtiTrig()->code();
  if(candidate->BtiTrig()->btiSL()==1) cod *= 10;
  // k output is 5 bits!!! SV
  int K=candidate->K();
  if(K>31)
    K-=32;
  int ioflag = 0;
  if(candidate->position()>4)
    ioflag = 1;
  tct->setPV(first, cod, K, ioflag); // this is already BTI_K-KRAD

  if(config()->debug()==4){
    std::cout << "Selected candidate stored for preview is: ";
    candidate->print();
  }
  return tct;
}

int
DTTracoChip::storeCorr(DTTracoTrig* tctrig, DTTracoCand* inner, DTTracoCand* outer, int tkn) {

  // Bunch crossing
  int is = tctrig->step();

  // Debugging...
  if(config()->debug()==4){
    std::cout << "DTTracoChip::storeCorr called with candidates: " << std::endl;
    if(inner)inner->print();
    if(outer)outer->print();
    std::cout << "--------------------------------------------------" << std::endl;
  }
  // End debugging

  //old orca shift definition
  float shift = 0.;
  //if( config()->trigSetupGeom()!=1 )
    shift = (int)( _geom->distSL()/_geom->cellH() + 0.5 );
  //else
    //shift = DD();
 
  int kcor = 9999;
  int xcor = 0;
  int icor = 0;

  // Check correlation only if --> this cuts LL follow by H in next 4 bx
  // SV 1/IV/04 BUG FIX: this cuts LL preview also, traco outputs preview when LTS cut!!!
  //if( !config()->TcBxLts() ||          // BX LTS is not enabled or
  //    !_bxlts.element(is) ||           // no HTRIG in next 4 BX or
  //    inner->BtiTrig()->code()==8 ||   // inner track is HTRIG  or
  //    outer->BtiTrig()->code()==8 ){   // outer track is HTRIG
  //otherwise in and out trig are L, and necessary one is suppressed for LTS  

    int xq1 = inner->X();
    int xq2 = outer->X();
    xcor = (xq2+xq1)/2;
    kcor = (xq1-xq2)+512;
    int kq1 = int(shift/2.) * (inner->BtiTrig()->K()-BTIC()) + 512;
    int kq2 = int(shift/2.) * (outer->BtiTrig()->K()-BTIC()) + 512;
    //int kd1 = abs(kcor/16-kq1/16);
    //int kd2 = abs(kcor/16-kq2/16);
    int kd1 = abs(kcor/16-kq1/16);
    int kd2 = abs(kcor/16-kq2/16);
  
    icor =  kd1<=config()->TcKToll(tkn) && 
            kd2<=config()->TcKToll(tkn) && 
            xcor>0;
 
    // Debugging...
    if(config()->debug()==4){
      std::cout << "*************************************************************";
      std::cout << std::endl;
      std::cout << " shift = " << shift;
      std::cout << " xq1 = " << xq1;
      std::cout << " xq2 = " << xq2;
      std::cout << " xcor = " << xcor;
      std::cout << " kcor = " << kcor;
      std::cout << " kq1 = " << kq1;
      std::cout << " kq2 = " << kq2;
      std::cout << " kd1 = " << kd1;
      std::cout << " kd2 = " << kd2;
      std::cout << " icor = " << icor;
      std::cout << std::endl;
      std::cout << "*************************************************************";
      std::cout << std::endl;
    }// End debugging

  //}//end if TcBxLts....

  if(icor){
    // correlation was successfull
    // set the preview correlation bit. It isn't reset if outside ang. window
    tctrig->setPVCorr(1);
    // set K and X
    tctrig->setK(kcor - 512);
    //std::cout<<"Set K " << kcor << " instead of " << kcor-512 << std::endl;
    //tctrig->setK(kcor);
    tctrig->setX(xcor);
    // set codes
    tctrig->setCodeIn( inner->BtiTrig()->code());
    tctrig->setCodeOut(outer->BtiTrig()->code());
    // set position mask
    //tctrig->setPosIn(inner->position());
    //tctrig->setPosOut(outer->position());
    //SV number of bti instead of position...
    tctrig->setPosIn( inner->BtiTrig()->btiNumber() );
    tctrig->setPosOut( outer->BtiTrig()->btiNumber() );
    //SV store also equation: pattern numbers are 1-32
    tctrig->setEqIn( inner->BtiTrig()->eq() + 1 ); 
    tctrig->setEqOut( outer->BtiTrig()->eq() + 1 );

    // calculate psi, psi_r and Delta(psi_r)
    calculateAngles(tctrig);
    // check angular window for LL  --> fixed by SV 27/III/03 --> NO, for all!
    //if( (tctrig->qdec()==4) && !insideAngWindow(tctrig)) {
    if( !insideAngWindow(tctrig) ) {
       // reset codes, K, X and angles
        tctrig->resetVar() ;
    }     
    //SV 1/IV/04 BUG FIX: check LTS after angle cut...
    else if( tctrig->qdec()==4   &&          // cut only LL
             config()->TcBxLts() ){          // BX LTS is  enabled or
        // reset codes, K, X and angles
        if(tkn==0 && _bxlts.element(is) )      // I track : there is H -4,+1
          tctrig->resetVar() ;
        if(tkn==1 && _bxlts.element(is+1) )    // II track : there is H -4,+1 1 bx later
          tctrig->resetVar() ;
    }
    else {
      // set links to BTI triggers
      tctrig->addDTBtiTrig(inner->BtiTrig());
      tctrig->addDTBtiTrig(outer->BtiTrig());
    }

    // Debugging...
    if(config()->debug()>1){
      std::cout << "*************************************************************";
      std::cout << std::endl;
      std::cout << "               Correlation was successfull:                  ";
      std::cout << std::endl;
      std::cout << " Code = " << tctrig->code();
      std::cout << " K = " << tctrig->K();
      std::cout << " X = " << tctrig->X();
      std::cout << std::endl;
      std::cout << "*************************************************************";
      std::cout << std::endl;
    }
    // End debugging

  } else {

    // Debugging...
    if(config()->debug()>1){
      std::cout << "*************************************************************";
      std::cout << std::endl;
      std::cout << "               No correlation possible                       ";
      std::cout << std::endl;
      std::cout << "*************************************************************";
      std::cout << std::endl;
    }
    // End debugging

  }

  return icor;
}

int
DTTracoChip::storeUncorr(DTTracoTrig* tctrig, DTTracoCand* inner, DTTracoCand* outer, int tkn) {

  // Bunch crossing
  int is = tctrig->step();

  // Debugging...
  if(config()->debug()==4){
    std::cout << "DTTracoChip::storeUncorr called with candidates: " << std::endl;
    if(inner)inner->print();
    if(outer)outer->print();
    std::cout << "--------------------------------------------------" << std::endl;
  }

  // End debugging
  // priority selector 
  // select which of the inner/outer segments should be used
  // allow re-use of other segment according to configuration
  DTTracoCand* candidate=0;  
  if(inner!=0&&outer!=0) {
//    if(config()->prefHtrig(tkn)){  
// --> BUG: selector I preference is ALWAYS for H trig
// fixed by Sara Vanini - 25/III/03
      if(inner->BtiTrig()->code()==8 && outer->BtiTrig()->code()<8 ){
        candidate=inner;
        //if(config()->TcReuse(1)) outer->setUnused(); // reusable
      } else if(inner->BtiTrig()->code()<8  && outer->BtiTrig()->code()==8){
        candidate=outer;
        //if(config()->TcReuse(0)) inner->setUnused(); // reusable
      } else { //for the same quality triggers:
        if(!config()->prefInner(tkn)) {
          candidate=inner;
          //if(config()->TcReuse(1))  outer->setUnused(); // reusable
        } else {
          candidate=outer;
          //if(config()->TcReuse(0))  inner->setUnused(); // reusable
	}
      }//end else
/*
    } else {//no Htrig preference
      if(!config()->prefInner(tkn)) {
	candidate=inner;
	if(config()->TcReuse(1))  outer->setUnused(); // reusable
      } else {
	candidate=outer;
	if(config()->TcReuse(0))  inner->setUnused(); // reusable
      }
    }
*/
  } else if(inner==0&&outer!=0) {
    candidate=outer;
  } else if(inner!=0&&outer==0) {
    candidate=inner;
  } else {
    return 0; // no candidates 
  }

  //SV *** FOR TESTBEAM OR TEST BENCH PURPOSE ***
  //theta trigger bin present(1) or absent(0)
  //int thTr = (_flag[is-DTConfigTraco::NSTEPF].element(7)) ?
  //   _flag[is-DTConfigTraco::NSTEPF].element(7):
  //   _flag[is-DTConfigTraco::NSTEPF].element(8);

  // priority selector II: accept or discard candidate according to masks:
  // ** LOW TRIGGERS
  if( candidate->BtiTrig()->code()<8 ) { 
    // first check with LVALIDIFH: if 1, accept low if there is a H in traco at bx
    if(config()->LVALIDIFH() && _flag[is-DTConfigTraco::NSTEPF].element(9)){ 
      if(config()->debug()>1)
        std::cout << "Low accepted because LVALIDIFH on...." << std::endl;
    }
    else {//LVALIDIFH==0 or there isn't H in traco in bx : check theta!
      //theta check
      if( !config()->singleLenab(tkn) ) {
        // LTF: single LTRIG not always en. Check cond.:
        if( config()->singleLflag(tkn)==1 ||      //always discarded
	    ( config()->singleLflag(tkn)==2 && !(_card->TSTh()->nHTrig(is)) ) ||
	    ( config()->singleLflag(tkn)==0 && !(_card->TSTh()->nTrig(is))  ) ){ 
// SV --> for TESTS version
//        config()->singleLflag(tkn)==0 && thTr==0 ||   //only with theta trig. 
//        config()->singleLflag(tkn)==2 && thTr==0  ){  //only with theta H trig (not hw)
        if(config()->debug()>1)
          std::cout << "Single low trigger discarded by preview and "
               << "priority selector for ltmsk!" <<std::endl;
        return 0;}
        //       ^-------- trigger is suppressed and will not be stored
      }//end theta

    } //end else
    //REUSE : mark candidates reusable HERE! SV BUG FIX 6IV04
    if(candidate==inner && config()->TcReuse(1) && outer)
      outer->setUnused();
    if(candidate==outer && config()->TcReuse(0) && inner)
      inner->setUnused();

    // LTS suppression
    if(config()->TcBxLts()){
      if( (tkn==0 && _bxlts.element(is))       // I track : there is H -4,+1
                        ||
          (tkn==1 && _bxlts.element(is+1)) ){  // II track : there is H -4,+1 1 bx later
        tctrig->resetVar() ;
        if(config()->debug()>1)
          std::cout << "Low trigger suppressed because H in next 4 bx "<<
               " and LTS flag on...." << std::endl;
        return 1; // trigger is suppressed but preview will be stored
      }
    }//end lts

//    } //end else
  } //Low trigs

  // Preview Htmsk not implemented: awaiting decision 
  // --> implemented in priority selector by SV
  else { // HTRIG
    //if(config()->singleHflag(tkn)==1 && thTr==0 )  //this is for testing
    if( config()->singleHflag(tkn)==1 && !(_card->TSTh()->nTrig(is) ) )
      return 0;
      // ^-----trigger is suppressed and will not be stored

    //mark candidates reusable HERE! SV BUG FIX 6IV04
    if(candidate==inner && config()->TcReuse(1) && outer)
      outer->setUnused();
    if(candidate==outer && config()->TcReuse(0) && inner)
      inner->setUnused();
  }

  // set code, position, K and X  
  float shift;
  //if(config()->trigSetupGeom()!=1 )
    shift = (int)( _geom->distSL()/_geom->cellH() + 0.5 );
  //else 
    //shift = DD();  //SV 19/III/03
  int kucor = (int)( 0.5*shift * (candidate->BtiTrig()->K()-BTIC()) );
  tctrig->setK(kucor);
  tctrig->setX( candidate->X() );
  // correlation wasn't successfull
  // set the preview correlation bit. 
  tctrig->setPVCorr(0);
  if(candidate->BtiTrig()->btiSL()==1){    // inner track
    tctrig->setCodeIn(candidate->BtiTrig()->code());
    tctrig->setCodeOut(0);
    //tctrig->setPosIn(candidate->position());
    //SV number of bti instead of position...
    tctrig->setPosIn(candidate->BtiTrig()->btiNumber() );
    tctrig->setPosOut(0);
    //SV store also equation
    tctrig->setEqIn( candidate->BtiTrig()->eq() + 1 );
    tctrig->setEqOut( 0 );
  } else {                                               // outer track
    tctrig->setCodeIn(0);
    tctrig->setCodeOut(candidate->BtiTrig()->code());
    tctrig->setPosIn(0);
    //SV number of bti instead of position...
    tctrig->setPosOut(candidate->BtiTrig()->btiNumber() );
    //tctrig->setPosOut(candidate->position());
    //SV store also equation
    tctrig->setEqIn( 0 );
    tctrig->setEqOut( candidate->BtiTrig()->eq() + 1);
  }

  // coordinate converter LUT
  // calculate psi, psi_r and Delta(psi_r)
  calculateAngles(tctrig);
  // check angular window only for Low!!  --> fixed SV 27/III/03--> NO, for all!
  //if( candidate->BtiTrig()->code() < 8 && !insideAngWindow(tctrig) ){
  if( !insideAngWindow(tctrig) ){
    // reset codes, K, X and angles
    tctrig->resetVar() ;
    if(config()->debug()>1)
      std::cout << "L rejected because outside angular window!" << std::endl;
   } else {
    // set links to BTI trigger
    tctrig->addDTBtiTrig(candidate->BtiTrig());
  }

  // Debugging...
  if(config()->debug()>1){
    std::cout << "*************************************************************";
    std::cout << std::endl;
    std::cout << "               Single trigger stored:                        ";
    std::cout << std::endl;
    std::cout << " Code = " << tctrig->code();
    std::cout << " K = " << tctrig->K();
    std::cout << " X = " << tctrig->X();
    std::cout << std::endl;
    std::cout << "*************************************************************";
    std::cout << std::endl;
  }
  // End debugging

  return 1;

}

void
DTTracoChip::add_btiT(int step, int pos, const DTBtiTrigData* btitrig) {

  if(pos<1 || pos>4*DTConfigTraco::NBTITC) {
    std::cout << "DTTracoChip::add_btiT: wrong position: " << pos;
    std::cout << "trigger not added!" << std::endl;
    return;
  }
  if(step<DTConfigTraco::NSTEPF||step>DTConfigTraco::NSTEPL){
    std::cout << "DTTracoChip::add_btiT: step out of range: " << step;
    std::cout << "trigger not added!" << std::endl;
    return;
  }

  if(!config()->usedBti(pos)) {
    if(config()->debug()==4){
      std::cout << "DTTracoChip::add_btiT: position: " << pos;
      std::cout << "has disconnected bti" << std::endl;
    }
    return;
  }


  // 091103 SV: acceptances are taken from geometry if useAcceptParam()=false
  // otherwise cuts based on LL,LH,CL,CH,RL,RH taken from  configuration are applied in TracoCard::loadTraco 
  if(_card->useAcceptParamFlag()==false) {
    // check K inside acceptance
    if(btitrig->K()<_PSIMIN[pos-1] || btitrig->K()>_PSIMAX[pos-1] ) {
      if(config()->debug()>1){
        std::cout << "In TRACO num. " << number() << " BTI trig. in pos " << pos << " outside K acceptance (";
        std::cout << _PSIMIN[pos-1] << "-->";
        std::cout << _PSIMAX[pos-1] << ") - Not added" << std::endl;
      }
      return;
    }
  } 

  // Store trigger candidate
  if(pos<=DTConfigTraco::NBTITC){
    _innerCand[step-DTConfigTraco::NSTEPF].push_back(
      DTTracoCand(this,btitrig,pos,step));
  } else {
    _outerCand[step-DTConfigTraco::NSTEPF].push_back(
      DTTracoCand(this,btitrig,pos,step));
  }

  // Fill array for BX LTS
  if(btitrig->code()==8){
    for(int is=step-4;is<step;is++){       // set flag for 4 previous BX
      if(is>0&&is<=DTConfigTraco::NSTEPL) _bxlts.set(is);
    }
    //SV 1/IV/04 BUG FIX
    _bxlts.set(step+1);
    // Debugging
    if(config()->debug()==4)
      for(int is=0;is<DTConfigTraco::NSTEPL;is++)
       std::cout<<"_bxlts["<<is<<"]="<<_bxlts.element(is)<<std::endl;
  }

  // Debugging
  if(config()->debug()>1){
    std::cout << "BTI Trigger added at step " << step;
    std::cout << " to TRACO " << _id.traco() << " at position " << pos << std::endl;
    btitrig->print();
  } // End debugging

}


void
DTTracoChip::addTrig(int step, DTTracoTrig* tctrig) {
  if(step<DTConfigTraco::NSTEPF||step>DTConfigTraco::NSTEPL){
    std::cout << "DTTracoChip::addTrig: step out of range: " << step;
    std::cout << " trigger not added!" << std::endl;
    return;
  }
  _tracotrig[step-DTConfigTraco::NSTEPF].push_back(tctrig);

  // Debugging...

  if(config()->debug()==4){
    std::cout << "DTTracoChip::addTrig: adding trigger:"<< std::endl; 
    tctrig->print();
  }
  // End debugging

}

int
DTTracoChip::nTrig(int step) const {
  if(step<DTConfigTraco::NSTEPF||step>DTConfigTraco::NSTEPL){
    std::cout << "DTTracoChip::nTrig: step out of range: " << step;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }
  return _tracotrig[step-DTConfigTraco::NSTEPF].size();
}

DTTracoTrig*
DTTracoChip::trigger(int step, unsigned n) const {
  if(step<DTConfigTraco::NSTEPF||step>DTConfigTraco::NSTEPL){
    std::cout << "DTTracoChip::trigger: step out of range: " << step;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  if(n<1 || n>_tracotrig[step-DTConfigTraco::NSTEPF].size()) {
    std::cout << "DTTracoChip::trigger: requested trigger doesn't exist: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  std::vector<DTTracoTrig*>::const_iterator p = 
    _tracotrig[step-DTConfigTraco::NSTEPF].begin()+n-1;
  return *p;
}

DTTracoTrigData
DTTracoChip::triggerData(int step, unsigned n) const {
  if(step<DTConfigTraco::NSTEPF||step>DTConfigTraco::NSTEPL){
    std::cout << "DTTracoChip::triggerData: step out of range: " << step;
    std::cout << " dummy trigger returned!" << std::endl;
    return DTTracoTrigData();
  }
  if(n<1 || n>_tracotrig[step-DTConfigTraco::NSTEPF].size()) {
    std::cout << "DTTracoChip::trigger: requested trigger doesn't exist: " << n;
    std::cout << " dummy trigger returned!" << std::endl;
    return DTTracoTrigData();
  }
  std::vector<DTTracoTrig*>::const_iterator p = 
    _tracotrig[step-DTConfigTraco::NSTEPF].begin()+n-1;
  return (*p)->data();
}

float
DTTracoChip::psiRad(int sl) const {
/*
  // Radial angle of correlator center in mrad in CMS frame
  LocalPoint p = localPosition();
  float x = p.x();
  float y = p.y();
  float z = p.z();
  if        (sl==1) {
    z -= 0.5 * _geom->distSL();
  } else if (sl==3) {
    z += 0.5 * _geom->distSL();
  }
  float fpsir = _geom->stat()->toGlobal(LocalPoint(x,y,z)).phi()-
                _geom->phiCh();
  if(fpsir<-M_PI)fpsir+=M_PI*2;
  if(fpsir>M_PI)fpsir-=M_PI*2;
  return fpsir*1000;
*/
  return 0.0;
}

int
DTTracoChip::KRad() const {
  // K parameter of the radial angle of correlator center
  //float distp2 = (int)(2*_geom->cellH()*config()->ST()/_geom->cellPitch());
  //return -(int)(tan(psiRad(sl)/1000)*distp2); // sign is reverted!!!
  //return _krad;
  
  //SV V/03: for harware bug in traco....
  int KRad=0;
  return KRad;
   
}

int
DTTracoChip::useSecondTrack(int step) const {
  // return 1 if II track use is allow
  // return 0 if II track has been rejected
  if(step<DTConfigTraco::NSTEPF||step>DTConfigTraco::NSTEPL){
    std::cout << "DTTracoChip::useSecondTrack: step out of range: " << step;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }
  return !(_flag[step-DTConfigTraco::NSTEPF].element(2));
}

int
DTTracoChip::edgeBTI(int step, int io, int lr) const {
  if(step<DTConfigTraco::NSTEPF||step>DTConfigTraco::NSTEPL){
    std::cout << "DTTracoChip::edgeBTI: step out of range: " << step;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }
  //
  // inner supl ==> io=1; outer supl ==> io=2      |21   |     |   22|
  // right edge ==> rl=1;  left edge ==> rl=2            |11 12|
  //
  std::vector<DTTracoCand>::const_iterator p;
  if(io==1){
    if(_innerCand[step-DTConfigTraco::NSTEPF].size()>0) {
      // SV 24/IX/03 fix: only HTRIG accepted
      for(p=_innerCand[step-DTConfigTraco::NSTEPF].begin();
	  p<_innerCand[step-DTConfigTraco::NSTEPF].end(); p++){
	if(lr==1 && (*p).position()==1 && (*p).BtiTrig()->code()==8 ) 
          return  1; 
	if(lr==2 && (*p).position()==DTConfigTraco::NBTITC && (*p).BtiTrig()->code()==8 )
          return  1; 
      }
    }
  } else {
    if(_outerCand[step-DTConfigTraco::NSTEPF].size()>0) {
      for(p=_outerCand[step-DTConfigTraco::NSTEPF].begin();
	  p<_outerCand[step-DTConfigTraco::NSTEPF].end(); p++){
	//SV: is the following correct???FIX if using _card to set _flag
        //if(lr==1 && (*p).position()==DTConfigTraco::NBTITC+1)return 1; //or pos=8??
	//if(lr==2 && (*p).position()==DTConfigTraco::NBTITC*4)return 1; //or pos=13?? 
        //SV 24/IX/03 fix 
        if(lr==1 && (*p).position()==DTConfigTraco::NBTITC*3+1 && (*p).BtiTrig()->code()==8 )
          return  1; 
	if(lr==2 && (*p).position()==DTConfigTraco::NBTITC*2   && (*p).BtiTrig()->code()==8 )
          return  1;  
      }
    }
  }
  return  0;
}

void 
DTTracoChip::calculateAngles(DTTracoTrig* tct) {

  int ipsi=0;
  int iphir=0;
  int idpsir=0; 
/* obsolete  
  //TB 2004 luts formula from minicrate CCB
  if( config()->trigSetupGeom()==2 ){
    ipsi = _lutsCCB->get_k( (tct->K()+511) );

    int flag = 0;
    int qual=tct->data().qdec();
    if(qual==3 || qual==1)                //case 0:outer
      flag=0;
    if(qual==2 || qual==0)                //case 1:inner
      flag=1;
    if(qual==6 || qual==5 || qual==4)     //case 2:correlated
      flag=2;

    iphir = _lutsCCB->get_x( (tct->X()+512*flag) );

    idpsir = ipsi - iphir/8;
  }

 //TB 2003 luts data format
 if( config()->trigSetupGeom()==1 ){
    //compute bending angles of traco output with lookup tables
    //SV TB2003: first trigger board isn't connected;
    ipsi = _luts->getPsi(tct->K());  
    int flag = 0;
    int qual=tct->data().qdec();
    if(qual==3 || qual==1)                //case 0:outer
      flag=0;  
    if(qual==2 || qual==0)                //case 1:inner
      flag=1;  
    if(qual==6 || qual==5 || qual==4)     //case 2:correlated
      flag=2;  
    iphir = _luts->getPhiRad( tct->X(), flag);
    idpsir = _luts->getBendAng( tct->X(), tct->K(), flag);
  }
 */

  // 091030 SV angles computed from DB lut parameters
  if( _card->lutFromDBFlag()==1 )
  {
    ipsi = _lutsCCB->get_k( (tct->K()+512) );

    int flag = 0;
    int qual=tct->data().qdec();
    if(qual==3 || qual==1)                //case 0:outer
      flag=0;
    if(qual==2 || qual==0)                //case 1:inner
      flag=1;
    if(qual==6 || qual==5 || qual==4)     //case 2:correlated
      flag=2;

    iphir = _lutsCCB->get_x( (tct->X()+512*flag) );

    idpsir = ipsi - iphir/8;
  }
  else
  // compute angles from CMSSW geometry 
  //if( config()->trigSetupGeom()==0 )
  {
    DTTracoTrigData td = tct->data();
    // psi
    //  float fpsi = atan( (float)(tct->K()) * _geom->cellPitch() / 
    //		     (_geom->distSL() * config()->ST()) );
    float fpsi = atan( _card->localDirection(&td).x() /   // e.g. x>0 and
                     _card->localDirection(&td).z() );    //      z<0 => fpsi<0

    // Change sign in case of wheel<0 or wheel==0 and station == 1,4,5,8,9,12
    int mywh = tct->ChamberId().wheel();
    if (mywh<0   ||
	(mywh==0 && (tct->ChamberId().sector()%4)<2))
      fpsi = -fpsi;
    
    fpsi*=DTConfigTraco::RESOLPSI;
    if(fpsi<=0)
      fpsi-=1.0;
    ipsi = (int)fpsi;
    // if outside range set to lower edge
    if( ipsi>= DTConfigTraco::RESOLPSI || ipsi< -DTConfigTraco::RESOLPSI ) 
      ipsi=-DTConfigTraco::RESOLPSI;


    // psi_r
    float fpsir = _card->CMSPosition(&td).phi()-_geom->phiCh();

    if(fpsir<-M_PI)
      fpsir+=M_PI*2;
    if(fpsir>M_PI)
      fpsir-=M_PI*2;
    fpsir*=DTConfigTraco::RESOLPSIR;
    if(fpsir<=0)
      fpsir-=1.0;
    iphir = (int)fpsir;
    // if outside range set to lower edge
    if( iphir>= DTConfigTraco::RESOLPSIR/2 || iphir <-DTConfigTraco::RESOLPSIR/2 ) 
      iphir=-DTConfigTraco::RESOLPSIR/2;

    // Delta(psi_r)
    int dpsir = (iphir*DTConfigTraco::RESOLPSI) / DTConfigTraco::RESOLPSIR;
    idpsir = ipsi-dpsir;
    // if outside range set to lower edge
    if(idpsir>= DTConfigTraco::RESOLPSI || idpsir <-DTConfigTraco::RESOLPSI ) 
      idpsir=-DTConfigTraco::RESOLPSI;
  }
  
  tct->setAngles(ipsi,iphir,idpsir);

  // debugging
  if(config()->debug()==4){
    std::cout << "DTTracoChip::calculateAngles :" << std::endl;
    tct->print();
    std::cout << std::dec << "K = " << tct->K() << " X = " << tct->X(); 
    std::cout << " ipsi = " << ipsi << " iphir = " << iphir;
    std::cout << " idpsir = " << idpsir << std::endl;
    if( _card->lutFromDBFlag()==1 )
      std::cout << "Angles are calculated from LUT parameters from DB!" << std::endl; 
  }// end debugging

}

int
DTTracoChip::insideAngWindow(DTTracoTrig* tctrig) const {
  // check angular window for every station type
  // return 1 for accept, 0 for reject
  // abs value of bending angle is 9 bits

  BitArray<10> bendAng;
  bendAng.assign(0,10,tctrig->DeltaPsiR());
  //bendAng.print();
  if(bendAng.element(9))  //negativo!
    bendAng.twoComplement(); 
  int bendAngINT = bendAng.read(0,9);
  //std::cout<<"abs bend angle int ="<< bendAngINT <<std::endl;

  if( config()->BendingAngleCut()!= -1 && 
     bendAngINT > 2*(config()->BendingAngleCut())) {
    int absBendAng = tctrig->DeltaPsiR() & 0x1FF;
    if(config()->debug()==4)
      std::cout << "Attention: abs(bendAng)=" << absBendAng << " > " 
           << 2*config()->BendingAngleCut() <<"!! reject trigger"<<std::endl;
    return 0;
  }
  return 1;
}

 
void 
DTTracoChip::setTracoAcceptances()
{  
  // Set K acceptances of DTTracoChip MT ports: Ktraco = Xinner - Xouter 
  float h = _geom->cellH();
  float pitch = _geom->cellPitch();
  float distsl = _geom->distSL();
  float K0 = config()->BTIC();
  float shiftSL = _geom->phiSLOffset() / pitch * K0;

  // mt  ports from orca geometry: this is always the case with new DTConfig
  //if(config_traco(tracoid)->trigSetupGeom() != 1){
  {
    // Master Plane
    int i = 0;
    for(i=0;i<DTConfig::NBTITC;i++){
      float Xin_min     =  (i + DTConfig::NBTITC) * K0 + shiftSL;
      float Xin_max     =  Xin_min + K0;
      float Xout_min    =  0;
      float Xout_max    =  3 * DTConfig::NBTITC * K0;
      _PSIMAX[i]  =  int( 2.*h/distsl * (Xin_max - Xout_min) + K0 + 1.01 );
      _PSIMIN[i]  =  int( 2.*h/distsl * (Xin_min - Xout_max) + K0 );
    }

    // Slave Plane
    for(i=0;i<3*DTConfig::NBTITC;i++){
      float Xin_min     =  (DTConfig::NBTITC) * K0 + shiftSL;
      float Xin_max     =  2. * DTConfig::NBTITC * K0 + shiftSL;
      float Xout_min    =  i * K0;
      float Xout_max    =  Xout_min + K0;
      _PSIMAX[DTConfig::NBTITC+i]  =  int( 2.*h/distsl * (Xin_max - Xout_min) + K0 + 1.01 );
      _PSIMIN[DTConfig::NBTITC+i]  =  int( 2.*h/distsl * (Xin_min - Xout_max) + K0 );
    }
  }


  // debugging
  if(config()->debug()==4){
    //if(wheel()==2&&station()==3&&sector()==1){ // only 1 chamber
      std::cout << "Acceptance of mt ports for offset (cell unit) " 
           << _geom->phiSLOffset() / pitch << std::endl;
      for(int i=0;i<4*DTConfig::NBTITC;i++){
        std::cout << "Port " << i+1 << " : ";
        std::cout << _PSIMIN[i] << " --> " << _PSIMAX[i] << std::endl;
      }
    //}
  }// end debugging

}



