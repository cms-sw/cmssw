//-------------------------------------------------
//
//   Class: DTBtiChip
//
//   Description: Implementation of DTBtiChip 
//                trigger algorithm
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   S. Vanini
//   30/IX/03 SV : wire dead time = ST added
//   22/VI/04 SV : last trigger code update
//   15/I/07  SV : new DTConfig setup
//   17/III/07 SV : distp2 truncation bug fixed 
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTBti/interface/DTBtiChip.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTBti/interface/DTBtiHit.h"
#include "L1Trigger/DTBti/interface/DTBtiTrig.h"

#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <DataFormats/MuonDetId/interface/DTSuperLayerId.h>
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <cmath>

using namespace std;

//----------------
// Constructors --
//----------------

DTBtiChip::DTBtiChip(DTBtiCard* card, DTTrigGeom* geom, int supl, int n, DTConfigBti* conf): _card(card), _geom(geom), _config(conf) {


 // original constructor
 setSnap();
 reSumSet(); 

 // Debugging...
  if(config()->debug()>2){
    cout << "DTBtiChip constructor called for BTI number " << n;
    cout << " in superlayer " << supl << endl;
  }

  // reserve the appropriate amount of space for vectors
  for(auto& t: _trigs) {
    t.reserve(2);
  }

  for(auto& d: _digis) {
    d.reserve(10);
  }
  for(auto& h: _hits) {
    h.reserve(10);
  }

  //SV wire dead time init
  int DEAD = config()->DEADpar();
  for(auto& c: _busyStart_clock) {
    c = - DEAD -1;
  }

  // Identifier
  DTChamberId sid = _geom->statId();
  _id = DTBtiId(sid, supl, n);

  //if(config()->trigSetupGeom() == 0){
    // set K acceptance in theta view for this BTI: 6 bit resolution....  
    _MinKAcc = 0;
    _MaxKAcc = 63;
	
/*	   DTBtiId _id1 = DTBtiId(sid,supl,1);

	   cout <<"superlayer" << _id.superlayer()<< "BTI1   " <<  _id1.bti()  << " BTICur " << _id.bti()<< endl;
	   cout <<endl;
	   GlobalPoint gp1 = _geom->CMSPosition(_id1);
	   cout << "pos of BTI "<<  _id1.bti()  << gp1 <<endl;
	   // K of tracks from vertex
	   GlobalPoint gp = CMSPosition();
	   cout << "pos of BTI" << _id.bti()  << gp <<endl;	   
	   cout << endl ; */
	   
	   
//     // theta bti acceptance cut is in bti chip (no traco in theta!)
//     // acceptance from orca geom: bti theta angle in CMS frame +-2 in K units 
//     if(_id.superlayer()==2){
//       float distp2 = (int)(2*_geom->cellH()*config()->ST()/_geom->cellPitch());
//       float K0 = config()->ST();

// /*      DTBtiId _id1 = DTBtiId(sid,supl,1);
      
//       cout << "BTI1   " <<  _id1.bti() << endl;
//       cout << "BTICur " << _id.bti() <<endl;
//       GlobalPoint gp1 = _geom->CMSPosition(_id1);
//       cout << "pos of BTI 1 " << gp1 <<endl;*/
	  
//       // K of tracks from vertex
//       GlobalPoint gp = CMSPosition();
//       if(config()->debug()>3){
//         cout << "Position: R=" << gp.perp() << "cm, Phi=" << gp.phi()*180/3.14159;
//         cout << " deg, Z=" << gp.z() << " cm" << endl;
//       }
//       // CB TEST WITH NEW GEOMETRY
//       // new geometry: modified wrt old due to specularity of theta SLs (still to understand on wheel zero) 19/06/06
//       float theta;
//       if (_id.wheel()==0){ 
// 	if(_id.sector()%4>1) theta = atan( gp.z()/gp.perp() );
// 	else theta = atan( -gp.z()/gp.perp() );
//       }
//       else theta = atan( fabs(gp.z())/gp.perp() );
//       // .11 =TAN(6.3 deg) ==> k=2 (e' ancora vero? forse questa parte va aggiornata sena ripassare per gli angoli) 19/6/06
//       float thetamin = theta-config()->KAccTheta()*0.055;
//       float thetamax = theta+config()->KAccTheta()*0.055;

//       float fktmin = tan(thetamin)*distp2 + K0;
//       int ktmin = (fktmin>0) ? (int)(fktmin+0.5) : (int)(fktmin-0.5);
//       float fktmax = tan(thetamax)*distp2 + K0;
//       int ktmax = (fktmax>0) ? (int)(fktmax+0.5) : (int)(fktmax-0.5);
// //      float fkbti = -gp.z()/gp.perp()*distp2;
// //      int kbti = (fkbti>0) ? (int)(fkbti+0.5) : (int)(fkbti-0.5);
// //      // K acceptance to point to vertex
// //      int ktmin = kbti-config()->KAccTheta();  // minimum
// //      int ktmax = kbti+config()->KAccTheta();  // maximum
//       if(ktmin>_MinKAcc)_MinKAcc=ktmin;
//       if(ktmax<_MaxKAcc)_MaxKAcc=ktmax;
//     }

//     // debugging
//     if(config()->debug()>2){
//       cout << "CMS position:" << CMSPosition() << endl;
//       cout << "K acceptance:" << _MinKAcc << "," << _MaxKAcc << endl;
//     }
//     // end debugging
// theta bti acceptance cut is in bti chip (no traco in theta!)
    // acceptance is determined about BTI angle wrt vertex with programmable value 
    if(_id.superlayer()==2){
      // 091105 SV theta bti trigger angular acceptance in CMSSW is computed from geometry 
      // (theta of the bti) +- a given tolerance config()->KAccTheta(): tolerance NOT in 
      // hardware configuration. The hw tolerance is given in the system and the 
      // overall acceptance is computed "before" data (CH,CL) is given to the MC
      // or written in the DB. No way to "extract" the tolerance from DB yet.

      if(_card->useAcceptParamFlag()==0){

	//float distp2 = (int)(2*_geom->cellH()*config()->ST()/_geom->cellPitch());   SV fix 17/III/07
      	float distp2 = 2*_geom->cellH()*config()->ST()/_geom->cellPitch();
      	float K0 = config()->ST();

      	// position of BTI 1 and of current one
      	DTBtiId _id1 = DTBtiId(sid,supl,1);
      	GlobalPoint gp1 = _geom->CMSPosition(_id1); 
      	GlobalPoint gp = CMSPosition();
      	if(config()->debug()>3){
        	cout << "Position: R=" << gp.perp() << "cm, Phi=" << gp.phi()*180/3.14159;
        	cout << " deg, Z=" << gp.z() << " cm" << endl;
      	}
	// new geometry: modified wrt old due to specularity of theta SLs --> fixed 6/9/06 
      	float theta;
      	if(gp1.z() < 0.0) 
      		theta = atan( -(gp.z())/gp.perp() );				
      	else 
        	theta = atan( (gp.z())/gp.perp() );

	// set BTI acceptance window : fixed wrt ORCA on 6/9/06  
      	float fktmin = tan(theta)*distp2+K0 ;
	int ktmin = static_cast<int>(fktmin)-config()->KAccTheta();
      	float fktmax = tan(theta)*distp2+K0+1;
	int ktmax = static_cast<int>(fktmax)+config()->KAccTheta();
      	if(ktmin>_MinKAcc)
		_MinKAcc=ktmin;
      	if(ktmax<_MaxKAcc)
		_MaxKAcc=ktmax;
      }
      // 091105 SV acceptance is taken simply from CH, CL parameters
      else {
	_MinKAcc = config()->CL();
	_MaxKAcc = config()->CH();
      }	

      // debugging
      if(config()->debug()>2){
        cout << "CMS position:" << CMSPosition() << endl;
        cout << "K acceptance (theta view):" << _MinKAcc << "," << _MaxKAcc  << endl;
      }// end debugging
	 
    }//end theta acceptance computation 

  //}// end if trigSetupGeom=0


  //SV flag for initialization....
  init_done = 0; 

}

//--------------
// Destructor --
//--------------
DTBtiChip::~DTBtiChip(){
  clear();
}

void 
DTBtiChip::add_digi(int cell, const DTDigi* digi) {

  if(_id.bti()<1 || _id.bti() >_geom->nCell(superlayer()))return;
  if(cell<1 || cell>9){
    cout << "DTBtiChip::add_digi : wrong cell number: " << cell;
    cout << ". Digi not added!" << endl;
    return;
  }

  int DEAD = config()->DEADpar();
  float stepTimeTdc = DTBtiHit::_stepTimeTdc;


  if( int(digi->countsTDC()/stepTimeTdc) - _busyStart_clock[cell-1] > DEAD ){
    _busyStart_clock[cell-1] = int(digi->countsTDC()/stepTimeTdc);
    _digis[cell-1].push_back(digi);

    // debugging
    if(config()->debug()>1){
    cout << "DTBtiChip::add_digi: DTBtiChip # " <<_id.bti() <<
    " cell " << cell << " --> drift time (tdc units)= " << digi->countsTDC()<< endl;
    digi->print();
    }
  }
  else {
  // debugging
  if(config()->debug()>1)
    cout << "DTBtiChip::add_digi: DTBtiChip # " <<_id.bti() <<
     " cell " << cell << " in dead time -> digi not added! " << endl;
  }

}


void 
DTBtiChip::add_digi_clock(int cell, int digi) {

  if(cell<1 || cell>9){
    cout << "DTBtiChip::add_digi_clock : wrong cell number: " << cell;
    cout << ". Digi not added!" << endl;
    return;
  }

  int DEAD = config()->DEADpar();

  if( digi - _busyStart_clock[cell-1] > DEAD ){
    _busyStart_clock[cell-1] = digi;
    _digis_clock[cell-1].push_back(digi);
    // debugging
    if(config()->debug()>1)
      cout << "DTBtiChip::add_digi_clock: DTBtiChip # " <<number() <<
      " cell " << cell << " --> clock time = " << digi << endl;
  }
  else{
  // debugging
  if(config()->debug()>1)
    cout << "DTBtiChip::add_digi_clock: DTBtiChip # " << number() <<
     " cell " << cell << " in dead time -> digi not added! " << endl;
  }
}


int
DTBtiChip::nCellHit() const {
  int n=0;
  int i=0;
  for(auto const& d: _digis) {
    if( !d.empty() ) n++;
  }
  if(config()->debug()>2) {
    cout << n << " cells with hits found:" << endl;
  }
  if(config()->debug()>2) {
    for(i=0;i<9;i++) {
      for(auto const&d: _digis[i]) {
        cout << "DTBtiChip # " << 
        _id.bti() << 
        " cell " << i+1;
        cout << " --> drift time (tdc units): " << d->countsTDC() << endl;
        d->print();
      }
    }
  }
  return n;
/*
 //SV 2/IV/03 counting hits from _hits
  int n=0;
  int i=0;
  for(i=0;i<9;i++) {
    if( _hits[i].size() >0 ) n++;
  }
  if(config()->debug()>2) {
    cout << n << " cells with hits found:" << endl;
  }
  if(config()->debug()>2) {
    for(i=0;i<9;i++) {
      vector<const DTBtiHit*>::const_iterator p;
      for(p=_hits[i].begin();p<_hits[i].end();p++) {
        cout << "DTBtiChip # " << 
        number() << 
        " cell " << i+1;
        if((*p)->curTime()!=4000)
          cout << " --> drift time: " << (*p)->curTime() << endl;
        else
          cout << " --> clock time: " << (*p)->clockTime() << endl;
      }
    }
  }
  return n;
*/
}

void 
DTBtiChip::addTrig(int step, std::unique_ptr<DTBtiTrig> btitrig) { 
  if(step>=DTConfig::NSTEPF&&step<=DTConfig::NSTEPL){
    if(config()->debug()>3) 
      cout << "DTBtiChip: adding trigger..." <<endl;
    _trigs[step-DTConfig::NSTEPF].emplace_back(std::move(btitrig));
  } else {
    if(config()->debug()>3){    
      cout << "DTBtiChip::addTrig: step " << step ;
      cout << " outside range. Trigger not added" << endl;
    }
  } 
}

int
DTBtiChip::nTrig(int step) const {
  if(step<DTConfig::NSTEPF||step>DTConfig::NSTEPL){
    cout << "DTBtiChip::nTrig: step out of range: " << step ;
    cout << " 0 returned" << endl;
    return 0;
  }
  return _trigs[step-DTConfig::NSTEPF].size(); 
}

vector<std::unique_ptr<DTBtiTrig>> const&
DTBtiChip::trigList(int step) const {
  if(step<DTConfig::NSTEPF||step>DTConfig::NSTEPL){
    cout << "DTBtiChip::trigList: step out of range: " << step ;
    cout << " empty pointer returned" << endl;
    //return 0;
  } 
  return _trigs[step-DTConfig::NSTEPF]; 
}

DTBtiTrig const*
DTBtiChip::trigger(int step, unsigned n) const {
  if(step<DTConfig::NSTEPF||step>DTConfig::NSTEPL){
    cout << "DTBtiChip::trigger: step out of range: " << step ;
    cout << " empty pointer returned" << endl;
    return nullptr;
  } 
  if(n<1 || n>_trigs[step-DTConfig::NSTEPF].size()) {
    cout << "DTBtiChip::trigger: requested trigger does not exist: " << n;
    cout << " empty pointer returned!" << endl;
    return nullptr;
  }
  auto p = _trigs[step-DTConfig::NSTEPF].begin();
  return (*(p+n-1)).get();
}

DTBtiTrigData
DTBtiChip::triggerData(int step, unsigned n) const {
  if(step<DTConfig::NSTEPF||step>DTConfig::NSTEPL){
    cout << "DTBtiChip::triggerData: step out of range: " << step ;
    cout << " dummy trigger returned" << endl;
    return DTBtiTrigData();
  } 
  if(n<1 || n>_trigs[step-DTConfig::NSTEPF].size()) {
    cout << "DTBtiChip::triggerData: requested trig. doesn't exist: " << n;
    cout << " dummy trigger returned!" << endl;
    return DTBtiTrigData();
  }
  auto p = _trigs[step-DTConfig::NSTEPF].begin();
  return (*(p+n-1))->data();
}

void
DTBtiChip::eraseTrigger(int step, unsigned n) {
  if(step<DTConfig::NSTEPF||step>DTConfig::NSTEPL){
    cout << "DTBtiChip::eraseTrigger: step out of range: " << step ;
    cout << " trigger not deleted!" << endl;
  } 
  if(n<1 || n>_trigs[step-DTConfig::NSTEPF].size()) {
    cout << "DTBtiChip::trigger: requested trigger does not exist: " << n;
    cout << " trigger not deleted!" << endl;
  }
  auto p = _trigs[step-DTConfig::NSTEPF].begin()+n-1;
  _trigs[step-DTConfig::NSTEPF].erase(p);
}

void
DTBtiChip::clear() {

  if(config()->debug()>3)
    cout << "DTBtiChip::clear()" << endl;
  
  for(auto& d: _digis) {
    d.clear();
  }
  for(auto& c: _digis_clock) {
    c.clear();
  }
  for(auto& h: _hits) {
    h.clear();
  }

  for(auto& t: _trigs) {
    t.clear();
  }
}

void
DTBtiChip::init() {

  if(config()->debug()>3)
    cout << "DTBtiChip::init() -> initializing bti chip" << endl;

  _curStep=0;
  for(int i=0;i<25;i++) {
    _sums[i] = 1000;
    _difs[i] = 1000;
  } 

  for(int cell=0;cell<9;cell++) {
    int WEN = config()->WENflag(cell+1);
    if( WEN==1 ){
      _thisStepUsedHit[cell]=nullptr;
      vector<const DTDigi*>::const_iterator p;
      for(p=_digis[cell].begin();p<_digis[cell].end();p++){
        DTBtiHit* hit = new DTBtiHit(*p,config());
        //int clockTime = (int)(fabs(((*p)->time()+config()->SetupTime())/12.5));
        //DTBtiHit* hit = new DTBtiHit(clockTime,config());
        _hits[cell].push_back(hit);
      }

      //debugging
      if(config()->debug()>2){
        vector<DTBtiHit*>::const_iterator p1;
        for(p1=_hits[cell].begin();p1<_hits[cell].end();p1++){
        	cout << " Filling hit in cell " << cell+1;
                if((*p1)->curTime()!=4000) 
                  cout << " raw time in trigger: " << (*p1)->curTime() << endl;
                cout << " time (clock units): " << (*p1)->clockTime() << endl; 
        }
      }
      // end debugging
    }
  }
}


void
DTBtiChip::init_clock() {

  if(config()->debug()>3)
    cout << "DTBtiChip::init_clock() -> initializing bti chip" << endl;

  init_done = 1;
  _curStep=0;

  for(int i=0;i<25;i++) {
    _sums[i] = 1000;
    _difs[i] = 1000;
  } 

  for(int cell=0;cell<9;cell++) {
    int WEN = config()->WENflag(cell+1);
    if( WEN==1 ){
      _thisStepUsedHit[cell]=nullptr;
    for(unsigned int i=0; i<_digis_clock[cell].size(); i++){
      const int clockTime = (_digis_clock[cell])[i];
      DTBtiHit* hit = new DTBtiHit(clockTime,config());
      _hits[cell].push_back(hit);
    }
	
    //debugging
    if(config()->debug()>2){
      vector<DTBtiHit*>::const_iterator p1;
      for(p1=_hits[cell].begin();p1<_hits[cell].end();p1++){
      	cout << " Filling hit in cell " << cell+1;
        if((*p1)->curTime()!=4000) 
          cout << " time: " << (*p1)->curTime() << endl;
        else
          cout << " time (clock units): " << (*p1)->clockTime() << endl; 
        }
      }
      // end debugging
    }
  }
}


void 
DTBtiChip::run() {

  // Debugging...
  if(config()->debug()>2){
    cout << "DTBtiChip::run: Processing BTI " << _id.bti() << endl;
    cout << " in SL " << _id.superlayer() << endl;
  }
  // End debugging

  if(_id.bti()<1 || _id.bti() >_geom->nCell(superlayer())) {
    if(config()->debug()>1)
      cout << "DTBtiChip::run : wrong BTI number: " << _id.bti() << endl;
    return;
  }

  // run algorithm
  if(!init_done)
    init();
  if( nCellHit()<3 ) return;   // check that at least 3 cell have hits

  for(int ints=0; ints<2*DTConfig::NSTEPL; ints++) { // 80 MHz 
    tick(); // Do a 12.5 ns step

    // In electronics equations are computed every 12.5 ns
    // but since triggers are searched for only every 25 ns, skip
    // also equation's computing at odd values of internal step
    if((currentIntStep()/2)*2!=currentIntStep())continue;
    //if((currentIntStep()/2)*2==currentIntStep())continue; 


    if(config()->debug()>2){
      cout << "DTBtiChip::run : internal step " << currentIntStep();
      cout << " number of JTRIG hits is " << _nStepUsedHits << endl;
    }
    if(currentStep()>=DTConfig::NSTEPF && _nStepUsedHits>2) { 
      // at least 3 good hits in this step -> run algorithm
      computeSums();
      computeEqs();
      findTrig();
    }
  }
  if( config()->LTS()>0 ) doLTS(); // low trigger suppression
}

void
DTBtiChip::tick() {
  //
  // fills the DTBtiChip registers ( _thisStepUsedHit[cell] )
  // for a given clock (Syncronizer and Shaper functionalities)
  //

  _curStep++; // increase internal step (12.5 ns --> 80 MHz)

  // debugging
  if(config()->debug()>2){
    cout << "DTBtiChip::tick: internal step is now " << currentIntStep()<< endl; 
  }
  // end debugging

  // Loop on cells
  _nStepUsedHits=0;
  for(int cell=0;cell<9;cell++) {

    // decrease drift time by 12.5 ns for each hit
    for(auto& h: _hits[cell]) {
      h->stepDownTime();
    }

    // loop on hits
    _thisStepUsedHit[cell]=nullptr;
    for(auto& h: _hits[cell]) {
      if       ( h->isDrifting() ) { // hit is drifting
        break;                          //   --> don't consider others
      } else if( h->isInsideReg() ) {  // hit is already in registers
	_thisStepUsedHit[cell]=h;
	_nStepUsedHits++;
	// debugging
	if(config()->debug()>2){
          if(h->curTime() != 4000)
            cout << "DTBtiChip::tick: hit in register: time=" << h->curTime();
          else
            cout << "DTBtiChip::tick: hit in register! " << endl;
          cout <<                           " jtrig=" << h->jtrig() << endl;

	}
	// end debugging
        break;                          //   --> don't consider other triggers
      }
      // hit is not drifting and not in registers: it is gone out of register, but
      // jtrig value is still=ST ; save in array and consider next one if exists
    } // end loop on cell hits

    // debugging...
    if(config()->debug()>2){
      if(_thisStepUsedHit[cell]!=nullptr){
	cout << "int. step=" << currentIntStep() << " cell=" << cell+1;
	cout << " jtrig=" << _thisStepUsedHit[cell]->jtrig();
        if( _thisStepUsedHit[cell]->curTime() != 4000 )  
	  cout << " (time=" << _thisStepUsedHit[cell]->curTime() << ")" << endl;
        else 
          cout << endl;
      }
    } 
    // end debugging

  } // end loop on cells

}

void
DTBtiChip::doLTS() {
 
  if(config()->debug()>2)
    cout<<"Do LTS"<<endl;
  int lts = config()->LTS();
  int nbxlts = config()->SET();

  // Do LTS only on the requested SL
  //if (superlayer()==2 && lts==1) return;
  //if (superlayer()!=2 && lts==2) return;
  //new DTConfig: do LTS only is LTS!=0  --> somewhat redundant !
  if (lts==0) return;

  // loop on steps
  for(int is=DTConfig::NSTEPF; is<=DTConfig::NSTEPL; is++) {
    if(nTrig(is)>0) { // non empty step
      if( trigger(is,1)->code()==8 ) { // HTRIG at this step
        // do LTS on nbxLTS[superlayer] following steps
        for(int js=is+1;(js<=is+nbxlts&&js<=DTConfig::NSTEPL);js++){
          if(nTrig(js)>0) { // non empty step
            DTBtiTrig const* tr = trigger(js,1);
            if( tr->code()<8 && (lts==1 || lts==3)) {
              if(config()->debug()>3)
                cout<<"LTS: erasing trigger!"<<endl; 
              eraseTrigger(js,1); // delete trigger
            }
          }
        }
        // do LTS on previous step
        if(is>DTConfig::NSTEPF && nTrig(is-1)>0) { // non empty step
          DTBtiTrig const* tr = trigger(is-1,1);
          if( tr->code()<8 && (lts==2 || lts==3) ) {
            if(config()->debug()>3)
                cout<<"LTS: erasing trigger!"<<endl;                                
            eraseTrigger(is-1,1); // delete trigger
          }
        }

      }
    }
  }
}

int
DTBtiChip::store(const int eq, const int code, const int K, const int X, 
                     float KeqAB, float KeqBC, float KeqCD, 
                     float KeqAC, float KeqBD, float KeqAD) {

  // remove negative position triggers
  if(X<0)return 0;

  
  // accept in range triggers (acceptances defined in constructor)
  if(K>=_MinKAcc && K<=_MaxKAcc) 
  {
    int trig_step = currentStep();

/*
    //SV test 27/I/2003 1-clock delay for critical patterns in default ACx configuration 
    int AC1 = config()->AccPattAC1(); //default 0
    int AC2 = config()->AccPattAC2(); //default 3
    int ACH = config()->AccPattACH(); //default 1
    int ACL = config()->AccPattACL(); //default 2     

    if(AC1==0 && AC2==3 && ACH==1 && ACL==2){
      if(eq==1 || eq==4 || eq==7 || eq==8 || eq==9 || eq==12 || eq==15
	|| eq==19 || eq==22 || eq==23 || eq==24 || eq==25 || eq==27 )
	   trig_step = currentStep()+1;
    }
*/     
    //store strobe: SV no strobe defined for this setup SV 15/I/2007
    int strobe=-1;

    // create a new trigger
    float Keq[6] = {KeqAB,KeqBC,KeqCD,KeqAC,KeqBD,KeqAD};
    //DTBtiTrig* trg = new DTBtiTrig(this,code,K,X,currentStep(),eq);
    auto trg = std::make_unique<DTBtiTrig>(this,code,K,X,trig_step,eq,strobe,Keq);

    // store also the digis
    for(auto& h: _thisStepUsedHit) {
      if(h) {
        const DTDigi* digi = h->hitDigi();
        if(digi)
          trg->addDigi(digi);
      }
    }

    // Debugging...
    if(config()->debug()>1)
      trg->print();
    // end debugging
    
    //addTrig(currentStep(),trg);
    addTrig(trig_step,std::move(trg));

    return 1;
  }
  else{
    // remove out of range triggers (acceptances defined in constructor)
    if(config()->debug()>2){
      cout << "DTBtiChip::store, REJECTED TRIGGER at step "<< currentStep();
      cout << " allowed K range in theta view is: [";
      cout << _MinKAcc << ","<< _MaxKAcc << "]";
      cout << "K value is " << K << endl; 
    }
    return 0;
  }//end else
}//end store


void
DTBtiChip::setSnap(){

 //set the internally calculated drift velocity parameters
  ST43 = config()->ST43();
  RE43 = config()->RE43();
  ST23 = int(double(ST43)/2.);
  RE23 = (RE43==1) ? 2 : int(double(RE43)/2.);


  ST =  int(  double(ST43) * 3./4. + double(RE43) * 1./4.     );
  ST2 = int( (double(ST43) * 3./4. + double(RE43) * 1./4.)*2. );
  ST3 = int( (double(ST43) * 3./4. + double(RE43) * 1./4.)*3. );
  ST4 = int( (double(ST43) * 3./4. + double(RE43) * 1./4.)*4. );
  ST5 = int( (double(ST43) * 3./4. + double(RE43) * 1./4.)*5. );
  ST7 = int( (double(ST43) * 3./4. + double(RE43) * 1./4.)*7. );

  if(config()->debug()>3){
    cout << "Snap register dump: " << endl;
    cout << "ST43 = " << ST43 << endl;
    cout << "RE43 = " << RE43 << endl;
    cout << "ST23 = " << ST23 << endl;
    cout << "RE23 = " << RE23 << endl;
    cout << "ST = " << ST << endl;
    cout << "ST2 = " << ST2 << endl;
    cout << "ST3 = " << ST3 << endl;
    cout << "ST4 = " << ST4 << endl;
    cout << "ST5 = " << ST5 << endl;
    cout << "ST7 = " << ST7 << endl;
  }
}

