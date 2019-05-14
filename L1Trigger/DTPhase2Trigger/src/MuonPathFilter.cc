#include "L1Trigger/DTPhase2Trigger/interface/MuonPathFilter.h"

using namespace edm;
using namespace std;



// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathFilter::MuonPathFilter(const ParameterSet& pset) {
  // Obtention of parameters
  debug         = pset.getUntrackedParameter<Bool_t>("debug");
  filter_cousins = pset.getUntrackedParameter<bool>("filter_cousins");
  tanPhiTh      = pset.getUntrackedParameter<double>("tanPhiTh");
  if (debug) cout <<"MuonPathFilter: constructor" << endl;
}


MuonPathFilter::~MuonPathFilter() {
  if (debug) cout <<"MuonPathFilter: destructor" << endl;
}



// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathFilter::initialise(const edm::EventSetup& iEventSetup) {
  if(debug) cout << "MuonPathFilter::initialiase" << endl;
}

int MuonPathFilter::areCousins(metaPrimitive primera, metaPrimitive segunda) {
  if(primera.rawId!=segunda.rawId) return 0;
  if(primera.wi1==segunda.wi1 and primera.tdc1==segunda.tdc1 and primera.wi1!=-1 and primera.tdc1!=-1) return 1;
  if(primera.wi2==segunda.wi2 and primera.tdc2==segunda.tdc2 and primera.wi2!=-1 and primera.tdc2!=-1) return 2;
  if(primera.wi3==segunda.wi3 and primera.tdc3==segunda.tdc3 and primera.wi3!=-1 and primera.tdc3!=-1) return 3;
  if(primera.wi4==segunda.wi4 and primera.tdc4==segunda.tdc4 and primera.wi4!=-1 and primera.tdc4!=-1) return 4;
  return 0;
}
int MuonPathFilter::rango(metaPrimitive primera) {
  int rango=0;
  if(primera.wi1!=-1)rango++;
  if(primera.wi2!=-1)rango++;
  if(primera.wi3!=-1)rango++;
  if(primera.wi4!=-1)rango++;
  return rango;
}

void MuonPathFilter::run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, 
			 std::vector<metaPrimitive> &inMPaths, 
			 std::vector<metaPrimitive> &outMPaths) 
{
  
  if (debug) cout <<"MuonPathFilter: run" << endl;  
  
  if(filter_cousins) 
    filterCousins(inMPaths,outMPaths); 
  else 
    filterTanPhi(inMPaths,outMPaths);
  
  if (debug) cout <<"MuonPathFilter: done" << endl;
}

void MuonPathFilter::filterCousins(std::vector<metaPrimitive> &inMPaths, 
				  std::vector<metaPrimitive> &outMPaths) 
{
  if(debug) std::cout<<"filtering: starting cousins filtering"<<std::endl;    
  
  int primo_index=0;
  bool oneof4=false;
  
  if(inMPaths.size()==1){
    if(debug){
      std::cout<<"filtering:";
      printmP(inMPaths[0]);
      std::cout<<" \t is:"<<0<<" "<<primo_index<<" "<<" "<<oneof4<<std::endl;
    }
    if(fabs(inMPaths[0].tanPhi)<tanPhiTh){
      outMPaths.push_back(inMPaths[0]);
      if(debug)std::cout<<"filtering: kept1 i="<<0<<std::endl;
    }
  }
  else {
    for(int i=1; i<int(inMPaths.size()); i++){ 
      if(fabs(inMPaths[i].tanPhi)>tanPhiTh) continue;
      if(rango(inMPaths[i])==4)oneof4=true;
      if(debug){
	std::cout<<"filtering:";
	printmP(inMPaths[i]);
	std::cout<<" \t is:"<<i<<" "<<primo_index<<" "<<" "<<oneof4<<std::endl;
      }
      if(areCousins(inMPaths[i],inMPaths[i-1])!=0  and areCousins(inMPaths[i],inMPaths[i-primo_index-1])!=0){
	primo_index++;
      }else{
	if(primo_index==0){
	  outMPaths.push_back(inMPaths[i]);
	  if(debug)std::cout<<"filtering: kept2 i="<<i<<std::endl;
	}else{
	  if(oneof4){
	    double minchi2=99999;
	    int selected_i=0;
	    for(int j=i-1;j>=i-primo_index-1;j--){
	      if(rango(inMPaths[j])!=4) continue;
	      if(minchi2>inMPaths[j].chi2){
		minchi2=inMPaths[j].chi2;
		selected_i=j;
	      }
	    }
	    outMPaths.push_back(inMPaths[selected_i]);
	    if(debug)std::cout<<"filtering: kept4 i="<<selected_i<<std::endl;
	  }else{
	    for(int j=i-1;j>=i-primo_index-1;j--){
	      outMPaths.push_back(inMPaths[j]);
	      if(debug)std::cout<<"filtering: kept3 i="<<j<<std::endl;
	    }
	  }
	}
	primo_index=0;
	oneof4=false;
      }
    }
  }
}
void MuonPathFilter::filterTanPhi(std::vector<metaPrimitive> &inMPaths, 
				  std::vector<metaPrimitive> &outMPaths) 
{
  for (size_t i=0; i<inMPaths.size(); i++){ 
    if(fabs(inMPaths[i].tanPhi)>tanPhiTh) continue;
    outMPaths.push_back(inMPaths[i]); 
  }
}

void MuonPathFilter::finish() {
  if (debug) cout <<"MuonPathFilter: finish" << endl;
};


void MuonPathFilter::printmP(metaPrimitive mP){
    DTSuperLayerId slId(mP.rawId);
    std::cout<<slId<<"\t"
	     <<" "<<setw(2)<<left<<mP.wi1
	     <<" "<<setw(2)<<left<<mP.wi2
	     <<" "<<setw(2)<<left<<mP.wi3
	     <<" "<<setw(2)<<left<<mP.wi4
	     <<" "<<setw(5)<<left<<mP.tdc1
	     <<" "<<setw(5)<<left<<mP.tdc2
	     <<" "<<setw(5)<<left<<mP.tdc3
	     <<" "<<setw(5)<<left<<mP.tdc4
	     <<" "<<setw(10)<<right<<mP.x
	     <<" "<<setw(9)<<left<<mP.tanPhi
	     <<" "<<setw(5)<<left<<mP.t0
	     <<" "<<setw(13)<<left<<mP.chi2
	     <<" r:"<<rango(mP);
}

