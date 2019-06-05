#include "L1Trigger/DTPhase2Trigger/interface/MPQualityEnhancerFilter.h"

using namespace edm;
using namespace std;



// ============================================================================
// Constructors and destructor
// ============================================================================
MPQualityEnhancerFilter::MPQualityEnhancerFilter(const ParameterSet& pset) :
  MPFilter(pset)
{
  // Obtention of parameters
  debug         = pset.getUntrackedParameter<Bool_t>("debug");
  filter_cousins = pset.getUntrackedParameter<bool>("filter_cousins");
  if (debug) cout <<"MPQualityEnhancerFilter: constructor" << endl;
}


MPQualityEnhancerFilter::~MPQualityEnhancerFilter() {
  if (debug) cout <<"MPQualityEnhancerFilter: destructor" << endl;
}



// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MPQualityEnhancerFilter::initialise(const edm::EventSetup& iEventSetup) {
  if(debug) cout << "MPQualityEnhancerFilter::initialiase" << endl;
}

void MPQualityEnhancerFilter::run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, 
			 std::vector<metaPrimitive> &inMPaths, 
			 std::vector<metaPrimitive> &outMPaths)
{
  
  if (debug) cout <<"MPQualityEnhancerFilter: run" << endl;  
  
  std::vector<metaPrimitive> buff; 
  
  filterCousins(inMPaths,buff); 
  if (debug){ cout <<"Ended Cousins Filter. The final primitives before UniqueFilter are: " << endl;
    for (unsigned int i=0; i<buff.size(); i++){
      printmP(buff[i]);cout<<endl;
    }
      cout << "Total Primitives = " << buff.size()<< endl;  
  } 
  filterUnique(buff,outMPaths);

  if (debug){ cout <<"Ended Unique Filter. The final primitives are: " << endl;
    for (unsigned int i=0; i<outMPaths.size(); i++){
      printmP(outMPaths[i]);cout<<endl; 
    }
      cout << "Total Primitives = " << outMPaths.size()<< endl;  
  } 

  buff.clear();
  buff.erase(buff.begin(),buff.end());
  
  if (debug) cout <<"MPQualityEnhancerFilter: done" << endl;
}

void MPQualityEnhancerFilter::finish() {
  if (debug) cout <<"MPQualityEnhancerFilter: finish" << endl;
};

///////////////////////////
///  OTHER METHODS
/*int MPQualityEnhancerFilter::areCousins(metaPrimitive mp, metaPrimitive second_mp) {
    if(mp.rawId!=second_mp.rawId) return 0;
    if(mp.wi1==second_mp.wi1 and mp.tdc1==second_mp.tdc1 and mp.wi1!=-1 and mp.tdc1!=-1) return 1;
    if(mp.wi2==second_mp.wi2 and mp.tdc2==second_mp.tdc2 and mp.wi2!=-1 and mp.tdc2!=-1) return 2;
    if(mp.wi3==second_mp.wi3 and mp.tdc3==second_mp.tdc3 and mp.wi3!=-1 and mp.tdc3!=-1) return 3;
    if(mp.wi4==second_mp.wi4 and mp.tdc4==second_mp.tdc4 and mp.wi4!=-1 and mp.tdc4!=-1) return 4;
    return 0;
}*/
int MPQualityEnhancerFilter::areCousins(metaPrimitive mp, metaPrimitive second_mp) {
    if(mp.rawId!=second_mp.rawId) return 0;
    if(mp.wi1==second_mp.wi1 and mp.wi1!=-1 and mp.tdc1!=-1) return 1;
    if(mp.wi2==second_mp.wi2 and mp.wi2!=-1 and mp.tdc2!=-1) return 2;
    if(mp.wi3==second_mp.wi3 and mp.wi3!=-1 and mp.tdc3!=-1) return 3;
    if(mp.wi4==second_mp.wi4 and mp.wi4!=-1 and mp.tdc4!=-1) return 4;
    return 0;
}


int MPQualityEnhancerFilter::rango(metaPrimitive mp){
    if(mp.quality==1 or mp.quality==2) return 3;
    if(mp.quality==3 or mp.quality==4) return 4;
    return 0;
}

/*
void MPQualityEnhancerFilter::filterCousins(std::vector<metaPrimitive> &inMPaths, 
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
	outMPaths.push_back(inMPaths[0]);
	if(debug)std::cout<<"filtering: kept0 i="<<0<<std::endl;
    }
    else {
	for(int i=1; i<=int(inMPaths.size()); i++){ 
	//for(int i=1; i<=int(inMPaths.size()); i++){ 
	    if(rango(inMPaths[i])==4)oneof4=true;
	    if(debug){
		std::cout<<"filtering:";
		if(i!=(int) inMPaths.size()){
	          printmP(inMPaths[i]);
		  std::cout<<" \t is:"<<i<<" "<<primo_index<<" "<<" "<<oneof4<<std::endl;
		}
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
			for(int j=i-1;j>=i-primo_index-1;j--){ //GUARDA EL PRIMERO COMO KEPT2 y KEPT3
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
*/

void MPQualityEnhancerFilter::filterCousins(std::vector<metaPrimitive> &inMPaths, 
				   std::vector<metaPrimitive> &outMPaths) 
{
    if(debug) std::cout<<"filtering: starting cousins filtering"<<std::endl;    
  
    int primo_index=0;
    bool oneof4=false;
    int bestI = -1; 
    double bestChi2 = 9999; 
    if(inMPaths.size()==1){
	if(debug){
	    std::cout<<"filtering:";
	    printmP(inMPaths[0]);
	    std::cout<<" \t is:"<<0<<" "<<primo_index<<" "<<" "<<oneof4<<std::endl;
	}
	outMPaths.push_back(inMPaths[0]);
	if(debug)std::cout<<"filtering: kept0 i="<<0<<std::endl;
    }
    else {
	for(int i=0; i<int(inMPaths.size()); i++){ 
	    if(debug){
	        std::cout<<"filtering:";
	        printmP(inMPaths[i]);
	        std::cout<<" \t is:"<<i<<" "<<primo_index<<" "<<" "<<oneof4<<std::endl;
	    }
	    if(rango(inMPaths[i])==4){
		oneof4=true;
		if (bestChi2 > inMPaths[i].chi2){
		    bestChi2 = inMPaths[i].chi2; 
		    bestI = i;
		}
	    }
	    if (areCousins(inMPaths[i],inMPaths[i+1])!=0) {
		primo_index++;
	    } else {
		if (oneof4) {
		   outMPaths.push_back(inMPaths[bestI]);
		   if(debug)std::cout<<"filtering: kept4 i="<<bestI<<std::endl;
		   bestI = -1; bestChi2 = 9999; oneof4=false; 
		} else {
		   for (int j = i-primo_index; j<=i; j++){
		       outMPaths.push_back(inMPaths[j]);
		       if(debug)std::cout<<"filtering: kept3 i="<<j<<std::endl;
		   }
		}
		primo_index = 0; 
	    }
	}
    }


} //End filterCousins



void MPQualityEnhancerFilter::filterUnique(std::vector<metaPrimitive> &inMPaths,
				  std::vector<metaPrimitive> &outMPaths)
{
    double xTh = 0.001;
    double tPhiTh = 0.001; 
    double t0Th = 0.001;
    for (size_t i=0; i<inMPaths.size();i++){
	bool visto = false; 
	for (size_t j=i+1; j<inMPaths.size();j++){
	    if ((fabs(inMPaths[i].x-inMPaths[j].x)<=xTh)&&(fabs(inMPaths[i].tanPhi-inMPaths[j].tanPhi)<=tPhiTh)&&(fabs(inMPaths[i].t0-inMPaths[j].t0)<=t0Th)) visto = true; 
	}
	if (!visto) outMPaths.push_back(inMPaths[i]);
    }
}


void MPQualityEnhancerFilter::printmP(metaPrimitive mP){
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

