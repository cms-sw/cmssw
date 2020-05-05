#include "L1Trigger/DTTriggerPhase2/interface/MPQualityEnhancerFilter.h"

using namespace edm;
using namespace std;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPQualityEnhancerFilter::MPQualityEnhancerFilter(const ParameterSet &pset) : MPFilter(pset) {
  // Obtention of parameters
  debug = pset.getUntrackedParameter<Bool_t>("debug");
  filter_cousins = pset.getUntrackedParameter<bool>("filter_cousins");
  if (debug)
    cout << "MPQualityEnhancerFilter: constructor" << endl;
}

MPQualityEnhancerFilter::~MPQualityEnhancerFilter() {
  if (debug)
    cout << "MPQualityEnhancerFilter: destructor" << endl;
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MPQualityEnhancerFilter::initialise(const edm::EventSetup &iEventSetup) {
  if (debug)
    cout << "MPQualityEnhancerFilter::initialiase" << endl;
}

void MPQualityEnhancerFilter::run(edm::Event &iEvent,
                                  const edm::EventSetup &iEventSetup,
                                  std::vector<metaPrimitive> &inMPaths,
                                  std::vector<metaPrimitive> &outMPaths) {
  if (debug)
    cout << "MPQualityEnhancerFilter: run" << endl;

  std::vector<metaPrimitive> buff;
  std::vector<metaPrimitive> buff2;

  filterCousins(inMPaths, buff);
  if (debug) {
    cout << "Ended Cousins Filter. The final primitives before Refiltering are: " << endl;
    //if (debug){ cout <<"Ended Cousins Filter. The final primitives before UniqueFilter are: " << endl;
    for (unsigned int i = 0; i < buff.size(); i++) {
      printmP(buff[i]);
      cout << endl;
    }
    cout << "Total Primitives = " << buff.size() << endl;
  }
  refilteringCousins(buff, buff2);
  if (debug) {
    cout << "Ended Cousins Refilter. The final primitives before UniqueFilter are: " << endl;
    for (unsigned int i = 0; i < buff2.size(); i++) {
      printmP(buff2[i]);
      cout << endl;
    }
    cout << "Total Primitives = " << buff2.size() << endl;
  }
  filterUnique(buff2, outMPaths);
  // filterUnique(buff,outMPaths);

  if (debug) {
    cout << "Ended Unique Filter. The final primitives are: " << endl;
    for (unsigned int i = 0; i < outMPaths.size(); i++) {
      printmP(outMPaths[i]);
      cout << endl;
    }
    cout << "Total Primitives = " << outMPaths.size() << endl;
  }

  buff.clear();
  buff.erase(buff.begin(), buff.end());
  buff2.clear();
  buff2.erase(buff2.begin(), buff2.end());

  if (debug)
    cout << "MPQualityEnhancerFilter: done" << endl;
}

void MPQualityEnhancerFilter::finish() {
  if (debug)
    cout << "MPQualityEnhancerFilter: finish" << endl;
};

///////////////////////////
///  OTHER METHODS
/*int MPQualityEnhancerFilter::areCousins(metaPrimitive mp, metaPrimitive second_mp) {
  if (mp.rawId != second_mp.rawId)
    return 0;
  if (mp.wi1 == second_mp.wi1 and mp.tdc1 == second_mp.tdc1 and mp.wi1 != -1 and mp.tdc1 != -1)
    return 1;
  if (mp.wi2 == second_mp.wi2 and mp.tdc2 == second_mp.tdc2 and mp.wi2 != -1 and mp.tdc2 != -1)
    return 2;
  if (mp.wi3 == second_mp.wi3 and mp.tdc3 == second_mp.tdc3 and mp.wi3 != -1 and mp.tdc3 != -1)
    return 3;
  if (mp.wi4 == second_mp.wi4 and mp.tdc4 == second_mp.tdc4 and mp.wi4 != -1 and mp.tdc4 != -1)
    return 4;
  return 0;
}*/
int MPQualityEnhancerFilter::areCousins(metaPrimitive mp, metaPrimitive second_mp) {
    if(mp.rawId!=second_mp.rawId) 
      return 0;
    if(mp.wi1==second_mp.wi1 and mp.wi1!=-1 and mp.tdc1!=-1) 
      return 1;
    if(mp.wi2==second_mp.wi2 and mp.wi2!=-1 and mp.tdc2!=-1) 
      return 2;
    if(mp.wi3==second_mp.wi3 and mp.wi3!=-1 and mp.tdc3!=-1)
      return 3;
    if(mp.wi4==second_mp.wi4 and mp.wi4!=-1 and mp.tdc4!=-1) 
      return 4;
    return 0;
}

int MPQualityEnhancerFilter::rango(metaPrimitive mp) {
  if (mp.quality == 1 or mp.quality == 2)
    return 3;
  if (mp.quality == 3 or mp.quality == 4)
    return 4;
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
                                            std::vector<metaPrimitive> &outMPaths) {
  if (debug)
    std::cout << "filtering: starting cousins filtering" << std::endl;

  int primo_index = 0;
  bool oneof4 = false;
  int bestI = -1;
  double bestChi2 = 9999;
  if (inMPaths.size() == 1) {
    if (debug) {
      std::cout << "filtering:";
      printmP(inMPaths[0]);
      std::cout << " \t is:" << 0 << " " << primo_index << " "
                << " " << oneof4 << std::endl;
    }
    outMPaths.push_back(inMPaths[0]);
    if (debug)
      std::cout << "filtering: kept0 i=" << 0 << std::endl;
  } else if (inMPaths.size() > 1) {
    for (int i = 0; i < int(inMPaths.size()); i++) {
      if (debug) {
        std::cout << "filtering:";
        printmP(inMPaths[i]);
        std::cout << " \t is:" << i << " " << primo_index << " "
                  << " " << oneof4 << std::endl;
      }
      if (areCousins(inMPaths[i], inMPaths[i - primo_index]) == 0) {
        if (oneof4) {
          outMPaths.push_back(inMPaths[bestI]);

          if (debug)
            std::cout << "filtering: kept4 i=" << bestI << std::endl;
          bestI = -1;
          bestChi2 = 9999;
          oneof4 = false;
        } else {
          for (int j = i - primo_index; j < i; j++) {
            outMPaths.push_back(inMPaths[j]);
            if (debug)
              std::cout << "filtering: kept3 i=" << j << std::endl;
          }
        }
        i--;
        primo_index = 0;
        continue;
      }
      if (rango(inMPaths[i]) == 4) {
        oneof4 = true;
        if (bestChi2 > inMPaths[i].chi2) {
          bestChi2 = inMPaths[i].chi2;
          bestI = i;
        }
      }
      if (areCousins(inMPaths[i], inMPaths[i + 1]) != 0) {
        primo_index++;
      } else if (areCousins(inMPaths[i], inMPaths[i + 1]) == 0) {
        if (oneof4) {
          outMPaths.push_back(inMPaths[bestI]);

          if (debug)
            std::cout << "filtering: kept4 i=" << bestI << std::endl;
          bestI = -1;
          bestChi2 = 9999;
          oneof4 = false;
        } else {
          for (int j = i - primo_index; j <= i; j++) {
            outMPaths.push_back(inMPaths[j]);
            if (debug)
              std::cout << "filtering: kept3 i=" << j << std::endl;
          }
        }
        primo_index = 0;
      }
    }
  }

}  //End filterCousins
/*
void MPQualityEnhancerFilter::refilteringCousins(std::vector<metaPrimitive> &inMPaths, 
				   std::vector<metaPrimitive> &outMPaths) 
{

    if(debug) std::cout<<"filtering: starting cousins refiltering"<<std::endl;    
    int bestI = -1; 
    double bestChi2 = 9999;
    bool oneOf4 = false; 
    bool enter = true; 
    if (inMPaths.size()>1){
      for (int i = 0; i<(int)inMPaths.size(); i++){
	if (debug) { cout << "filtering: starting with mp " << i << ": "; printmP(inMPaths[i]); cout << endl; }
	enter = true; 
	if (rango(inMPaths[i])==4) {
	  oneOf4 = true; 
          bestI = i;
	  bestChi2 = inMPaths[i].chi2; 
	}
        //for (int j = 0; j<(int)inMPaths.size(); j++){
        for (int j = i+1; j<(int)inMPaths.size(); j++){
          if (areCousins(inMPaths[i],inMPaths[j])==0){ //they arent cousins
	    if (debug) { cout << "filtering:          mp " << i << " is not cousin from mp " << j << ": "; printmP(inMPaths[j]); cout << endl; }
	    enter = false; // We dont want to save them two times
	    if (oneOf4 == false){
	      if (debug) cout << "kept3 mp" << i << endl; ;  
	      outMPaths.push_back(inMPaths[i]);
	    } else {
	      outMPaths.push_back(inMPaths[bestI]);
	      if (debug) cout << "kept4 mp" << bestI << endl;  
	      bestI = -1; 
	      bestChi2 = 9999;
	      oneOf4 = false; 
	      i = j -1; 
	    }
	    break;
	  } else { //they are cousins
	    if (rango(inMPaths[j])==4) {
	      if (oneOf4 == true) {
		if (bestChi2 > inMPaths[j].chi2) {
		  bestI = j; 
		  bestChi2 = inMPaths[j].chi2;
		}
	      } else { // if rango of j is 4 and this MP has no rango 4, I will not accept this rango 3 mp
		enter = false; break;
	      }
	    } // if range of j is not 4, I do not do anything until I get a rango 4 or a no-primo-mp
	  }
        }
        if (enter == true) outMPaths.push_back(inMPaths[i]);
      }
    } else if (inMPaths.size() == 1) {
      outMPaths.push_back(inMPaths[0]);
    }


} 
*/

void MPQualityEnhancerFilter::refilteringCousins(std::vector<metaPrimitive> &inMPaths,
                                                 std::vector<metaPrimitive> &outMPaths) {
  if (debug)
    std::cout << "filtering: starting cousins refiltering" << std::endl;
  int bestI = -1;
  double bestChi2 = 9999;
  bool oneOf4 = false;
  int back = 0;

  if (inMPaths.size() > 1) {
    for (unsigned int i = 0; i < inMPaths.size(); i++) {
      if (debug) {
        cout << "filtering: starting with mp " << i << ": ";
        printmP(inMPaths[i]);
        cout << endl;
      }
      if (rango(inMPaths[i]) == 4 && bestChi2 > inMPaths[i].chi2) {  // 4h prim with a smaller chi2
        if (debug) {
          cout << "filtering: mp " << i << " is the best 4h primitive" << endl;
        }
        oneOf4 = true;
        bestI = i;
        bestChi2 = inMPaths[i].chi2;
      }
      if (i == inMPaths.size() - 1) {  //You can't compare the last one with the next one
        if (oneOf4) {
          outMPaths.push_back(inMPaths[bestI]);
        } else {
          for (unsigned int j = i - back; j <= i; j++) {
            outMPaths.push_back(inMPaths[j]);
          }
        }
      } else {
        if (areCousins(inMPaths[i], inMPaths[i + 1]) == 0) {  //they arent cousins
          if (debug) {
            cout << "mp " << i << " and mp " << i + 1 << " are not cousins" << endl;
          }
          if (oneOf4) {
            outMPaths.push_back(inMPaths[bestI]);
            if (debug)
              cout << "kept4 mp " << bestI << endl;
            oneOf4 = false;  //reset 4h variables
            bestI = -1;
            bestChi2 = 9999;
          } else {
            for (unsigned int j = i - back; j <= i; j++) {
              outMPaths.push_back(inMPaths[j]);
              if (debug)
                cout << "kept3 mp " << j << endl;
            }
          }
          back = 0;
        } else {  // they are cousins
          back++;
        }
      }
    }
  } else if (inMPaths.size() == 1) {
    outMPaths.push_back(inMPaths[0]);
  }
}

void MPQualityEnhancerFilter::filterUnique(std::vector<metaPrimitive> &inMPaths,
                                           std::vector<metaPrimitive> &outMPaths) {
  double xTh = 0.001;
  double tPhiTh = 0.001;
  double t0Th = 0.001;
  for (size_t i = 0; i < inMPaths.size(); i++) {
    bool visto = false;
    for (size_t j = i + 1; j < inMPaths.size(); j++) {
      if ((fabs(inMPaths[i].x - inMPaths[j].x) <= xTh) && (fabs(inMPaths[i].tanPhi - inMPaths[j].tanPhi) <= tPhiTh) &&
          (fabs(inMPaths[i].t0 - inMPaths[j].t0) <= t0Th))
        visto = true;
    }
    if (!visto)
      outMPaths.push_back(inMPaths[i]);
  }
}

void MPQualityEnhancerFilter::printmP(metaPrimitive mP) {
  DTSuperLayerId slId(mP.rawId);
  std::cout << slId << "\t"
            << " " << setw(2) << left << mP.wi1 << " " << setw(2) << left << mP.wi2 << " " << setw(2) << left << mP.wi3
            << " " << setw(2) << left << mP.wi4 << " " << setw(5) << left << mP.tdc1 << " " << setw(5) << left
            << mP.tdc2 << " " << setw(5) << left << mP.tdc3 << " " << setw(5) << left << mP.tdc4 << " " << setw(10)
            << right << mP.x << " " << setw(9) << left << mP.tanPhi << " " << setw(5) << left << mP.t0 << " "
            << setw(13) << left << mP.chi2 << " r:" << rango(mP);
}
