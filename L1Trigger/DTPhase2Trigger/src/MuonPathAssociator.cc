#include "L1Trigger/DTPhase2Trigger/interface/MuonPathAssociator.h"
#include "L1Trigger/DTPhase2Trigger/interface/MuonPathAnalyzerPerSL.h"
#include "L1Trigger/DTPhase2Trigger/interface/constants.h"

using namespace edm;
using namespace std;



// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathAssociator::MuonPathAssociator(const ParameterSet& pset) {
    // Obtention of parameters
    debug            = pset.getUntrackedParameter<Bool_t>("debug");
    dT0_correlate_TP = pset.getUntrackedParameter<double>("dT0_correlate_TP");
    minx_match_2digis = pset.getUntrackedParameter<double>("minx_match_2digis");
    chi2corTh = pset.getUntrackedParameter<double>("chi2corTh");

    if (debug) cout <<"MuonPathAssociator: constructor" << endl;

    //shift
    int rawId;
    shift_filename = pset.getParameter<edm::FileInPath>("shift_filename");
    std::ifstream ifin3(shift_filename.fullPath());
    double shift;
    if (ifin3.fail()) {
      throw cms::Exception("Missing Input File")
        << "MuonPathAnalyzerPerSL::MuonPathAnalyzerPerSL() -  Cannot find " << shift_filename.fullPath();
    }
    while (ifin3.good()){
	ifin3 >> rawId >> shift;
	shiftinfo[rawId]=shift;
    }



}


MuonPathAssociator::~MuonPathAssociator() {
    if (debug) cout <<"MuonPathAssociator: destructor" << endl;
}



// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathAssociator::initialise(const edm::EventSetup& iEventSetup) {
    if(debug) cout << "MuonPathAssociator::initialiase" << endl;

    iEventSetup.get<MuonGeometryRecord>().get(dtGeo);//1103
}


void MuonPathAssociator::run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, 
			     edm::Handle<DTDigiCollection> digis,
			     std::vector<metaPrimitive> &inMPaths, 
			     std::vector<metaPrimitive> &outMPaths) 
{
  
    if (dT0_correlate_TP)  correlateMPaths(digis, inMPaths,outMPaths);
    else { 
	for (auto metaPrimitiveIt = inMPaths.begin(); metaPrimitiveIt != inMPaths.end(); ++metaPrimitiveIt)
	    outMPaths.push_back(*metaPrimitiveIt);
    }
}

void MuonPathAssociator::finish() {
    if (debug) cout <<"MuonPathAssociator: finish" << endl;
};

void MuonPathAssociator::correlateMPaths(edm::Handle<DTDigiCollection> dtdigis,
					 std::vector<metaPrimitive> &inMPaths, 
					 std::vector<metaPrimitive> &outMPaths) {

  
    //Silvia's code for correlationg filteredMetaPrimitives
  
    if(debug) std::cout<<"starting correlation"<<std::endl;
  
  
    for(int wh=-2;wh<=2;wh++){
	for(int st=1;st<=4;st++){
	    for(int se=1;se<=14;se++){
		if(se>=13&&st!=4)continue;
	
		DTChamberId ChId(wh,st,se);
		DTSuperLayerId sl1Id(wh,st,se,1);
		DTSuperLayerId sl3Id(wh,st,se,3);
	
		//filterSL1
		std::vector<metaPrimitive> SL1metaPrimitives;
		for(auto metaprimitiveIt = inMPaths.begin();metaprimitiveIt!=inMPaths.end();++metaprimitiveIt)
		    if(metaprimitiveIt->rawId==sl1Id.rawId())
			SL1metaPrimitives.push_back(*metaprimitiveIt);
	
		//filterSL3
		std::vector<metaPrimitive> SL3metaPrimitives;
		for(auto metaprimitiveIt = inMPaths.begin();metaprimitiveIt!=inMPaths.end();++metaprimitiveIt)
		    if(metaprimitiveIt->rawId==sl3Id.rawId())
			SL3metaPrimitives.push_back(*metaprimitiveIt);
	
		if(SL1metaPrimitives.size()==0 and SL3metaPrimitives.size()==0) continue;
	
		if(debug) std::cout<<"correlating "<<SL1metaPrimitives.size()<<" metaPrim in SL1 and "<<SL3metaPrimitives.size()<<" in SL3 for "<<sl3Id<<std::endl;
	
		bool at_least_one_correlation=false;
	
		//SL1-SL3
	
		for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end(); ++SL1metaPrimitive){
		    for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end(); ++SL3metaPrimitive){
			if(fabs(SL1metaPrimitive->t0-SL3metaPrimitive->t0) < dT0_correlate_TP){//time match
			    double PosSL1=SL1metaPrimitive->x;
			    double PosSL3=SL3metaPrimitive->x;
			    double NewSlope=(PosSL1-PosSL3)/23.5;     
			    double MeanT0=(SL1metaPrimitive->t0+SL3metaPrimitive->t0)/2;
			    double MeanPos=(PosSL3+PosSL1)/2;
			    //double newChi2=(SL1metaPrimitive->chi2+SL3metaPrimitive->chi2)*0.5;//to be recalculated
			   
			    DTSuperLayerId SLId1(SL1metaPrimitive->rawId);
             		    DTSuperLayerId SLId3(SL3metaPrimitive->rawId);

           		    DTWireId wireId1(SLId1,2,1);
           	     	    DTWireId wireId3(SLId3,2,1);

			    double xH[8], xReco[8];
			    int  wi[8], tdc[8], lat[8];
			    for (int i = 0; i<8; i++){ xH[i]=0; xReco[i]=0;} 
			    wi[0]=SL1metaPrimitive->wi1;tdc[0]=SL1metaPrimitive->tdc1; lat[0]=SL1metaPrimitive->lat1;  
			    wi[1]=SL1metaPrimitive->wi2;tdc[1]=SL1metaPrimitive->tdc2; lat[1]=SL1metaPrimitive->lat2;  
			    wi[2]=SL1metaPrimitive->wi3;tdc[2]=SL1metaPrimitive->tdc3; lat[2]=SL1metaPrimitive->lat3;  
			    wi[3]=SL1metaPrimitive->wi4;tdc[3]=SL1metaPrimitive->tdc4; lat[3]=SL1metaPrimitive->lat4;  
			    wi[4]=SL3metaPrimitive->wi1;tdc[4]=SL3metaPrimitive->tdc1; lat[4]=SL3metaPrimitive->lat1;  
			    wi[5]=SL3metaPrimitive->wi2;tdc[5]=SL3metaPrimitive->tdc2; lat[5]=SL3metaPrimitive->lat2;  
			    wi[6]=SL3metaPrimitive->wi3;tdc[6]=SL3metaPrimitive->tdc3; lat[6]=SL3metaPrimitive->lat3;  
			    wi[7]=SL3metaPrimitive->wi4;tdc[7]=SL3metaPrimitive->tdc4; lat[7]=SL3metaPrimitive->lat4;  
			
           		    for (int i=0; i<4; i++){
				if (wi[i]!=-1) {
				    if (i%2==0){
					 xH[i] = shiftinfo[wireId1.rawId()]+(42.*(double)wi[i]+ 21. + DRIFT_SPEED*((double)tdc[i]-MeanT0)*(-1.+2.*(double)lat[i]))/10;
					 xReco[i] = MeanPos + (23.5/2 - ((double)i-1.5)*1.3)*NewSlope;
				    }
				    if (i%2!=0){
					 xH[i] = shiftinfo[wireId1.rawId()]+(42.*(double)wi[i]+     + DRIFT_SPEED*((double)tdc[i]-MeanT0)*(-1+2*(double)lat[i]))/10;
					 xReco[i] = MeanPos + (23.5/2 - ((double)i-1.5)*1.3)*NewSlope;
			            }
				}
			    } 
           		    for (int i=4; i<8; i++){
				if (wi[i]!=-1) {
				    if (i%2==0){
					 xH[i] = shiftinfo[wireId3.rawId()]+(42.*(double)wi[i]+ 21. + DRIFT_SPEED*((double)tdc[i]-MeanT0)*(-1+2*(double)lat[i]))/10;
					 xReco[i] = MeanPos + (-23.5/2 - ((double)i-4-1.5)*1.3)*NewSlope;
				    }
				    if (i%2!=0){
					 xH[i] = shiftinfo[wireId3.rawId()]+(42.*(double)wi[i]+     + DRIFT_SPEED*((double)tdc[i]-MeanT0)*(-1+2*(double)lat[i]))/10;
					 xReco[i] = MeanPos + (-23.5/2 - ((double)i-4-1.5)*1.3)*NewSlope;
			            }
				}
			    }
			    double newChi2 = 0; 
			    for (int i = 0; i<8; i++){
				newChi2 = newChi2 + (xH[i]-xReco[i])*(xH[i]-xReco[i]);
			    } 
			    if(newChi2>chi2corTh) continue;

	                    int quality = 0;
			    if(SL3metaPrimitive->quality <= 2 and SL1metaPrimitive->quality <=2) quality=6;
	      
			    if((SL3metaPrimitive->quality >= 3 && SL1metaPrimitive->quality <=2)
			       or (SL1metaPrimitive->quality >= 3 && SL3metaPrimitive->quality <=2) ) quality=8;
	      
			    if(SL3metaPrimitive->quality >= 3 && SL1metaPrimitive->quality >=3) quality=9;
			    
			    double z=0;
			    if(ChId.station()>=3)z=1.8;
			    GlobalPoint jm_x_cmssw_global = dtGeo->chamber(ChId)->toGlobal(LocalPoint(MeanPos,0.,z));//Jm_x is already extrapolated to the middle of the SL
			    int thisec = ChId.sector();
			    if(se==13) thisec = 4;
			    if(se==14) thisec = 10;
			    double phi= jm_x_cmssw_global.phi()-0.5235988*(thisec-1);
			    double psi=atan(NewSlope);
			    double phiB=hasPosRF(ChId.wheel(),ChId.sector()) ? psi-phi :-psi-phi ;
			    
			    outMPaths.push_back(metaPrimitive({ChId.rawId(),MeanT0,MeanPos,NewSlope,phi,phiB,newChi2,quality,
					    SL1metaPrimitive->wi1,SL1metaPrimitive->tdc1,SL1metaPrimitive->lat1,
					    SL1metaPrimitive->wi2,SL1metaPrimitive->tdc2,SL1metaPrimitive->lat2,
					    SL1metaPrimitive->wi3,SL1metaPrimitive->tdc3,SL1metaPrimitive->lat3,
					    SL1metaPrimitive->wi4,SL1metaPrimitive->tdc4,SL1metaPrimitive->lat4,
					    SL3metaPrimitive->wi1,SL3metaPrimitive->tdc1,SL1metaPrimitive->lat5,
					    SL3metaPrimitive->wi2,SL3metaPrimitive->tdc2,SL1metaPrimitive->lat6,
					    SL3metaPrimitive->wi3,SL3metaPrimitive->tdc3,SL1metaPrimitive->lat7,
					    SL3metaPrimitive->wi4,SL3metaPrimitive->tdc4,SL1metaPrimitive->lat8,
					    -1
					    }));
			    at_least_one_correlation=true;
			}
		    }
	  
		    if(at_least_one_correlation==false){//no correlation was found, trying with pairs of two digis in the other SL
			int matched_digis=0;
			double minx=minx_match_2digis;
			double min2x=minx_match_2digis;
			int best_tdc=-1;
			int next_tdc=-1;
			int best_wire=-1;
			int next_wire=-1;
			int best_layer=-1;
			int next_layer=-1;
			int best_lat=-1;
			int next_lat=-1;
			int lat=-1;
	    
			for (auto dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
			    const DTLayerId dtLId = (*dtLayerId_It).first;
			    DTSuperLayerId dtSLId(dtLId);
			    if(dtSLId.rawId()!=sl3Id.rawId()) continue;
			    double l_shift=0;
			    if(dtLId.layer()==4)l_shift=1.95;
			    if(dtLId.layer()==3)l_shift=0.65;
			    if(dtLId.layer()==2)l_shift=-0.65;
			    if(dtLId.layer()==1)l_shift=-1.95;
			    double x_inSL3=SL1metaPrimitive->x-SL1metaPrimitive->tanPhi*(23.5+l_shift);
			    for (auto digiIt = ((*dtLayerId_It).second).first;digiIt!=((*dtLayerId_It).second).second; ++digiIt){
				DTWireId wireId(dtLId,(*digiIt).wire());
				int x_wire = shiftinfo[wireId.rawId()]+((*digiIt).time()-SL1metaPrimitive->t0)*0.00543; 
				int x_wire_left = shiftinfo[wireId.rawId()]-((*digiIt).time()-SL1metaPrimitive->t0)*0.00543; 
				lat=1;
				if(fabs(x_inSL3-x_wire)>fabs(x_inSL3-x_wire_left)){
				    x_wire=x_wire_left; //choose the closest laterality
				    lat=0;
				}
				if(fabs(x_inSL3-x_wire)<minx){
				    minx=fabs(x_inSL3-x_wire);
				    next_wire=best_wire;
				    next_tdc=best_tdc;
				    next_layer=best_layer;
				    next_lat=best_lat;
		  
				    best_wire=(*digiIt).wire();
				    best_tdc=(*digiIt).time();
				    best_layer=dtLId.layer();
				    best_lat=lat;
				    matched_digis++;
				} else if ((fabs(x_inSL3-x_wire)>=minx)&&(fabs(x_inSL3-x_wire)<min2x)){
				    min2x=fabs(x_inSL3-x_wire);
                                    next_wire=(*digiIt).wire();
                                    next_tdc=(*digiIt).time();
                                    next_layer=dtLId.layer();
                                    next_lat=lat;
				    matched_digis++;
				}
			    }
	      
			}
			if(matched_digis>=2 and best_layer!=-1 and next_layer!=-1){
			    int new_quality=7;
			    if(SL1metaPrimitive->quality<=2) new_quality=5;
	      
			    int wi1=-1;int tdc1=-1;int lat1=-1;
			    int wi2=-1;int tdc2=-1;int lat2=-1;
			    int wi3=-1;int tdc3=-1;int lat3=-1;
			    int wi4=-1;int tdc4=-1;int lat4=-1;
	      
			    if(next_layer==1) {wi1=next_wire; tdc1=next_tdc; lat1=next_lat;}
			    if(next_layer==2) {wi2=next_wire; tdc2=next_tdc; lat2=next_lat;}
			    if(next_layer==3) {wi3=next_wire; tdc3=next_tdc; lat3=next_lat;}
			    if(next_layer==4) {wi4=next_wire; tdc4=next_tdc; lat4=next_lat;}
	      
			    if(best_layer==1) {wi1=best_wire; tdc1=best_tdc; lat1=best_lat;}
			    if(best_layer==2) {wi2=best_wire; tdc2=best_tdc; lat2=best_lat;}
			    if(best_layer==3) {wi3=best_wire; tdc3=best_tdc; lat3=best_lat;}
			    if(best_layer==4) {wi4=best_wire; tdc4=best_tdc; lat4=best_lat;}    
			    
			    outMPaths.push_back(metaPrimitive({ChId.rawId(),SL1metaPrimitive->t0,SL1metaPrimitive->x,SL1metaPrimitive->tanPhi,SL1metaPrimitive->phi,SL1metaPrimitive->phiB,SL1metaPrimitive->chi2,
					    new_quality,
					    SL1metaPrimitive->wi1,SL1metaPrimitive->tdc1,SL1metaPrimitive->lat1,
					    SL1metaPrimitive->wi2,SL1metaPrimitive->tdc2,SL1metaPrimitive->lat2,
					    SL1metaPrimitive->wi3,SL1metaPrimitive->tdc3,SL1metaPrimitive->lat3,
					    SL1metaPrimitive->wi4,SL1metaPrimitive->tdc4,SL1metaPrimitive->lat4,
					    wi1,tdc1,lat1,
					    wi2,tdc2,lat2,
					    wi3,tdc3,lat3,
					    wi4,tdc4,lat4,
					    -1
					    }));
			    at_least_one_correlation=true;
			}
		    }
		}
	
		//finish SL1-SL3

		//SL3-SL1
		for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end(); ++SL3metaPrimitive){
		    for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end(); ++SL1metaPrimitive){
			if(fabs(SL1metaPrimitive->t0-SL3metaPrimitive->t0) < dT0_correlate_TP){//time match
			    //this comb was already filled up in the previous loop now we just want to know if there was at least one match
			    at_least_one_correlation=true;
			}
		    }
	  
		    if(at_least_one_correlation==false){//no correlation was found, trying with pairs of two digis in the other SL
	    
			int matched_digis=0;
			double minx=minx_match_2digis;
			double min2x=minx_match_2digis;
			int best_tdc=-1;
			int next_tdc=-1;
			int best_wire=-1;
			int next_wire=-1;
			int best_layer=-1;
			int next_layer=-1;
			int best_lat=-1;
			int next_lat=-1;
			int lat=-1;
			
			for (auto dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
			    const DTLayerId dtLId = (*dtLayerId_It).first;
			    DTSuperLayerId dtSLId(dtLId);
			    if(dtSLId.rawId()!=sl1Id.rawId()) continue;
			    double l_shift=0;
			    if(dtLId.layer()==4)l_shift=1.95;
			    if(dtLId.layer()==3)l_shift=0.65;
			    if(dtLId.layer()==2)l_shift=-0.65;
			    if(dtLId.layer()==1)l_shift=-1.95;
			    double x_inSL1=SL3metaPrimitive->x+SL3metaPrimitive->tanPhi*(23.5-l_shift);
			    for (auto digiIt = ((*dtLayerId_It).second).first;digiIt!=((*dtLayerId_It).second).second; ++digiIt){
				DTWireId wireId(dtLId,(*digiIt).wire());
				int x_wire = shiftinfo[wireId.rawId()]+((*digiIt).time()-SL3metaPrimitive->t0)*0.00543; 
				int x_wire_left = shiftinfo[wireId.rawId()]-((*digiIt).time()-SL3metaPrimitive->t0)*0.00543; 
				lat=1;
				if(fabs(x_inSL1-x_wire)>fabs(x_inSL1-x_wire_left)){
				    x_wire=x_wire_left; //choose the closest laterality
				    lat=0;
				}
				if(fabs(x_inSL1-x_wire)<minx){
				    minx=fabs(x_inSL1-x_wire);
				    next_wire=best_wire;
				    next_tdc=best_tdc;
				    next_layer=best_layer;
				    next_lat=best_lat;
		  
				    best_wire=(*digiIt).wire();
				    best_tdc=(*digiIt).time();
				    best_layer=dtLId.layer();
				    best_lat=lat;
				    matched_digis++;
				} else if((fabs(x_inSL1-x_wire)>=minx)&&(fabs(x_inSL1-x_wire<min2x))){
				    minx=fabs(x_inSL1-x_wire);
                                    next_wire=(*digiIt).wire();
                                    next_tdc=(*digiIt).time();
                                    next_layer=dtLId.layer();
                                    next_lat=lat;
				    matched_digis++;
				}
			    }
	      
			}
			if(matched_digis>=2 and best_layer!=-1 and next_layer!=-1){
			    int new_quality=7;
			    if(SL3metaPrimitive->quality<=2) new_quality=5;
	      
			    int wi1=-1;int tdc1=-1;int lat1=-1;
			    int wi2=-1;int tdc2=-1;int lat2=-1;
			    int wi3=-1;int tdc3=-1;int lat3=-1;
			    int wi4=-1;int tdc4=-1;int lat4=-1;
	      
			    if(next_layer==1) {wi1=next_wire; tdc1=next_tdc; lat1=next_lat;}
			    if(next_layer==2) {wi2=next_wire; tdc2=next_tdc; lat2=next_lat;}
			    if(next_layer==3) {wi3=next_wire; tdc3=next_tdc; lat3=next_lat;}
			    if(next_layer==4) {wi4=next_wire; tdc4=next_tdc; lat4=next_lat;}
	      
			    if(best_layer==1) {wi1=best_wire; tdc1=best_tdc; lat1=best_lat;}
			    if(best_layer==2) {wi2=best_wire; tdc2=best_tdc; lat2=best_lat;}
			    if(best_layer==3) {wi3=best_wire; tdc3=best_tdc; lat3=best_lat;}
			    if(best_layer==4) {wi4=best_wire; tdc4=best_tdc; lat4=best_lat;}    
			    
			    outMPaths.push_back(metaPrimitive({ChId.rawId(),SL3metaPrimitive->t0,SL3metaPrimitive->x,SL3metaPrimitive->tanPhi,SL3metaPrimitive->phi,SL3metaPrimitive->phiB,SL3metaPrimitive->chi2,
					    new_quality,
					    wi1,tdc1,lat1,
					    wi2,tdc2,lat2,
					    wi3,tdc3,lat3,
					    wi4,tdc4,lat4,
					    SL3metaPrimitive->wi1,SL3metaPrimitive->tdc1,SL3metaPrimitive->lat1,
					    SL3metaPrimitive->wi2,SL3metaPrimitive->tdc2,SL3metaPrimitive->lat2,
					    SL3metaPrimitive->wi3,SL3metaPrimitive->tdc3,SL3metaPrimitive->lat3,
					    SL3metaPrimitive->wi4,SL3metaPrimitive->tdc4,SL3metaPrimitive->lat4,
					    -1
					    }));
			    at_least_one_correlation=true;
			}
		    }
		}
	
		//finish SL3-SL1
		if(at_least_one_correlation==false){
		    if(debug) std::cout<<"correlation we found zero correlations, adding both collections as they are to the outMPaths"<<std::endl;
		    if(debug) std::cout<<"correlation sizes:"<<SL1metaPrimitives.size()<<" "<<SL3metaPrimitives.size()<<std::endl;
		    for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end(); ++SL1metaPrimitive){
			DTSuperLayerId SLId(SL1metaPrimitive->rawId);
			DTChamberId(SLId.wheel(),SLId.station(),SLId.sector());
			outMPaths.push_back(metaPrimitive({ChId.rawId(),SL1metaPrimitive->t0,SL1metaPrimitive->x,SL1metaPrimitive->tanPhi,SL1metaPrimitive->phi,SL1metaPrimitive->phiB,SL1metaPrimitive->chi2,SL1metaPrimitive->quality,
					SL1metaPrimitive->wi1,SL1metaPrimitive->tdc1,SL1metaPrimitive->lat1,
					SL1metaPrimitive->wi2,SL1metaPrimitive->tdc2,SL1metaPrimitive->lat2,
					SL1metaPrimitive->wi3,SL1metaPrimitive->tdc3,SL1metaPrimitive->lat3,
					SL1metaPrimitive->wi4,SL1metaPrimitive->tdc4,SL1metaPrimitive->lat4,
					-1,-1,-1,
					-1,-1,-1,
					-1,-1,-1,
					-1,-1,-1,
					-1
					}));
		    }
		    for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end(); ++SL3metaPrimitive){
			DTSuperLayerId SLId(SL3metaPrimitive->rawId);
			DTChamberId(SLId.wheel(),SLId.station(),SLId.sector());
			outMPaths.push_back(metaPrimitive({ChId.rawId(),SL3metaPrimitive->t0,SL3metaPrimitive->x,SL3metaPrimitive->tanPhi,SL3metaPrimitive->phi,SL3metaPrimitive->phiB,SL3metaPrimitive->chi2,SL3metaPrimitive->quality,
					-1,-1,-1,
					-1,-1,-1,
					-1,-1,-1,
					-1,-1,-1,
					SL3metaPrimitive->wi1,SL3metaPrimitive->tdc1,SL3metaPrimitive->lat1,
					SL3metaPrimitive->wi2,SL3metaPrimitive->tdc2,SL3metaPrimitive->lat2,
					SL3metaPrimitive->wi3,SL3metaPrimitive->tdc3,SL3metaPrimitive->lat3,
					SL3metaPrimitive->wi4,SL3metaPrimitive->tdc4,SL3metaPrimitive->lat4,
					-1
					}));
		    }
		}
	
		SL1metaPrimitives.clear();
		SL1metaPrimitives.erase(SL1metaPrimitives.begin(),SL1metaPrimitives.end());
		SL3metaPrimitives.clear();
		SL3metaPrimitives.erase(SL3metaPrimitives.begin(),SL3metaPrimitives.end());
	    }
	}
    }
}
	
/*
  void MuonPathAssociator::associate(MuonPath *mpath) {
  
  // First try to match 
  if (mpath->getNPrimitivesUp()>=3 && mpath->getNPrimitivesDown()>=3) {
  if(fabs(mpath->getBxTimeValue(0)-mpath->getBxTimeValue(2)) < dT0_correlate_TP) { //time match
  float PosSL1=mpath->getHorizPos(0);
  float PosSL3=mpath->getHorizPos(2);
  float NewSlope=(PosSL1-PosSL3)/23.5;     
  float MeanT0=(mpath->getBxTimeValue(0)+mpath->getBxTimeValue(2))/2;
  float MeanPos=(PosSL3+PosSL1)/2;
  float newChi2=(mpath->getChiSq(0)+mpath->getChiSq(2))*0.5;//to be recalculated
  MP_QUALITY quality=NOPATH;
      
  if (mpath->getQuality(0) <=LOWQ and mpath->getQuality(2) <=LOWQ)  quality=LOWLOWQ;
  if ((mpath->getQuality(0) >=HIGHQ and mpath->getQuality(2) <=LOWQ) or 
  (mpath->getQuality(0) <=LOWQ and mpath->getQuality(2) >=HIGHQ))
  quality=HIGHLOWQ;
  if (mpath->getQuality(0) >=3 and mpath->getQuality(2) >=3)  quality=HIGHHIGHQ;
      
  DTChamberId ChId(mpath->getRawId());
  GlobalPoint jm_x_cmssw_global = dtGeo->chamber(ChId)->toGlobal(LocalPoint(MeanPos,0.,0.));//jm_x is already extrapolated to the middle of the SL
  int thisec = ChId.sector();
  float phi= jm_x_cmssw_global.phi()-0.5235988*(thisec-1);
  float psi=atan(NewSlope);
  float phiB=(hasPosRF(ChId.wheel(),ChId.sector())) ? psi-phi :-psi-phi ;
			
  mpath->setBxTimeValue(MeanT0);
  mpath->setTanPhi(NewSlope);
  mpath->setHorizPos(MeanPos);
  mpath->setPhi(phi);
  mpath->setPhiB(phiB);
  mpath->setChiSq(newChi2);
  mpath->setQuality(quality);
  }
  }
  else if (mpath->getNPrimitivesUp()>=3 && mpath->getNPrimitivesDown()<3 && mpath->getNPrimitivesDown()>0 ) {
  // IF this is not the case try to confirm with other SL: 
  mpath->setBxTimeValue(mpath->getBxTimeValue(2));
  mpath->setTanPhi(mpath->getTanPhi(2));
  mpath->setHorizPos(mpath->getHorizPos(2));
  mpath->setPhi(mpath->getPhi(2));
  mpath->setPhiB(mpath->getPhiB(2));
  mpath->setChiSq(mpath->getChiSq(2));

  if (mpath->getQuality(2) == HIGHQ) 
  mpath->setQuality(CHIGHQ);
  else if (mpath->getQuality(2) == LOWQ) 
  mpath->setQuality(CLOWQ);
    
  }
  else if (mpath->getNPrimitivesDown()>=3 && mpath->getNPrimitivesDown()<3 && mpath->getNPrimitivesDown()>0 ) {
  // IF this is not the case try to confirm with other SL: 
  mpath->setBxTimeValue(mpath->getBxTimeValue(2));
  mpath->setTanPhi(mpath->getTanPhi(2));
  mpath->setHorizPos(mpath->getHorizPos(2));
  mpath->setPhi(mpath->getPhi(2));
  mpath->setPhiB(mpath->getPhiB(2));
  mpath->setChiSq(mpath->getChiSq(2));
  mpath->setQuality(CHIGHQ);

  if (mpath->getQuality(0) == HIGHQ) 
  mpath->setQuality(CHIGHQ);
  else if (mpath->getQuality(0) == LOWQ) 
  mpath->setQuality(CLOWQ);
    
  }
  
  }

*/

