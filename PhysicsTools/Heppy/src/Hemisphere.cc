#include "PhysicsTools/Heppy/interface/Hemisphere.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace std;

using std::vector;
using std::cout;
using std::endl;

namespace heppy {

// constructor specifying the seed and association methods
Hemisphere::Hemisphere(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector,
	vector<float> E_vector, int seed_method, int hemisphere_association_method) : Object_Px(Px_vector),
	Object_Py(Py_vector), Object_Pz(Pz_vector), Object_E(E_vector), seed_meth(seed_method),
	hemi_meth(hemisphere_association_method), status(0),
	dRminSeed1(0.5),nItermax(100),
	rejectISR(0), rejectISRPt(0), rejectISRPtmax(10000.),
rejectISRDR(0), rejectISRDRmax(100.), dbg(0)  {
	for(int i = 0; i < (int) Object_Px.size(); i++){
		Object_Noseed.push_back(0);
		Object_Noassoc.push_back(0);
	}
	numLoop =0;
}

// constructor without specification of the seed and association methods
// in this case, the latter must be given by calling SetMethod before invoking reconstruct()
Hemisphere::Hemisphere(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector,
	vector<float> E_vector) : Object_Px(Px_vector),
	Object_Py(Py_vector), Object_Pz(Pz_vector), Object_E(E_vector), seed_meth(0),
	hemi_meth(0), status(0),
	dRminSeed1(0.5),nItermax(100),
	rejectISR(0), rejectISRPt(0), rejectISRPtmax(10000.),
rejectISRDR(0), rejectISRDRmax(100.), dbg(0)  {
	for(int i = 0; i < (int) Object_Px.size(); i++){
		Object_Noseed.push_back(0);
		Object_Noassoc.push_back(0);
	}
	numLoop =0;
}

vector<float> Hemisphere::getAxis1(){
	if (status != 1) {
		if (rejectISR == 0) {
			this->Reconstruct();
		} else {
			this->RejectISR();
		}
	}
	return Axis1;
}
vector<float> Hemisphere::getAxis2(){
	if (status != 1) {
		if (rejectISR == 0) {
			this->Reconstruct();
		} else {
			this->RejectISR();
		}
	}
	return Axis2;
}

vector<int> Hemisphere::getGrouping(){
	if (status != 1) {
		if (rejectISR == 0) {
			this->Reconstruct();
		} else {
			this->RejectISR();
		}
	}
	return Object_Group;
}

int Hemisphere::Reconstruct(){

// Performs the actual hemisphere reconstrucion
//
// definition of the vectors used internally:
// Object_xxx :
//        xxx = Px, Py, Pz, E for input values
//        xxx = P, Pt, Eta, Phi, Group for internal use
// Axis1 : final hemisphere axis 1
// Axis2 : final hemisphere axis 2
// Sum1_xxx : hemisphere 1 being updated during the association iterations
// Sum2_xxx : hemisphere 2 being updated during the association iterations
// NewAxis1_xxx, NewAxis1_xxx : temporary axes for calculation in association methods 2 and 3


	numLoop=0; // initialize numLoop for Zero
	int vsize = (int) Object_Px.size();
	if((int) Object_Py.size() != vsize || (int) Object_Pz.size() != vsize){
		cout << "WARNING!!!!! Input vectors have different size! Fix it!" << endl;
		return 0;
	}
	if (dbg > 0) {
//    cout << " Hemisphere method, vsn = " << hemivsn << endl;
		cout << " Hemisphere method " << endl;
	}

	// clear some vectors if method reconstruct() is called again
	if (!Object_P.empty()){
		Object_P.clear();
		Object_Pt.clear();
		Object_Eta.clear();
		Object_Phi.clear();
		Object_Group.clear();
		Axis1.clear();
		Axis2.clear();
	}
	// initialize the vectors
	for(int j = 0; j < vsize; ++j){
		Object_P.push_back(0);
		Object_Pt.push_back(0);
		Object_Eta.push_back(0);
		Object_Phi.push_back(0);
		Object_Group.push_back(0);
	}
	for(int j = 0; j < 5; ++j){
		Axis1.push_back(0);
		Axis2.push_back(0);
	}

	// compute additional quantities for vectors Object_xxx
	float theta;
	for (int i = 0; i <vsize; ++i){
		Object_P[i] = sqrt(Object_Px[i]*Object_Px[i]+Object_Py[i]*Object_Py[i]+Object_Pz[i]*Object_Pz[i]);
		if (Object_P[i] > Object_E[i]+0.001) {
			cout << "WARNING!!!!! Object " << i << " has E = " << Object_E[i]
				<< " less than P = " << Object_P[i] << " *** Fix it!" << endl;
			return 0;
		}
		Object_Pt[i] = sqrt(Object_Px[i]*Object_Px[i]+Object_Py[i]*Object_Py[i]);
	// protection for div by 0
		if (fabs(Object_Pz[i]) > 0.001) {
			theta = atan(sqrt(Object_Px[i]*Object_Px[i]+Object_Py[i]*Object_Py[i])/Object_Pz[i]);
		}
		else {
			theta = 1.570796327;
		}
		if (theta < 0.) {theta = theta + 3.141592654;}
		Object_Eta[i] = -log(tan(0.5*theta));
		Object_Phi[i] = atan2(Object_Py[i], Object_Px[i]);
		if (dbg > 0) {
			cout << " Object " << i << " Eta = " << Object_Eta[i]
				<< " Phi = " << Object_Phi[i] << endl;
		}
	}

	if (dbg > 0) {
		cout << endl;
		cout << " Seeding method = " << seed_meth << endl;
	}
	// I_Max and J_Max are indices of the seeds in the vectors
	int I_Max = -1;
	int J_Max = -1;

	// determine the seeds for seed method 1
	if (seed_meth == 1) {

		float P_Max = 0.;
		float DeltaRP_Max = 0.;

	// take highest momentum object as first seed
		for (int i = 0; i < vsize; ++i){
			Object_Group[i] = 0;
		//cout << "Object_Px[i] = " << Object_Px[i] << ", Object_Py[i] = " << Object_Py[i]
		//<< ", Object_Pz[i] = " << Object_Pz[i] << "  << endl;
			if (Object_Noseed[i] == 0 && P_Max < Object_P[i]){
				P_Max = Object_P[i];
				I_Max = i;
			}
		}
	// if 1st seed is found, save it as initial hemisphere 1 axis
		if (I_Max >= 0){
			Axis1[0] = Object_Px[I_Max] /  Object_P[I_Max];
			Axis1[1] = Object_Py[I_Max] /  Object_P[I_Max];
			Axis1[2] = Object_Pz[I_Max] /  Object_P[I_Max];
			Axis1[3] = Object_P[I_Max];
			Axis1[4] = Object_E[I_Max];
		} else {
		// cout << " This is an empty event." << endl;
			return 0;
		}

	// take as second seed the object with largest DR*P w.r.t. the first seed
		for (int i = 0; i < vsize; ++i){
		  //			float DeltaR = sqrt((Object_Eta[i] - Object_Eta[I_Max])*(Object_Eta[i] - Object_Eta[I_Max])
		  //				+ (Util::DeltaPhi(Object_Phi[i], Object_Phi[I_Max]))*(Util::DeltaPhi(Object_Phi[i], Object_Phi[I_Max]))  );
			float DeltaR = sqrt((Object_Eta[i] - Object_Eta[I_Max])*(Object_Eta[i] - Object_Eta[I_Max])
				+ (deltaPhi(Object_Phi[i], Object_Phi[I_Max]))*(deltaPhi(Object_Phi[i], Object_Phi[I_Max]))  );
			if (Object_Noseed[i] == 0 && DeltaR > dRminSeed1) {
				float DeltaRP = DeltaR * Object_P[i];
				if (DeltaRP > DeltaRP_Max){
					DeltaRP_Max = DeltaRP;
					J_Max = i;
				}
			}
		}
	// if 2nd seed is found, save it as initial hemisphere 2 axis
		if (J_Max >= 0){
			Axis2[0] = Object_Px[J_Max] /  Object_P[J_Max];
			Axis2[1] = Object_Py[J_Max] /  Object_P[J_Max];
			Axis2[2] = Object_Pz[J_Max] /  Object_P[J_Max];
			Axis2[3] = Object_P[J_Max];
			Axis2[4] = Object_E[J_Max];
		} else {
		// cout << " This is a MONOJET." << endl;
			return 0;
		}
		if (dbg > 0) {
			cout << " Axis 1 is Object = " << I_Max << endl;
			cout << " Axis 2 is Object = " << J_Max << endl;
		}

	// determine the seeds for seed methods 2 and 3
	} else if (seed_meth == 2 || seed_meth == 3) {

		float Mass_Max = 0.;
		float InvariantMass = 0.;

	// maximize the invariant mass of two objects
		for (int i = 0; i < vsize; ++i){
			Object_Group[i] = 0;
			if (Object_Noseed[i] == 0){
				for (int j = i+1; j < vsize; ++j){
					if (Object_Noseed[j] == 0){
				// either the invariant mass
						if (seed_meth == 2){
							InvariantMass = (Object_E[i] + Object_E[j])* (Object_E[i] + Object_E[j])
								- (Object_Px[i] + Object_Px[j])* (Object_Px[i] + Object_Px[j])
								- (Object_Py[i] + Object_Py[j])* (Object_Py[i] + Object_Py[j])
								- (Object_Pz[i] + Object_Pz[j])* (Object_Pz[i] + Object_Pz[j]) ;
						}
				// or the transverse mass
						else if (seed_meth == 3){
							float pti = sqrt(Object_Px[i]*Object_Px[i] + Object_Py[i]*Object_Py[i]);
							float ptj = sqrt(Object_Px[j]*Object_Px[j] + Object_Py[j]*Object_Py[j]);
							InvariantMass =  2. * (pti*ptj - Object_Px[i]*Object_Px[j]
								- Object_Py[i]*Object_Py[j] );
						}
						if (Mass_Max < InvariantMass){
							Mass_Max = InvariantMass;
							I_Max = i;
							J_Max = j;
						}
					}
				}
			}
		}

	// if both seeds are found, save them as initial hemisphere axes
		if (J_Max > 0) {
			Axis1[0] = Object_Px[I_Max] /  Object_P[I_Max];
			Axis1[1] = Object_Py[I_Max] /  Object_P[I_Max];
			Axis1[2] = Object_Pz[I_Max] /  Object_P[I_Max];
			Axis1[3] = Object_P[I_Max];
			Axis1[4] = Object_E[I_Max];

			Axis2[0] = Object_Px[J_Max] /  Object_P[J_Max];
			Axis2[1] = Object_Py[J_Max] /  Object_P[J_Max];
			Axis2[2] = Object_Pz[J_Max] /  Object_P[J_Max];
			Axis2[3] = Object_P[J_Max];
			Axis2[4] = Object_E[J_Max];
		} else {
		// cout << " This is a MONOJET." << endl;
			return 0;
		}
		if (dbg > 0) {
			cout << " Axis 1 is Object = " << I_Max << endl;
			cout << " Axis 2 is Object = " << J_Max << endl;
		}

	} else if (seed_meth == 4) {

		float P_Max1 = 0.;
		float P_Max2 = 0.;

	// take largest Pt object as first seed
		for (int i = 0; i < vsize; ++i){
			Object_Group[i] = 0;
			if (Object_Noseed[i] == 0 && P_Max1 < Object_Pt[i]){
				P_Max1 = Object_Pt[i];
				I_Max = i;
			}
		}
		if(I_Max < 0) return 0;

	// take second largest Pt object as second seed, but require dR(seed1, seed2) > dRminSeed1 
		for (int i = 0; i < vsize; ++i){
			if( i == I_Max) continue;
			//			float DeltaR = Util::GetDeltaR(Object_Eta[i], Object_Eta[I_Max], Object_Phi[i], Object_Phi[I_Max]); 
			float DeltaR = deltaR(Object_Eta[i], Object_Eta[I_Max], Object_Phi[i], Object_Phi[I_Max]); 
			if (Object_Noseed[i] == 0 && P_Max2 < Object_Pt[i] && DeltaR > dRminSeed1){
				P_Max2 = Object_Pt[i];
				J_Max = i;
			}
		}
		if(J_Max < 0) return 0;

	// save first seed as initial hemisphere 1 axis
		if (I_Max >= 0){
			Axis1[0] = Object_Px[I_Max] /  Object_P[I_Max];
			Axis1[1] = Object_Py[I_Max] /  Object_P[I_Max];
			Axis1[2] = Object_Pz[I_Max] /  Object_P[I_Max];
			Axis1[3] = Object_P[I_Max];
			Axis1[4] = Object_E[I_Max];
		} 

	// save second seed as initial hemisphere 2 axis
		if (J_Max >= 0){
			Axis2[0] = Object_Px[J_Max] /  Object_P[J_Max];
			Axis2[1] = Object_Py[J_Max] /  Object_P[J_Max];
			Axis2[2] = Object_Pz[J_Max] /  Object_P[J_Max];
			Axis2[3] = Object_P[J_Max];
			Axis2[4] = Object_E[J_Max];
		}
		
		if (dbg > 0) {
			cout << " Axis 1 is Object = " << I_Max  << " with Pt " << Object_Pt[I_Max]<< endl;
			cout << " Axis 2 is Object = " << J_Max  << " with Pt " << Object_Pt[J_Max]<< endl;
		}

	} else if ( !(seed_meth == 0 && (hemi_meth == 8 || hemi_meth ==9) ) ) {
		cout << "Please give a valid seeding method!" << endl;
		return 0;
	}
	
	
	// seeding done
	// now do the hemisphere association

	if (dbg > 0) {
		cout << endl;
		cout << " Association method = " << hemi_meth << endl;
	}

	bool I_Move = true;

	// iterate to associate all objects to hemispheres (methods 1 to 3 only)
	//   until no objects are moved from one to the other hemisphere
	//   or the maximum number of iterations is reached
	while (I_Move && (numLoop < nItermax) && hemi_meth != 8 && hemi_meth !=9){

		I_Move = false;
		numLoop++;
		if (dbg > 0) {
			cout << " Iteration = " << numLoop << endl;
		}
		if(numLoop == nItermax-1){
			cout << " Hemishpere: warning - reaching max number of iterations " << endl;
		}

	// initialize the current sums of Px, Py, Pz, E for the two hemispheres
		float Sum1_Px = 0.;
		float Sum1_Py = 0.;
		float Sum1_Pz = 0.;
		float Sum1_E = 0.;
		float Sum2_Px = 0.;
		float Sum2_Py = 0.;
		float Sum2_Pz = 0.;
		float Sum2_E = 0.;

	// associate the objects for method 1
		if (hemi_meth == 1) {

			for (int i = 0; i < vsize; ++i){
				if (Object_Noassoc[i] == 0){
					float  P_Long1 = Object_Px[i]*Axis1[0] + Object_Py[i]*Axis1[1] + Object_Pz[i]*Axis1[2];
					float  P_Long2 = Object_Px[i]*Axis2[0] + Object_Py[i]*Axis2[1] + Object_Pz[i]*Axis2[2];
					if (P_Long1 >= P_Long2){
						if (Object_Group[i] != 1){
							I_Move = true;
						}
						Object_Group[i] = 1;
						Sum1_Px += Object_Px[i];
						Sum1_Py += Object_Py[i];
						Sum1_Pz += Object_Pz[i];
						Sum1_E += Object_E[i];
					} else {
						if (Object_Group[i] != 2){
							I_Move = true;
						}
						Object_Group[i] = 2;
						Sum2_Px += Object_Px[i];
						Sum2_Py += Object_Py[i];
						Sum2_Pz += Object_Pz[i];
						Sum2_E += Object_E[i];
					}
				}
			}

	// associate the objects for methods 2 and 3
		} else if (hemi_meth == 2 || hemi_meth == 3){

			for (int i = 0; i < vsize; ++i){
		// add the seeds to the sums, as they remain fixed
				if (i == I_Max) {
					Object_Group[i] = 1;
					Sum1_Px += Object_Px[i];
					Sum1_Py += Object_Py[i];
					Sum1_Pz += Object_Pz[i];
					Sum1_E += Object_E[i];
				} else if (i == J_Max) {
					Object_Group[i] = 2;
					Sum2_Px += Object_Px[i];
					Sum2_Py += Object_Py[i];
					Sum2_Pz += Object_Pz[i];
					Sum2_E += Object_E[i];

		// for the other objects
				} else {
					if (Object_Noassoc[i] == 0){

			// only 1 object maximum is moved in a given iteration
						if(!I_Move){
			// initialize the new hemispheres as the current ones
							float NewAxis1_Px = Axis1[0] * Axis1[3];
							float NewAxis1_Py = Axis1[1] * Axis1[3];
							float NewAxis1_Pz = Axis1[2] * Axis1[3];
							float NewAxis1_E = Axis1[4];
							float NewAxis2_Px = Axis2[0] * Axis2[3];
							float NewAxis2_Py = Axis2[1] * Axis2[3];
							float NewAxis2_Pz = Axis2[2] * Axis2[3];
							float NewAxis2_E = Axis2[4];
				// subtract the object from its hemisphere
							if (Object_Group[i] == 1){
							        NewAxis1_Px = NewAxis1_Px - Object_Px[i];
								NewAxis1_Py = NewAxis1_Py - Object_Py[i];
								NewAxis1_Pz = NewAxis1_Pz - Object_Pz[i];
								NewAxis1_E = NewAxis1_E - Object_E[i];
							} else if (Object_Group[i] == 2) {
								NewAxis2_Px = NewAxis2_Px - Object_Px[i];
								NewAxis2_Py = NewAxis2_Py - Object_Py[i];
								NewAxis2_Pz = NewAxis2_Pz - Object_Pz[i];
								NewAxis2_E = NewAxis2_E - Object_E[i];
							}

			// compute the invariant mass squared with each hemisphere (method 2)
							float mass1 =  NewAxis1_E
								- ((Object_Px[i]*NewAxis1_Px + Object_Py[i]*NewAxis1_Py
								+ Object_Pz[i]*NewAxis1_Pz) / Object_P[i]);
							float mass2 =  NewAxis2_E
								- ((Object_Px[i]*NewAxis2_Px + Object_Py[i]*NewAxis2_Py
								+ Object_Pz[i]*NewAxis2_Pz) / Object_P[i]);
			// or the Lund distance (method 3)
							if (hemi_meth == 3) {
								mass1 *= NewAxis1_E/((NewAxis1_E+Object_E[i])*(NewAxis1_E+Object_E[i]));
								mass2 *= NewAxis2_E/((NewAxis2_E+Object_E[i])*(NewAxis2_E+Object_E[i]));
							}
			// and associate the object to the best hemisphere and add it to the sum
							if (mass1 < mass2) {
							  //if (Object_Group[i] != 1){
							        if (Object_Group[i] != 1 && Object_Group[i] != 0){
									I_Move = true;
								}
								Object_Group[i] = 1;
								Sum1_Px += Object_Px[i];
								Sum1_Py += Object_Py[i];
								Sum1_Pz += Object_Pz[i];
								Sum1_E += Object_E[i];
							} else {
							  //if (Object_Group[i] != 2){
							        if (Object_Group[i] != 2 && Object_Group[i] != 0){
									I_Move = true;
								}
								Object_Group[i] = 2;
								Sum2_Px += Object_Px[i];
								Sum2_Py += Object_Py[i];
								Sum2_Pz += Object_Pz[i];
								Sum2_E += Object_E[i];
							}

				// but if a previous object was moved, add all other associated objects to the sum
						} else {
							if (Object_Group[i] == 1){
								Sum1_Px += Object_Px[i];
								Sum1_Py += Object_Py[i];
								Sum1_Pz += Object_Pz[i];
								Sum1_E += Object_E[i];
							} else if (Object_Group[i] == 2){
								Sum2_Px += Object_Px[i];
								Sum2_Py += Object_Py[i];
								Sum2_Pz += Object_Pz[i];
								Sum2_E += Object_E[i];
							}
						}
					}
				} // end loop over objects, Sum1_ and Sum2_ are now the updated hemispheres
			}

		} else {
			cout << "Please give a valid hemisphere association method!" << endl;
			return 0;
		}

	// recomputing the axes for next iteration

		Axis1[3] = sqrt(Sum1_Px*Sum1_Px + Sum1_Py*Sum1_Py + Sum1_Pz*Sum1_Pz);
		if (Axis1[3] < 0.0001) {
			cout << "ZERO objects in group 1! " << endl;
		} else {
			Axis1[0] = Sum1_Px / Axis1[3];
			Axis1[1] = Sum1_Py / Axis1[3];
			Axis1[2] = Sum1_Pz / Axis1[3];
			Axis1[4] = Sum1_E;
		}
		Axis2[3] = sqrt(Sum2_Px*Sum2_Px + Sum2_Py*Sum2_Py + Sum2_Pz*Sum2_Pz);
		if (Axis2[3] < 0.0001) {
			cout << " ZERO objects in group 2! " << endl;
		} else {
			Axis2[0] = Sum2_Px / Axis2[3];
			Axis2[1] = Sum2_Py / Axis2[3];
			Axis2[2] = Sum2_Pz / Axis2[3];
			Axis2[4] = Sum2_E;
		}

		if (dbg > 0) {
			cout << " Grouping = ";
			for (int i=0;i<vsize;i++){
				cout << "  " << Object_Group[i];
			}
			cout << endl;
		}

		if (numLoop <= 1) I_Move = true;
		
	} // end of iteration

	// associate all objects to hemispheres for method 8
	if (hemi_meth == 8 || hemi_meth == 9) {
	        float sumtot = 0.;
		for (int i = 0; i < vsize; ++i){
		        if (Object_Noassoc[i] != 0) continue;
			if(hemi_meth==8){ sumtot += Object_E[i]; }
			else sumtot += Object_Pt[i];
		}
		float sumdiff = fabs(sumtot - 2*Object_E[0]);
                float sum1strt = 0.;
		int ibest = 0, jbest = 0;

		// start by choosing object 0 in hemi 1, all others in hemi 2
		// then add each of the other objects one by one to hemi 1
		// then add object 1 to hemi one and add each of the others in turn, etc
		if(vsize > 2){
			for (int i = 0; i < vsize-1; ++i){
				 if (Object_Noassoc[i] != 0) continue;
				 if (hemi_meth ==8 ){sum1strt += Object_E[i];}
				 else               {sum1strt += Object_Pt[i];}
				 for (int j = i+1; j < vsize; ++j){
					  if (Object_Noassoc[j] != 0) continue;
					  float sum1_E =0;
					  if (hemi_meth ==8 ){
						  sum1_E = sum1strt + Object_E[j];
					  }else {
						  sum1_E = sum1strt + Object_Pt[j];
					  }
					  float sum2_E = sumtot - sum1_E;
					  if(sumdiff >= fabs(sum1_E - sum2_E)) {
						   sumdiff = fabs(sum1_E - sum2_E);
						   ibest = i;
						   jbest = j;
					  }
				 }
		       }
	       }
	    
	       // then store the best combination into the hemisphere axes
	       float Sum1_Px=0, Sum1_Py=0, Sum1_Pz=0, Sum1_E=0;
	       float Sum2_Px=0, Sum2_Py=0, Sum2_Pz=0, Sum2_E=0;
               for (int i = 0; i < vsize; ++i){
		         if (Object_Noassoc[i] != 0) Object_Group[i] = 0;
		         else if (i <= ibest || i == jbest) {
			          Sum1_Px += Object_Px[i];
				  Sum1_Py += Object_Py[i];
				  Sum1_Pz += Object_Pz[i];
				  Sum1_E += Object_E[i];
				  Object_Group[i] = 1;
			 } else {
			          Sum2_Px += Object_Px[i];
				  Sum2_Py += Object_Py[i];
				  Sum2_Pz += Object_Pz[i];
				  Sum2_E += Object_E[i];
				  Object_Group[i] = 2;
			 }
 	       }
	       Axis1[3] = sqrt(Sum1_Px*Sum1_Px + Sum1_Py*Sum1_Py + Sum1_Pz*Sum1_Pz);
	       Axis1[0] = Sum1_Px / Axis1[3];
	       Axis1[1] = Sum1_Py / Axis1[3];
	       Axis1[2] = Sum1_Pz / Axis1[3];
	       Axis1[4] = Sum1_E;
	       Axis2[3] = sqrt(Sum2_Px*Sum2_Px + Sum2_Py*Sum2_Py + Sum2_Pz*Sum2_Pz);
	       Axis2[0] = Sum2_Px / Axis2[3];
	       Axis2[1] = Sum2_Py / Axis2[3];
	       Axis2[2] = Sum2_Pz / Axis2[3];
	       Axis2[4] = Sum2_E;
	}

	status = 1;
	return 1;
}

int Hemisphere::RejectISR(){
// tries to avoid including ISR jets into hemispheres
//

	// iterate to remove all ISR objects from the hemispheres
	//   until no ISR objects are found
//   cout << " entered RejectISR() with rejectISRDR = " << rejectISRDR << endl;
	bool I_Move = true;
	while (I_Move) {
		I_Move = false;
		float valmax = 0.;
		int imax = -1;

	// start by making a hemisphere reconstruction
		int hemiOK = Reconstruct();
		if (hemiOK == 0) {return 0;}

	// convert the axes into Px, Py, Pz, E vectors
		float newAxis1_Px = Axis1[0] * Axis1[3];
		float newAxis1_Py = Axis1[1] * Axis1[3];
		float newAxis1_Pz = Axis1[2] * Axis1[3];
		//float newAxis1_E = Axis1[4];
		float newAxis2_Px = Axis2[0] * Axis2[3];
		float newAxis2_Py = Axis2[1] * Axis2[3];
		float newAxis2_Pz = Axis2[2] * Axis2[3];
		//float newAxis2_E = Axis2[4];

	// loop over all objects associated to a hemisphere
		int vsize = (int) Object_Px.size();
		for (int i = 0; i < vsize; ++i){
			if (Object_Group[i] == 1 || Object_Group[i] == 2){
//         cout << "  Object = " << i << ", Object_Group = " << Object_Group[i] << endl;

			// collect the hemisphere data
				float newPx = 0.;
				float newPy = 0.;
				float newPz = 0.;
				//float newE = 0.;
				if (Object_Group[i] == 1){
					newPx = newAxis1_Px;
					newPy = newAxis1_Py;
					newPz = newAxis1_Pz;
					//newE = newAxis1_E;
				} else if (Object_Group[i] == 2) {
					newPx = newAxis2_Px;
					newPy = newAxis2_Py;
					newPz = newAxis2_Pz;
					//newE = newAxis2_E;
				}

		// compute the quantities to test whether the object is ISR
				float ptHemi = 0.;
				float hemiEta = 0.;
				float hemiPhi = 0.;
				if (rejectISRPt == 1) {
					float plHemi = (Object_Px[i]*newPx + Object_Py[i]*newPy
						+ Object_Pz[i]*newPz)
						/ sqrt(newPx*newPx+newPy*newPy+newPz*newPz);
					float pObjsq = Object_Px[i]*Object_Px[i] + Object_Py[i]*Object_Py[i]
						+ Object_Pz[i]*Object_Pz[i];
					ptHemi = sqrt(pObjsq - plHemi*plHemi);
					if (ptHemi > valmax) {
						valmax = ptHemi;
						imax = i;
					}
				} else if (rejectISRDR == 1) {
					float theta = 1.570796327;
			// compute the new hemisphere eta, phi and DeltaR
					float pdiff = fabs(newPx-Object_Px[i]) + fabs(newPy-Object_Py[i])
						+ fabs(newPz-Object_Pz[i]);
					if (pdiff > 0.001) {
						if (fabs(newPz) > 0.001) {
							theta = atan(sqrt(newPx*newPx+newPy*newPy)/newPz);
						}
						if (theta < 0.) {theta = theta + 3.141592654;}
						hemiEta = -log(tan(0.5*theta));
						hemiPhi = atan2(newPy, newPx);
						//						float DeltaR = sqrt((Object_Eta[i] - hemiEta)*(Object_Eta[i] - hemiEta)
						//							+ (Util::DeltaPhi(Object_Phi[i], hemiPhi))*(Util::DeltaPhi(Object_Phi[i], hemiPhi)) );
						float DeltaR = sqrt((Object_Eta[i] - hemiEta)*(Object_Eta[i] - hemiEta)
							+ (deltaPhi(Object_Phi[i], hemiPhi))*(deltaPhi(Object_Phi[i], hemiPhi)) );
//             cout << "  Object = " << i << ", DeltaR = " << DeltaR << endl;
						if (DeltaR > valmax) {
							valmax = DeltaR;
							imax = i;
						}
					}
				}
			}
		} // end loop over objects

	// verify whether the ISR tests are fulfilled
		if (imax < 0) {
			I_Move = false;
		} else if (rejectISRPt == 1 && valmax > rejectISRPtmax) {
			SetNoAssoc(imax);
			I_Move = true;
		} else if (rejectISRDR == 1 && valmax > rejectISRDRmax) {
			SetNoAssoc(imax);
			I_Move = true;
		}

	} // end iteration over ISR objects

	status = 1;
	return 1;
}

}
